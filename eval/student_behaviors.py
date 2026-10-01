"""Fresh, unprivileged student measurements on the exact teacher-study cohort.

One invocation evaluates one base-model family. Repeat --run NAME=TRAINING_DIR
for answer/hint/full/solution runs (rollout is excluded). In best mode also pass
--benchmark-results NAME=RESULT_DIRECTORY for each run; each directory contains
checkpoint-N/{summary.json,results.json} from eval.run_eval.

Example (add further --run / --benchmark-results pairs as needed)::

    python -m eval.student_behaviors --phase prepare \
      --teacher-study-dir results/teacher_uncertainty/default_hint/Qwen_Qwen3-1.7B \
      --run hint=/mnt/data/ujan/self-distill/outputs/sdft/Qwen3-1.7B/deepmath_hint \
      --benchmark-results hint=results/aime24/deepmath/Qwen3-1.7B/sdft/hint/run-1 \
      --checkpoint-selection best --selection-benchmark aime24 \
      --output-dir results/student_behaviors/Qwen3-1.7B

Then run --phase sweep with the same selection flags and output directory; the
prepared plan supplies all input paths. --phase generate runs each pending solver
in a fresh process; --phase classify keeps one judge loaded across all jobs;
--phase summarize only reads artifacts and creates summaries and plots on CPU.
Generation settings default to the source teacher study. Eight generated samples
and four classified samples match the current teacher experiment.

Selections are frozen on prepare. Use --refresh-selection with the original run
arguments to discover newly evaluated/saved checkpoints. all and best selections
share checkpoint caches. --dry-run plans without writing files or loading models.
No PI is inserted into any student prompt, including the shared base baseline.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing
import os
import re
from collections import defaultdict
from importlib.metadata import version
from pathlib import Path

from eval.hint_compare_cache import ConditionCache, digest, model_identity

ARMS = ("answer", "hint", "full", "solution")
BEHAVIORS = ("verification", "backtracking", "subgoal_setting", "backward_chaining")
SCHEMA_VERSION = 1


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temp, path)


def read_rows(path):
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temp.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, allow_nan=False) + "\n")
    os.replace(temp, path)


def named_paths(values):
    result = {}
    for value in values or []:
        name, sep, path = value.partition("=")
        if (
            not sep
            or not path
            or not re.fullmatch(r"[\w.-]+", name)
            or name in ("base", ".", "..")
        ):
            raise ValueError("Use a unique NAME=PATH; 'base' is reserved")
        if name in result:
            raise ValueError(f"Duplicate run name: {name}")
        result[name] = str(Path(path).expanduser().resolve())
    return result


def load_teacher_cohort(study_dir, cohort_dir=None):
    """Recover the retained IDs, never reapply or relax teacher PI filters."""
    study_dir = Path(study_dir).resolve()
    meta = read_json(study_dir / "teacher_uncertainty_run_meta.json")
    source = meta["source"]
    recorded = source.get("cohort")
    if not recorded:
        raise ValueError(
            "Teacher study needs recorded cohort provenance and question IDs"
        )
    root = Path(cohort_dir or recorded["cohort_dir"])
    manifest = read_json(root / "manifest.json")
    rows = read_rows(root / "cohort.jsonl")
    if (
        digest(rows) != manifest["cohort_hash"]
        or manifest["cohort_hash"] != recorded["cohort_hash"]
    ):
        raise ValueError("Teacher source cohort checksum mismatch")
    if (
        manifest["model"] != source["problem_model"]
        or meta["teacher_model"] != source["problem_model"]
    ):
        raise ValueError("Use a self-teacher study for the student's base model")
    if manifest["tokenizer_hash"] != recorded["tokenizer_hash"]:
        raise ValueError("Teacher source tokenizer mismatch")
    ids, indices = source["question_ids"], source["question_indices"]
    if not ids or len(ids) != len(set(ids)) or len(indices) != len(set(indices)):
        raise ValueError("Teacher question identities must be nonempty and unique")
    by_id = {r["question_id"]: r for r in rows}
    if len(by_id) != len(rows) or len(ids) != meta["n_problems"]:
        raise ValueError("Invalid teacher cohort size or duplicate source identities")
    selected = []
    for qid, idx in zip(ids, indices, strict=True):
        row = by_id.get(qid)
        if row is None or row["question_idx"] != idx:
            raise ValueError(f"Teacher question missing or misaligned: {qid}")
        selected.append(
            {
                "question_id": qid,
                "question_idx": idx,
                "question": row["question"],
                "answer": str(row["final_answer"]),
                "dataset": manifest["dataset"],
            }
        )
    return selected, meta


def checkpoint_dirs(run_dir):
    return {
        int(p.name[11:]): p
        for p in Path(run_dir).iterdir()
        if p.is_dir() and re.fullmatch(r"checkpoint-[0-9]+", p.name)
    }


def require_weights(path):
    path = Path(path)
    if (
        not path.is_dir()
        or not any(path.glob("*.safetensors"))
        and not any(path.glob("pytorch_model*.bin"))
    ):
        raise ValueError(f"Selected checkpoint weights unavailable: {path}")


def benchmark_candidate(path, run_dir, meta, benchmark, allow_legacy=False):
    """Validate avg@16 and recover the actual benchmark question set and protocol."""
    summary = read_json(path)
    config, arm = summary.get("eval_config", {}), summary.get("arm", {})
    legacy = not config
    if legacy and not allow_legacy:
        raise ValueError(
            "Legacy result lacks eval_config; use --allow-legacy-benchmark-results to record unverified generation settings"
        )
    if summary.get("n_samples") != 16 or (not legacy and config.get("n") != 16):
        raise ValueError("avg@16 requires n_samples=16 and eval_config.n=16")
    score = summary.get("pass_at_k", {}).get("pass@1")
    if (
        not isinstance(score, (int, float))
        or isinstance(score, bool)
        or not math.isfinite(score)
        or not 0 <= score <= 1
    ):
        raise ValueError("Missing or invalid pass_at_k.pass@1")
    if not legacy and config.get("eval_dataset") != benchmark:
        raise ValueError("Benchmark identity differs")
    expected = {
        "algo": "sdft",
        "model": meta["model"].split("/")[-1],
        "train_dataset": meta["dataset"],
        "step": path.parent.name,
    }
    if (arm or not legacy) and any(arm.get(k) != v for k, v in expected.items()):
        raise ValueError("Benchmark arm metadata differs from the training run")
    recorded_path = Path(summary["model"]).expanduser().resolve()
    expected_path = (Path(run_dir) / path.parent.name).resolve()
    if recorded_path != expected_path:
        raise ValueError("Benchmark model path differs from the explicit training run")
    rows = read_json(path.parent / "results.json")
    if not rows or len(rows) != summary.get("dataset_size"):
        raise ValueError("Benchmark question count differs from summary")
    questions = []
    for r in rows:
        if r.get("n_samples") != 16 or len(r.get("samples", [])) != 16:
            raise ValueError("Incomplete benchmark samples")
        if not 0 <= r["n_correct"] <= 16 or r["n_correct"] != sum(
            bool(s["correct"]) for s in r["samples"]
        ):
            raise ValueError("Benchmark correctness count differs from saved samples")
        question = r.get("problem", r.get("question"))
        if not isinstance(question, str) or not question.strip():
            raise ValueError("Benchmark record lacks problem/question text")
        questions.append([question, str(r["answer"])])
    if len({digest(q) for q in questions}) != len(questions):
        raise ValueError("Duplicate benchmark questions")
    actual = sum(r["n_correct"] / 16 for r in rows) / len(rows)
    if not math.isclose(score, actual, abs_tol=1e-12):
        raise ValueError("Stored pass@1 does not equal avg@16 from saved results")
    if not legacy and (not config.get("sampling") or config.get("max_tokens") is None):
        raise ValueError("Benchmark generation protocol is missing")
    protocol = {
        k: config.get(k)
        for k in ("sampling", "max_tokens", "chat_template_model", "seed")
    }
    if legacy:
        protocol["max_tokens"] = summary.get("max_tokens")
    protocol["questions"] = digest(sorted(questions))
    return {
        "step": int(path.parent.name[11:]),
        "score": score,
        "summary_path": str(path.resolve()),
        "summary_hash": digest(summary),
        "protocol": protocol,
        "variant": arm.get("variant"),
        "run": arm.get("run"),
        "provenance_gaps": [
            "benchmark_identity_inferred_from_explicit_directory",
            "generation_settings_unrecorded",
        ]
        if legacy
        else [],
    }


def choose_checkpoints(
    run_dir, meta, mode, benchmark=None, results_dir=None, allow_legacy=False
):
    checkpoints = checkpoint_dirs(run_dir)
    if mode == "all":
        if not checkpoints:
            raise ValueError(f"No saved checkpoints: {run_dir}")
        selected = sorted(checkpoints)
        report = {"selected_steps": selected}
    else:
        if not results_dir:
            raise ValueError(
                "best selection requires --benchmark-results for every run"
            )
        candidates, excluded = [], []
        for path in sorted(Path(results_dir).glob("checkpoint-*/summary.json")):
            if not re.fullmatch(r"checkpoint-[0-9]+", path.parent.name):
                continue
            try:
                candidates.append(
                    benchmark_candidate(path, run_dir, meta, benchmark, allow_legacy)
                )
            except (ValueError, KeyError, OSError, TypeError) as error:
                excluded.append({"path": str(path), "reason": str(error)})
        if not candidates:
            raise ValueError(f"No eligible avg@16 results in {results_dir}: {excluded}")
        if len({digest(c["protocol"]) for c in candidates}) != 1:
            raise ValueError(
                f"Benchmark questions or generation settings differ within {results_dir}"
            )
        if len({(c["variant"], c["run"]) for c in candidates}) != 1:
            raise ValueError(f"Benchmark results mix training runs: {results_dir}")
        candidates.sort(key=lambda c: (-c["score"], c["step"]))
        selected = [candidates[0]["step"]]
        report = {
            "metric": "avg@16",
            "stored_metric": "pass_at_k.pass@1",
            "benchmark": benchmark,
            "tie_break": "earliest_step",
            "candidates": candidates,
            "excluded_results": excluded,
            "without_eligible_results": sorted(
                set(checkpoints) - {c["step"] for c in candidates}
            ),
            "selected_steps": selected,
            "generation_protocol_verified": not any(
                c["provenance_gaps"] for c in candidates
            ),
        }
    # Check AFTER ranking: never silently demote a winner whose weights disappeared.
    for step in selected:
        require_weights(Path(run_dir) / f"checkpoint-{step}")
    return selected, report


def selection_path(args):
    name = args.checkpoint_selection
    if name == "best":
        if not args.selection_benchmark or not re.fullmatch(
            r"[\w-]+", args.selection_benchmark
        ):
            raise ValueError("best requires --selection-benchmark BENCHMARK")
        name += f"-{args.selection_benchmark}"
    return Path(args.output_dir) / f"selection-{name}.json"


def make_plan(args):
    root = Path(args.output_dir)
    if args.teacher_study_dir:
        problems, teacher = load_teacher_cohort(args.teacher_study_dir, args.cohort_dir)
        generation = dict(teacher["generation"])
        generation["n"] = teacher["n_samples"]
        generation.update(min_p=0.0, repetition_penalty=1.0)
        for key in ("n", "max_tokens", "temperature", "top_p", "top_k", "seed"):
            value = getattr(args, key)
            if value is not None:
                generation[key] = value
        base = teacher["source"]["problem_model"]
        experiment = {
            "version": SCHEMA_VERSION,
            "base_model": base,
            "teacher_study_dir": str(Path(args.teacher_study_dir).resolve()),
            "teacher_meta_hash": digest(teacher),
            "problems": problems,
            "cohort_hash": digest(problems),
            "generation": generation,
        }
        if (root / "experiment.json").exists() and read_json(
            root / "experiment.json"
        ) != experiment:
            raise ValueError(
                "Cohort or generation settings changed; use a new output directory"
            )
    else:
        experiment = read_json(root / "experiment.json")
        generation = experiment["generation"]
        for key in ("n", "max_tokens", "temperature", "top_p", "top_k", "seed"):
            if getattr(args, key) is not None and getattr(args, key) != generation[key]:
                raise ValueError(
                    f"--{key.replace('_', '-')} differs from prepared experiment"
                )
    if not (
        generation["n"] >= 1
        and 0 < generation["max_tokens"] < generation["max_model_len"]
        and generation["temperature"] > 0
        and 0 < generation["top_p"] <= 1
        and (generation["top_k"] == -1 or generation["top_k"] >= 1)
    ):
        raise ValueError("Invalid generation configuration")
    if not 1 <= args.samples_per_problem <= generation["n"]:
        raise ValueError("Need 1 <= samples-per-problem <= generated n")
    runs, benchmark_dirs = named_paths(args.run), named_paths(args.benchmark_results)
    path = selection_path(args)
    if path.exists() and not args.refresh_selection:
        plan = read_json(path)
        if plan["experiment_hash"] != digest(experiment):
            raise ValueError("Selection belongs to a different experiment")
        if (
            runs
            and runs != plan["runs"]
            or benchmark_dirs
            and benchmark_dirs != plan["benchmark_results"]
        ):
            raise ValueError("Run mapping changed; use --refresh-selection")
        return experiment, plan
    if not runs:
        raise ValueError(
            "Prepare a selection using at least one --run NAME=TRAINING_DIR"
        )
    if set(benchmark_dirs) - set(runs):
        raise ValueError("Benchmark mapping contains unknown runs")
    jobs = [
        {
            "name": "base",
            "arm": "none",
            "step": 0,
            "model": experiment["base_model"],
            "revision": generation["revision"],
            "identity": {"model": experiment["base_model"]},
            "relative_dir": "base",
        }
    ]
    reports = {}
    for name, run_dir in runs.items():
        meta = read_json(Path(run_dir) / "run_meta.json")
        if (
            meta["model"] != experiment["base_model"]
            or meta["dataset"] != experiment["problems"][0]["dataset"]
        ):
            raise ValueError(
                f"Training model/dataset differs from teacher cohort: {name}"
            )
        if meta["pi_mode"] not in ARMS:
            raise ValueError(
                f"Unsupported training PI {meta['pi_mode']!r}; rollout is excluded"
            )
        steps, reports[name] = choose_checkpoints(
            run_dir,
            meta,
            args.checkpoint_selection,
            args.selection_benchmark,
            benchmark_dirs.get(name),
            args.allow_legacy_benchmark_results,
        )
        for step in steps:
            model = str(Path(run_dir) / f"checkpoint-{step}")
            jobs.append(
                {
                    "name": name,
                    "arm": meta["pi_mode"],
                    "step": step,
                    "model": model,
                    "revision": None,
                    "identity": model_identity(model),
                    "training_meta_hash": digest(meta),
                    "relative_dir": f"{name}/checkpoint-{step}",
                }
            )
    plan = {
        "version": SCHEMA_VERSION,
        "experiment_hash": digest(experiment),
        "mode": args.checkpoint_selection,
        "benchmark": args.selection_benchmark,
        "runs": runs,
        "benchmark_results": benchmark_dirs,
        "selection": reports,
        "jobs": jobs,
    }
    return experiment, plan


def validate_rows(rows, problems, n):
    expected = {
        (p["question_id"], i): p["question_idx"] for p in problems for i in range(n)
    }
    seen = set()
    for r in rows:
        key = (r["question_id"], r["sample_idx"])
        if key in seen or key not in expected or r["question_idx"] != expected[key]:
            raise ValueError("Duplicate, unexpected, or misaligned completion")
        if not r["completion_ids"] or r["n_tokens"] != len(r["completion_ids"]):
            raise ValueError("Invalid generated token count")
        seen.add(key)
    if seen != set(expected):
        raise ValueError("Missing question/sample completions")


def generation_config(experiment, job):
    return {
        "version": SCHEMA_VERSION,
        "model": job["model"],
        "identity": job["identity"],
        "revision": job["revision"],
        "cohort_hash": experiment["cohort_hash"],
        "generation": experiment["generation"],
        "evaluation_pi": "none",
    }


def load_artifact(root, kind, config=None):
    meta_path, data_path = root / f"{kind}_meta.json", root / f"{kind}.jsonl"
    if not meta_path.exists():
        return None
    meta = read_json(meta_path)
    if meta.get("status") != "complete" or not data_path.exists():
        raise ValueError(f"Incomplete committed artifact: {root}/{kind}")
    if config is not None and digest(meta["config"]) != digest(config):
        raise ValueError(
            f"Incompatible {kind} cache: {root}; use a new output directory"
        )
    rows = read_rows(data_path)
    if digest(rows) != meta["rows_hash"]:
        raise ValueError(f"Artifact content checksum mismatch: {data_path}")
    return rows


def save_artifact(root, kind, config, rows):
    write_rows(root / f"{kind}.jsonl", rows)
    write_json(
        root / f"{kind}_meta.json",
        {"status": "complete", "config": config, "rows_hash": digest(rows)},
    )


def render_prompts(problems, tokenizer, max_model_len, max_tokens):
    from utils import format_prompt

    prompts = []
    for problem in problems:
        prompt = tokenizer.apply_chat_template(
            format_prompt(problem["question"], problem["dataset"]),
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        if (
            len(tokenizer(prompt, add_special_tokens=False)["input_ids"]) + max_tokens
            > max_model_len
        ):
            raise ValueError(
                f"Question does not fit frozen cohort budget: {problem['question_id']}"
            )
        prompts.append(prompt)
    return prompts


def generate_job(experiment, job, options):
    """One isolated solver process. Cache each completed prompt's n samples."""
    from transformers import AutoTokenizer

    from eval.demo_gain import tokenizer_hash
    from eval.teacher_uncertainty import measure_completion

    root = Path(options["output_dir"]) / job["relative_dir"]
    config = generation_config(experiment, job)
    if load_artifact(root, "completions", config) is not None:
        return
    if job["step"]:
        require_weights(job["model"])
        if digest(model_identity(job["model"])) != digest(job["identity"]):
            raise ValueError(
                f"Checkpoint files changed since selection: {job['model']}"
            )
    gen = experiment["generation"]
    tokenizer = AutoTokenizer.from_pretrained(
        job["model"], revision=job["revision"], trust_remote_code=True
    )
    if tokenizer_hash(tokenizer) != gen["tokenizer_hash"]:
        raise ValueError(
            "Student tokenizer vocabulary/template differs from teacher study"
        )
    prompts = render_prompts(
        experiment["problems"], tokenizer, gen["max_model_len"], gen["max_tokens"]
    )
    cache_config = {
        **config,
        "vllm_version": version("vllm"),
        "transformers_version": version("transformers"),
    }
    cache = ConditionCache(root, "generation", cache_config)
    rows, pending = [], []
    for problem, prompt in zip(experiment["problems"], prompts, strict=True):
        key = cache.key({"problem": problem, "prompt": prompt})
        cached = cache.load(key)
        if cached is None:
            pending.append((problem, prompt, key))
        else:
            validate_rows(cached, [problem], gen["n"])
            rows.extend(cached)
    if pending:
        from vllm import LLM, SamplingParams

        llm = LLM(
            model=job["model"],
            revision=job["revision"],
            tokenizer_revision=job["revision"],
            dtype=gen["dtype"],
            generation_config="vllm",
            max_model_len=gen["max_model_len"],
            gpu_memory_utilization=options["gpu_memory_utilization"],
            tensor_parallel_size=options["tensor_parallel_size"],
            seed=gen["seed"],
            trust_remote_code=True,
        )
        sampling = SamplingParams(
            **{
                k: gen[k]
                for k in (
                    "n",
                    "max_tokens",
                    "temperature",
                    "top_p",
                    "top_k",
                    "min_p",
                    "repetition_penalty",
                    "seed",
                )
            }
        )
        for start in range(0, len(pending), options["batch_size"]):
            batch = pending[start : start + options["batch_size"]]
            outputs = llm.generate([item[1] for item in batch], sampling)
            for (problem, _, key), output in zip(batch, outputs, strict=True):
                completed = []
                for sample_idx, completion in enumerate(output.outputs):
                    row = measure_completion(
                        completion.text,
                        len(completion.token_ids),
                        completion.finish_reason,
                        problem["answer"],
                        problem["dataset"],
                    )
                    row.update(
                        question_id=problem["question_id"],
                        question_idx=problem["question_idx"],
                        sample_idx=sample_idx,
                        completion_ids=list(completion.token_ids),
                        finish_reason=completion.finish_reason,
                    )
                    completed.append(row)
                validate_rows(completed, [problem], gen["n"])
                cache.save(key, completed)
                rows.extend(completed)
            print(
                f"{job['relative_dir']}: {min(start + len(batch), len(pending))}/{len(pending)} pending questions",
                flush=True,
            )
    order = {p["question_id"]: i for i, p in enumerate(experiment["problems"])}
    rows.sort(key=lambda r: (order[r["question_id"]], r["sample_idx"]))
    validate_rows(rows, experiment["problems"], gen["n"])
    save_artifact(root, "completions", config, rows)
    write_json(root / "generation_runtime.json", cache_config)


def generate_phase(experiment, plan, args):
    # Check every selected path before launching the first expensive worker.
    for job in plan["jobs"]:
        root = Path(args.output_dir) / job["relative_dir"]
        if (
            load_artifact(root, "completions", generation_config(experiment, job))
            is None
            and job["step"]
        ):
            require_weights(job["model"])
            if digest(model_identity(job["model"])) != digest(job["identity"]):
                raise ValueError(f"Checkpoint changed since selection: {job['model']}")
    for job in plan["jobs"]:
        root = Path(args.output_dir) / job["relative_dir"]
        if (
            load_artifact(root, "completions", generation_config(experiment, job))
            is not None
        ):
            continue
        worker = multiprocessing.get_context("spawn").Process(
            target=generate_job, args=(experiment, job, vars(args))
        )
        worker.start()
        worker.join()
        if worker.exitcode:
            raise RuntimeError(
                f"Generation failed for {job['model']} (exit {worker.exitcode}); completed batches are cached"
            )


def judge_args(args):
    return argparse.Namespace(
        classifier_model=args.classifier_model,
        chunk_tokens=args.chunk_tokens,
        context_paragraphs=args.context_paragraphs,
        evidence=args.evidence,
        temperature=args.judge_temperature,
        top_p=args.judge_top_p,
        max_output_tokens=args.judge_max_tokens,
        seed=args.judge_seed,
        max_model_len=args.judge_max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
        max_num_seqs=args.judge_max_num_seqs,
        enable_prefix_caching=True,
    )


def classify_phase(experiment, plan, args):
    from transformers import AutoConfig, AutoTokenizer

    from eval import teacher_behaviors as tb

    judge = judge_args(args)
    # Pin the resolved revision in cached identity, even though the shared loader
    # accepts a model name. Local checkpoints have file-stat provenance as well.
    judge_config = vars(judge).copy()
    for key in (
        "gpu_memory_utilization",
        "tensor_parallel_size",
        "max_num_seqs",
        "enable_prefix_caching",
    ):
        judge_config.pop(key)
    judge_config["identity"] = model_identity(judge.classifier_model)
    judge_config["revision"] = getattr(
        AutoConfig.from_pretrained(judge.classifier_model, trust_remote_code=True),
        "_commit_hash",
        None,
    )
    judge.revision = judge_config["revision"]
    judge_config["rubric"] = tb.rubric_fingerprint()
    judge_config["rubric_version"] = tb.BEHAVIOR_RUBRIC_VERSION
    judge_config["vllm_version"] = version("vllm")
    tokenizer = None
    llm, sampling = None, None
    for job in plan["jobs"]:
        root = Path(args.output_dir) / job["relative_dir"]
        source = load_artifact(root, "completions", generation_config(experiment, job))
        if source is None:
            raise ValueError(f"Generate completions first: {root}")
        validate_rows(source, experiment["problems"], experiment["generation"]["n"])
        selected = [r for r in source if r["sample_idx"] < args.samples_per_problem]
        config = {
            "judge": judge_config,
            "source_hash": digest(selected),
            "samples_per_problem": args.samples_per_problem,
        }
        if not args.reclassify and load_artifact(root, "behaviors", config) is not None:
            continue
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(
                judge.classifier_model,
                revision=judge_config["revision"],
                trust_remote_code=True,
            )
        chunks = tb.build_chunk_plan(
            selected, tokenizer, args.chunk_tokens, args.context_paragraphs
        )
        cache = ConditionCache(root, "classification", config, force=args.reclassify)
        rows, pending = [], []
        for chunk in chunks:
            key = cache.key(chunk)
            cached = cache.load(key)
            if cached is None:
                pending.append((chunk, key))
            else:
                rows.append(cached)
        print(
            f"{job['relative_dir']}: {len(chunks)} judge segments, {len(pending)} pending",
            flush=True,
        )
        if pending and llm is None:
            llm, sampling = tb.create_classifier(judge)
        for start in range(0, len(pending), args.judge_batch_size):
            batch = pending[start : start + args.judge_batch_size]
            scored = tb.classify_chunks(
                llm,
                sampling,
                [c for c, _ in batch],
                tb.render_system_prompt(),
                args.evidence,
            )
            for (_, key), row in zip(batch, scored, strict=True):
                cache.save(key, row)
                rows.append(row)
        rows.sort(key=lambda r: (r["question_idx"], r["sample_idx"], r["chunk_idx"]))
        save_artifact(root, "behaviors", config, rows)


def metric_functions(cognitive=False):
    metrics = {
        "pass@1": (lambda r: r["correct"], lambda r: 1),
        "mean_tokens": (lambda r: r["n_tokens"], lambda r: 1),
        "trunc_rate": (lambda r: r["truncated"], lambda r: 1),
        "unclosed_rate": (lambda r: r["unclosed"], lambda r: 1),
        "mean_e_total": (lambda r: r["e_total"], lambda r: 1),
        "mean_e_think": (lambda r: r["e_think"], lambda r: 1),
        "e_per_1k_tokens": (lambda r: 1000 * r["e_total"], lambda r: r["n_tokens"]),
    }
    if cognitive:
        for behavior in BEHAVIORS:
            metrics[f"{behavior}/rate_per_1k"] = (
                lambda r, b=behavior: 1000 * r[b],
                lambda r: r["n_tokens"],
            )
            metrics[f"{behavior}/mean_per_trajectory"] = (
                lambda r, b=behavior: r[b],
                lambda r: 1,
            )
            metrics[f"{behavior}/prevalence"] = (
                lambda r, b=behavior: r[b] > 0,
                lambda r: 1,
            )
    else:
        metrics["mean_e_post"] = (lambda r: r["e_post"], lambda r: 1)
    return metrics


def bootstrap_metrics(rows, samples, seed, baseline=None, n=None, cognitive=False):
    """Question-clustered pooled ratios, optionally paired on complete sample groups."""
    import numpy as np

    def group(records):
        out = defaultdict(list)
        for r in records:
            out[r["question_idx"]].append(r)
        if baseline is not None:
            out = {
                q: rs
                for q, rs in out.items()
                if len(rs) == n and {r["sample_idx"] for r in rs} == set(range(n))
            }
        return out

    grouped = group(rows)
    other = group(baseline) if baseline is not None else None
    questions = sorted(set(grouped) & set(other) if other is not None else grouped)
    result = {
        "n_questions": len(questions),
        "question_indices": questions,
        "uncertainty_unit": "paired_question_bootstrap"
        if other is not None
        else "question_bootstrap",
        "metrics": {},
    }
    if not questions:
        return result
    draws = np.random.default_rng(seed).integers(
        len(questions), size=(samples, len(questions))
    )
    for name, (num, den) in metric_functions(cognitive).items():
        values, distributions = [], []
        for groups in [grouped, other] if other is not None else [grouped]:
            totals = np.array(
                [
                    [sum(num(r) for r in groups[q]), sum(den(r) for r in groups[q])]
                    for q in questions
                ],
                dtype=float,
            )
            values.append(totals[:, 0].sum() / totals[:, 1].sum())
            sums = totals[draws].sum(axis=1)
            distributions.append(sums[:, 0] / sums[:, 1])
        value = values[0] - values[1] if other is not None else values[0]
        distribution = (
            distributions[0] - distributions[1]
            if other is not None
            else distributions[0]
        )
        result["metrics"][name] = {
            "delta" if other is not None else "mean": float(value),
            "ci95": np.quantile(distribution, [0.025, 0.975]).tolist(),
        }
    return result


def summarize_phase(experiment, plan, args):
    from eval import teacher_behaviors as tb
    from eval.teacher_uncertainty import summarize

    output, sources, trajectories = {}, {}, {}
    common_judge = None
    for job in plan["jobs"]:
        key = job["relative_dir"]
        root = Path(args.output_dir) / key
        source = load_artifact(root, "completions", generation_config(experiment, job))
        if source is None:
            raise ValueError(f"Missing completions: {root}")
        validate_rows(source, experiment["problems"], experiment["generation"]["n"])
        sources[key] = source
        item = {
            "job": job,
            "uncertainty": summarize(source),
            "uncertainty_intervals": bootstrap_metrics(
                source, args.bootstrap_samples, args.bootstrap_seed
            ),
        }
        chunks = load_artifact(root, "behaviors")
        if chunks is not None:
            config = read_json(root / "behaviors_meta.json")["config"]
            selected = [r for r in source if r["sample_idx"] < args.samples_per_problem]
            if (
                config["source_hash"] != digest(selected)
                or config["samples_per_problem"] != args.samples_per_problem
            ):
                raise ValueError(
                    f"Classification source/sample selection changed: {root}"
                )
            if config["judge"]["rubric"] != tb.rubric_fingerprint():
                raise ValueError("Current rubric differs from saved classification")
            if common_judge is not None and config["judge"] != common_judge:
                raise ValueError(
                    "Selected checkpoints used different judges/configurations"
                )
            common_judge = config["judge"]
            collapsed = tb.collapse_to_trajectories(chunks, selected)
            trajectories[key] = collapsed
            behavior = tb.summarize_arm(
                collapsed, chunks, args.bootstrap_samples, args.bootstrap_seed
            )
            behavior["n_trajectories_dropped"] = len(selected) - len(collapsed)
            behavior["n_source_trajectories"] = len(selected)
            item["cognitive"] = behavior
            item["cognitive_intervals"] = bootstrap_metrics(
                collapsed, args.bootstrap_samples, args.bootstrap_seed, cognitive=True
            )
            write_rows(root / "trajectories.jsonl", collapsed)
        output[key] = item
    for key, item in output.items():
        item["uncertainty_vs_base"] = bootstrap_metrics(
            sources[key],
            args.bootstrap_samples,
            args.bootstrap_seed,
            baseline=sources["base"],
            n=experiment["generation"]["n"],
        )
        if key in trajectories and "base" in trajectories:
            item["cognitive_vs_base"] = bootstrap_metrics(
                trajectories[key],
                args.bootstrap_samples,
                args.bootstrap_seed,
                baseline=trajectories["base"],
                n=args.samples_per_problem,
                cognitive=True,
            )
        write_json(Path(args.output_dir) / key / "summary.json", item)
    result = {
        "experiment": experiment,
        "selection": plan,
        "judge": common_judge,
        "samples_per_problem_classified": args.samples_per_problem,
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.bootstrap_seed,
        "jobs": output,
        "missing_classifications": sorted(set(sources) - set(trajectories)),
    }
    result["teacher_references"] = teacher_references(experiment, args, common_judge)
    stem = selection_path(args).stem.replace("selection-", "summary-")
    write_json(Path(args.output_dir) / f"{stem}.json", result)
    if not args.no_plots:
        plot_results(result, Path(args.output_dir) / f"{stem}.png")
    return result


def teacher_references(experiment, args, student_judge):
    """Read matched teacher references; missing optional judgments are explicit.

    Teacher samples are independent of student samples even on matching questions.
    We do not attach paired trajectory interpretations to these reference lines.
    """
    from eval import teacher_behaviors as tb
    from eval.teacher_uncertainty import summarize

    study = Path(experiment["teacher_study_dir"])
    if not study.exists():
        return {
            "status": "unavailable",
            "reason": f"Teacher directory missing: {study}",
        }
    meta = read_json(study / "teacher_uncertainty_run_meta.json")
    if digest(meta) != experiment["teacher_meta_hash"]:
        raise ValueError("Teacher study metadata changed after preparation")
    gen = experiment["generation"]
    same_protocol = all(
        meta["generation"].get(k) == gen.get(k)
        for k in (
            "max_tokens",
            "temperature",
            "top_p",
            "top_k",
            "seed",
            "enable_thinking",
            "dtype",
        )
    )
    same_protocol = same_protocol and meta["n_samples"] == gen["n"]
    behavior_dir = (
        Path(args.teacher_behaviors_dir) if args.teacher_behaviors_dir else None
    )
    if behavior_dir is None:
        parts = list(study.parts)
        if "teacher_uncertainty" in parts:
            parts[parts.index("teacher_uncertainty")] = "teacher_behaviors"
            behavior_dir = Path(*parts)
    result = {"same_generation_protocol": same_protocol, "arms": {}}
    expected = {
        (p["question_idx"], p["question_id"], i)
        for p in experiment["problems"]
        for i in range(meta["n_samples"])
    }
    for arm in ("none", *ARMS):
        source_path = study / f"completions_{arm}.jsonl"
        if not source_path.exists():
            result["arms"][arm] = {"status": "missing_completions"}
            continue
        source = read_rows(source_path)
        identities = [
            (r["question_idx"], r["question_id"], r["sample_idx"]) for r in source
        ]
        if len(identities) != len(expected) or set(identities) != expected:
            raise ValueError(
                f"Teacher reference question/sample identities differ: {source_path}"
            )
        item = {"uncertainty": summarize(source), "cognitive_status": "unavailable"}
        meta_path = (
            behavior_dir / f"behaviors_meta_{arm}.json" if behavior_dir else None
        )
        if meta_path and meta_path.exists():
            saved = read_json(meta_path)
            config = saved["config"]
            selected = [r for r in source if r["sample_idx"] < args.samples_per_problem]
            comparisons = {
                "classifier_model": "classifier_model",
                "chunk_tokens": "chunk_tokens",
                "context_paragraphs": "context_paragraphs",
                "evidence": "evidence",
                "temperature": "temperature",
                "top_p": "top_p",
                "seed": "seed",
                "max_output_tokens": "max_output_tokens",
                "rubric_fingerprint": "rubric",
            }
            matched = student_judge is not None and all(
                config.get(k) == student_judge.get(v) for k, v in comparisons.items()
            )
            matched = (
                matched
                and config.get("samples_per_problem") == args.samples_per_problem
                and config.get("limit") is None
            )
            if saved.get("status") != "complete" or config.get(
                "source_fingerprint"
            ) != tb.source_fingerprint(selected):
                item["cognitive_status"] = "source_provenance_mismatch"
            elif not matched:
                item["cognitive_status"] = "judge_configuration_mismatch"
            else:
                chunks = read_rows(behavior_dir / f"behaviors_{arm}.jsonl")
                trajectories = tb.collapse_to_trajectories(chunks, selected)
                item["cognitive"] = tb.summarize_arm(
                    trajectories, chunks, args.bootstrap_samples, args.bootstrap_seed
                )
                item["cognitive"]["n_trajectories_dropped"] = len(selected) - len(
                    trajectories
                )
                item["cognitive_status"] = "available"
                item["judge_revision_verified"] = (
                    False  # Legacy teacher artifacts do not pin a judge revision.
                )
        result["arms"][arm] = item
    return result


def plot_results(result, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    jobs = result["jobs"]
    metrics = [
        ("uncertainty", m)
        for m in (
            "pass@1",
            "mean_tokens",
            "e_per_1k_tokens",
            "trunc_rate",
            "unclosed_rate",
        )
    ]
    metrics += [
        ("cognitive", f"{b}/{m}")
        for b in BEHAVIORS
        for m in ("rate_per_1k", "mean_per_trajectory", "prevalence")
    ]
    fig, axes = plt.subplots(6, 3, figsize=(16, 23), constrained_layout=True)
    names = sorted({item["job"]["name"] for item in jobs.values()} - {"base"})
    best = result["selection"]["mode"] == "best"
    for ax, (section, metric) in zip(axes.flat, metrics):
        base = (
            jobs["base"].get(section + "_intervals", {}).get("metrics", {}).get(metric)
        )
        if base:
            ax.axhline(base["mean"], color="gray", linestyle="--", label="Base")
            ax.axhspan(*base["ci95"], color="gray", alpha=0.1)
        for i, name in enumerate(names):
            points = sorted(
                [v for v in jobs.values() if v["job"]["name"] == name],
                key=lambda v: v["job"]["step"],
            )
            points = [
                p
                for p in points
                if metric in p.get(section + "_intervals", {}).get("metrics", {})
            ]
            if not points:
                continue
            xs = [i] if best else [p["job"]["step"] for p in points]
            stats = [p[section + "_intervals"]["metrics"][metric] for p in points]
            ys = [s["mean"] for s in stats]
            (line,) = ax.plot(xs, ys, "o-", label=name)
            for x, s in zip(xs, stats, strict=True):
                ax.vlines(x, *s["ci95"], alpha=0.6)
            if best:
                score = result["selection"]["selection"][name]["candidates"][0]["score"]
                ax.annotate(
                    f"step {points[0]['job']['step']}\navg@16 {100 * score:.1f}%",
                    (xs[0], ys[0]),
                    xytext=(0, 7),
                    textcoords="offset points",
                    fontsize=7,
                )
            references = result.get("teacher_references", {})
            teacher = references.get("arms", {}).get(points[0]["job"]["arm"], {})
            if references.get("same_generation_protocol") and section in teacher:
                value = teacher[section]
                for part in metric.split("/"):
                    value = value[part]
                if best:
                    ax.plot(
                        [i], [value], marker="x", color=line.get_color(), markersize=8
                    )
                else:
                    ax.axhline(value, color=line.get_color(), linestyle=":", alpha=0.5)
        ax.set_title(metric)
        if best:
            ax.set_xticks(range(len(names)), names, rotation=25)
        else:
            ax.set_xlabel("Training step")
        ax.grid(alpha=0.2)
    axes.flat[0].legend(fontsize=8)
    for ax in list(axes.flat)[len(metrics) :]:
        ax.set_visible(False)
    fig.suptitle(
        f"{result['experiment']['base_model']} — unprivileged student behaviors ({result['selection']['mode']})\n"
        "Matched teacher references: × (best) / dotted lines (all), when available"
    )
    fig.savefig(path, dpi=160)
    plt.close(fig)


def build_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--phase",
        choices=("prepare", "generate", "classify", "summarize", "sweep"),
        default="sweep",
    )
    p.add_argument("--teacher-study-dir")
    p.add_argument(
        "--cohort-dir",
        help="Override relocated source cohort directory; checksum must match",
    )
    p.add_argument(
        "--teacher-behaviors-dir",
        help="Optional saved teacher judgments; inferred for the standard layout",
    )
    p.add_argument("--run", action="append", help="NAME=TRAINING_DIR (repeat per run)")
    p.add_argument(
        "--benchmark-results",
        action="append",
        help="NAME=RESULT_DIRECTORY (repeat per run)",
    )
    p.add_argument("--checkpoint-selection", choices=("all", "best"), default="all")
    p.add_argument("--selection-benchmark")
    p.add_argument(
        "--allow-legacy-benchmark-results",
        action="store_true",
        help="Allow summaries without eval_config; verify avg@16/question sets and record unknown generation settings",
    )
    p.add_argument("--refresh-selection", action="store_true")
    p.add_argument(
        "--output-dir",
        required=True,
        help="Dedicated directory for one model and cohort",
    )
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--n", type=int)
    p.add_argument("--max-tokens", type=int)
    p.add_argument("--temperature", type=float)
    p.add_argument("--top-p", type=float)
    p.add_argument("--top-k", type=int)
    p.add_argument("--seed", type=int)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--samples-per-problem", type=int, default=4)
    p.add_argument("--classifier-model", default="Qwen/Qwen3.8-27B")
    p.add_argument("--chunk-tokens", type=int, default=1000)
    p.add_argument("--context-paragraphs", type=int, default=0)
    p.add_argument("--evidence", action="store_true")
    p.add_argument(
        "--reclassify",
        action="store_true",
        help="Replace saved judgments; keep solver generations",
    )
    p.add_argument("--judge-temperature", type=float, default=0.7)
    p.add_argument("--judge-top-p", type=float, default=1.0)
    p.add_argument("--judge-seed", type=int, default=42)
    p.add_argument("--judge-max-tokens", type=int)
    p.add_argument("--judge-max-model-len", type=int, default=16384)
    p.add_argument("--judge-max-num-seqs", type=int, default=256)
    p.add_argument("--judge-batch-size", type=int, default=256)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--bootstrap-samples", type=int, default=10000)
    p.add_argument("--bootstrap-seed", type=int, default=42)
    p.add_argument("--no-plots", action="store_true")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.judge_max_tokens = args.judge_max_tokens or (2048 if args.evidence else 1024)
    for key in (
        "batch_size",
        "judge_batch_size",
        "chunk_tokens",
        "samples_per_problem",
        "bootstrap_samples",
        "judge_max_tokens",
    ):
        if getattr(args, key) < 1:
            raise ValueError(f"{key} must be positive")
    if args.context_paragraphs < 0:
        raise ValueError("context-paragraphs must be nonnegative")
    experiment, plan = make_plan(args)
    print(
        f"{experiment['base_model']}: {len(experiment['problems'])} fixed teacher questions; {len(plan['jobs'])} models including base"
    )
    for job in plan["jobs"]:
        print(f"  {job['name']} ({job['arm']}), step {job['step']}: {job['model']}")
    for name, report in plan["selection"].items():
        if report.get("candidates"):
            print(
                f"  {name}: {plan['benchmark']} avg@16={report['candidates'][0]['score']:.6f}; missing results={report['without_eligible_results']}; excluded={report['excluded_results']}"
            )
            if not report["generation_protocol_verified"]:
                print(
                    f"  {name}: legacy benchmark generation settings are unverified (recorded in manifest)"
                )
    if args.dry_run:
        print(
            f"Plan only: {len(plan['jobs']) * len(experiment['problems']) * experiment['generation']['n']} generated responses; "
            f"{len(plan['jobs']) * len(experiment['problems']) * args.samples_per_problem} responses to classify"
        )
        return
    write_json(Path(args.output_dir) / "experiment.json", experiment)
    write_json(selection_path(args), plan)
    if args.phase in ("generate", "sweep"):
        generate_phase(experiment, plan, args)
    if args.phase in ("classify", "sweep"):
        classify_phase(experiment, plan, args)
    if args.phase in ("summarize", "sweep"):
        summarize_phase(experiment, plan, args)


if __name__ == "__main__":
    main()
