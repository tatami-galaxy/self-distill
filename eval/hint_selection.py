"""Sample hints, measure sufficiency/transfer, and export one selected hint per question.

See docs/hint_selection.md. Defaults: temperature 1.4, N=8 hints, K=8 teacher
solutions, epsilon=1/8, gamma=6, and four cached unhinted student trajectories.
Generation and HF scoring run in separate processes. Changing epsilon/gamma
only requires the CPU-only select phase. No hint generator is trained.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import multiprocessing
import os
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from eval.hint_compare_cache import ConditionCache, digest, model_identity

VERSION = 1
METHOD = "sampled_hint_selection"


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    os.replace(temporary, path)


def save_artifact(root, name, identity, rows):
    write_json(root / f"{name}.json", {
        "identity": identity, "fingerprint": digest(rows), "rows": rows,
    })


def load_artifact(root, name, identity):
    data = read_json(root / f"{name}.json")
    if data["identity"] != identity or data["fingerprint"] != digest(data["rows"]):
        raise ValueError(f"Stale or changed {name} artifact; rerun its phase")
    return data["rows"]


def protect_output(root, inputs=()):
    """Keep this experiment separate from established PI/rollout/result caches."""
    protected = [Path(p).resolve() for p in (
        "data/pi/hint", "data/pi/attempted_solution_8k", "data/pi/attempted_solution_16k",
        "data/rollouts", "results/hint_gen_compare", "results/hint_gen_compare_legacy",
        "results/hint_diversity", *inputs,
    ) if p]
    root = root.resolve()
    for path in protected:
        if root.is_relative_to(path) or path.is_relative_to(root):
            raise ValueError(f"Output {root} overlaps protected input/cache {path}")


def run_root(args):
    slug = args.model.rstrip("/").split("/")[-1]
    return Path(args.output_dir or (
        f"data/pi/hint_selection/{args.dataset}/{slug}/"
        f"n{args.num_hints}_k{args.teacher_rollouts}_t{args.temperature:g}_seed{args.seed}"
    )).resolve()


def prepare_config(args):
    from utils import PI_HINT, compose_pi_messages, format_prompt
    from utils.gen_hints import HINT_VALIDATION_VERSION, build_messages

    keys = ("model", "revision", "dataset", "num_questions", "num_hints", "temperature",
            "hint_budget", "teacher_rollouts", "teacher_max_tokens", "teacher_temperature",
            "teacher_top_k", "teacher_seed", "transfer_rollouts", "max_model_len", "seed",
            "dtype", "save_teacher_text")
    return {
        **{key: getattr(args, key) for key in keys},
        "version": VERSION, "method": METHOD,
        "cohort_dir": str(Path(args.cohort_dir).resolve()) if args.cohort_dir else None,
        "rollout_root": str(Path(args.rollout_root).resolve()),
        "model_identity": json.loads(json.dumps(model_identity(args.model))),
        "hint_prompt": build_messages("{question}", "{solution}", args.dataset),
        "teacher_prompt": compose_pi_messages(format_prompt("{question}", args.dataset), PI_HINT.format(hint="{hint}")),
        "hint_validation_version": HINT_VALIDATION_VERSION,
        "generator_top_p": 1.0, "generator_top_k": -1, "teacher_top_p": 1.0,
        "min_p": 0.0, "repetition_penalty": 1.0,
        "transfer_aggregation": "mean_of_within_rollout_token_means",
    }


def render(tokenizer, messages, *, hint=False):
    kwargs = {"enable_thinking": False} if hint else {}
    return list(tokenizer.apply_chat_template(
        [messages], add_generation_prompt=True, tokenize=True, return_dict=True, **kwargs,
    )["input_ids"][0])


def index_student_samples(rollouts):
    """Legacy caches use stored row order; modern caches use explicit sample indices."""
    explicit = "sample_idx" in rollouts.column_names
    sample_ids = rollouts["sample_idx"] if explicit else None
    indices = defaultdict(dict)
    for i, question in enumerate(rollouts["question"]):
        sample = int(sample_ids[i]) if explicit else len(indices[question])
        if sample < 0 or sample in indices[question]:
            raise ValueError(f"Invalid/duplicate student sample index for question: {question[:80]}")
        indices[question][sample] = i
    return indices, "stored_sample_idx" if explicit else "legacy_row_order"


def prepare(args, root):
    from datasets import load_from_disk
    from transformers import AutoTokenizer

    from utils import format_prompt, load_train_dataset, rollout_path
    from utils.gen_hints import build_messages

    config = prepare_config(args)
    if (root / "manifest.json").exists():
        manifest, _ = load_run(root)
        if manifest["config"] != config:
            raise ValueError("Preparation settings changed; use a new --output-dir (even with --force)")
        return
    if root.exists() and any(root.iterdir()):
        raise ValueError("Output is nonempty without a manifest; choose an empty experiment directory")
    if (Path(args.model) / "adapter_config.json").exists():
        raise ValueError("--model must be the frozen base/full model, not a hint-generator adapter")
    source = load_from_disk(args.cohort_dir) if args.cohort_dir else load_train_dataset(args.dataset, require_solution=True)
    if not {"question", "solution", "final_answer"}.issubset(source.column_names):
        raise ValueError("Source needs question, solution, and final_answer columns")
    rollouts = load_from_disk(rollout_path(args.model, args.dataset, args.rollout_root))
    required = {"question", "completion_ids", "gen_model", "dataset"}
    if not required.issubset(rollouts.column_names):
        raise ValueError(f"Student rollout cache needs {sorted(required)}")
    if set(rollouts.unique("gen_model")) != {args.model} or set(rollouts.unique("dataset")) != {args.dataset}:
        raise ValueError("Student rollout cache model/dataset does not match")
    indices, sample_index_source = index_student_samples(rollouts)
    # Do not inspect rewards or choose trajectories based on correctness.
    order = list(range(len(source)))
    random.Random(args.seed).shuffle(order)
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision, trust_remote_code=True)
    cohort, seen, excluded = [], {}, Counter()
    for index in order:
        row = source[index]
        question, answer, solution = row["question"], str(row["final_answer"]), row["solution"]
        if not question or not solution or row["final_answer"] is None:
            excluded["empty_source"] += 1
            continue
        if question in seen:
            if seen[question] != (answer, solution):
                raise ValueError("Conflicting demonstrations/answers for a repeated question")
            continue
        seen[question] = (answer, solution)
        selected = sorted(indices.get(question, {}))[:args.transfer_rollouts]
        if len(selected) != args.transfer_rollouts:
            excluded["missing_student_rollouts"] += 1
            continue
        cached = [rollouts[indices[question][s]] for s in selected]
        if any("final_answer" in r and str(r["final_answer"]) != answer for r in cached):
            raise ValueError("Source and student rollout reference answers differ")
        completions = [list(r["completion_ids"]) for r in cached]
        if any(not ids for ids in completions):
            raise ValueError("Empty cached student completion")
        hint_ids = render(tokenizer, build_messages(question, solution, args.dataset), hint=True)
        student_ids = render(tokenizer, format_prompt(question, args.dataset))
        if (len(hint_ids) + args.hint_budget > args.max_model_len or
                len(student_ids) + max(map(len, completions)) > args.max_model_len):
            excluded["context_limit"] += 1
            continue
        cohort.append({
            "question_id": digest([question, answer])[:24], "source_idx": index,
            "question": question, "final_answer": answer, "solution": solution,
            "hint_prompt_ids": hint_ids, "student_prompt_ids": student_ids,
            "student_sample_indices": selected, "student_completion_ids": completions,
            "student_sample_index_source": sample_index_source,
        })
        if args.num_questions and len(cohort) == args.num_questions:
            break
    if not cohort or (args.num_questions and len(cohort) != args.num_questions):
        raise ValueError(f"Found {len(cohort)} eligible questions; requested {args.num_questions}. Exclusions: {dict(excluded)}")
    manifest = {"config": config, "cohort_fingerprint": digest(cohort),
                "n_questions": len(cohort), "exclusions": dict(excluded),
                "tokenizer_revision": tokenizer.init_kwargs.get("_commit_hash"),
                "software": {p: importlib.metadata.version(p) for p in ("transformers", "datasets", "torch", "vllm")}}
    save_artifact(root, "cohort", digest(config), cohort)
    write_json(root / "manifest.json", manifest)
    print(f"Prepared {len(cohort)} questions -> {root}", flush=True)


def load_run(root):
    manifest = read_json(root / "manifest.json")
    cohort = load_artifact(root, "cohort", digest(manifest["config"]))
    if digest(cohort) != manifest["cohort_fingerprint"]:
        raise ValueError("Cohort fingerprint differs from manifest")
    return manifest, cohort


def cache_for(root, kind, manifest, force):
    return ConditionCache(root, kind, {"manifest": digest(manifest)}, force)


def new_engine(config, args):
    from vllm import LLM

    return LLM(model=config["model"], revision=config["revision"], dtype=config["dtype"],
               generation_config="vllm", max_model_len=config["max_model_len"],
               gpu_memory_utilization=args.gpu_memory_utilization,
               tensor_parallel_size=args.tensor_parallel_size, seed=config["seed"], trust_remote_code=True)


def candidates(root, manifest, cohort):
    rows = load_artifact(root, "candidates", digest(manifest))
    expected = {(q["question_id"], i) for q in cohort for i in range(manifest["config"]["num_hints"])}
    actual = [(r["question_id"], r["sample_idx"]) for r in rows]
    if len(actual) != len(expected) or set(actual) != expected or len({r["hint_id"] for r in rows}) != len(rows):
        raise ValueError("Incomplete or duplicate candidate hints")
    return rows


def generate(args, root):
    from utils.gen_hints import leaks_answer

    manifest, cohort = load_run(root)
    c = manifest["config"]
    cache = cache_for(root, "candidates", manifest, args.force)
    engine = None
    rows = []
    for question in cohort:
        key = cache.key(question["hint_prompt_ids"])
        group = cache.load(key)
        if group is None:
            from vllm import SamplingParams

            if engine is None:
                engine = new_engine(c, args)
            params = SamplingParams(n=c["num_hints"], temperature=c["temperature"], top_p=1.0,
                                    top_k=-1, min_p=0.0, repetition_penalty=1.0,
                                    max_tokens=c["hint_budget"], seed=c["seed"])
            output = engine.generate([{"prompt_token_ids": question["hint_prompt_ids"]}], params)[0]
            if len(output.outputs) != c["num_hints"]:
                raise RuntimeError("Wrong number of generated hints")
            group = []
            for i, sample in enumerate(output.outputs):
                hint = sample.text.strip()
                reason = ("empty" if not hint else "thinking" if "<think>" in hint or "</think>" in hint
                          else "answer_leak" if leaks_answer(hint, question["final_answer"], c["dataset"]) else "")
                truncated = sample.finish_reason == "length"
                group.append({"question_id": question["question_id"], "sample_idx": i,
                              "hint_id": digest([question["question_id"], i, hint]), "hint": hint,
                              "n_tokens": len(sample.token_ids), "token_ids": list(sample.token_ids),
                              "invalid_reason": reason, "truncated": truncated,
                              "valid": not reason and not truncated, "finish_reason": sample.finish_reason})
            cache.save(key, group)
        rows.extend(group)
    save_artifact(root, "candidates", digest(manifest), rows)
    candidates(root, manifest, cohort)
    print(f"Candidates: {len(rows)} total, {sum(r['valid'] for r in rows)} valid; {cache.stats()}", flush=True)


def teacher_messages(question, hint, dataset):
    from utils import PI_HINT, compose_pi_messages, format_prompt

    return compose_pi_messages(format_prompt(question, dataset), PI_HINT.format(hint=hint))


def score_identity(manifest, hints):
    return digest([digest(manifest), digest(hints)])


def sufficiency(args, root):
    from utils import grade

    manifest, cohort = load_run(root)
    c = manifest["config"]
    hints = candidates(root, manifest, cohort)
    questions = {q["question_id"]: q for q in cohort}
    cache = cache_for(root, "sufficiency", manifest, args.force)
    rows, pending = [], []
    for hint in hints:
        if not hint["valid"]:
            continue
        q = questions[hint["question_id"]]
        messages = teacher_messages(q["question"], hint["hint"], c["dataset"])
        key = cache.key([hint["hint_id"], messages, q["final_answer"]])
        result = cache.load(key)
        if result is None:
            pending.append((key, hint, q, messages))
        else:
            rows.append(result)
    if pending:
        from vllm import SamplingParams

        engine = new_engine(c, args)
        tokenizer = engine.get_tokenizer()
        params = SamplingParams(n=c["teacher_rollouts"], temperature=c["teacher_temperature"],
                                top_p=1.0, top_k=c["teacher_top_k"] or -1, min_p=0.0,
                                repetition_penalty=1.0, max_tokens=c["teacher_max_tokens"], seed=c["teacher_seed"])
        for start in range(0, len(pending), args.teacher_batch_size):
            batch = pending[start:start + args.teacher_batch_size]
            prompts = []
            for _, hint, _, messages in batch:
                ids = render(tokenizer, messages)
                if len(ids) + c["teacher_max_tokens"] > c["max_model_len"]:
                    raise ValueError(f"Teacher prompt does not fit for {hint['hint_id']}; no truncation allowed")
                prompts.append({"prompt_token_ids": ids})
            outputs = engine.generate(prompts, params)
            for (key, hint, q, _), output in zip(batch, outputs, strict=True):
                if len(output.outputs) != c["teacher_rollouts"]:
                    raise RuntimeError("Wrong number of teacher rollouts")
                correct = [bool(grade(s.text, q["final_answer"], c["dataset"])[1]) for s in output.outputs]
                result = {"hint_id": hint["hint_id"], "correct": correct, "n_samples": len(correct),
                          "n_correct": sum(correct), "sufficiency": sum(correct) / len(correct),
                          "teacher_token_counts": [len(s.token_ids) for s in output.outputs],
                          "finish_reasons": [s.finish_reason for s in output.outputs]}
                if c["save_teacher_text"]:
                    result["texts"] = [s.text for s in output.outputs]
                cache.save(key, result)
                rows.append(result)
            print(f"Teacher conditions {min(start + len(batch), len(pending))}/{len(pending)}", flush=True)
    save_artifact(root, "sufficiency", score_identity(manifest, hints), sorted(rows, key=lambda r: r["hint_id"]))
    print(f"Sufficiency: {cache.stats()}", flush=True)


def score_completion(model, prompt_ids, completion_ids):
    import torch

    from utils.model_scoring import per_token_logps

    ids = torch.tensor([prompt_ids + completion_ids], dtype=torch.long, device=model.device)
    completion = torch.tensor([completion_ids], dtype=torch.long, device=model.device)
    with torch.inference_mode():
        return per_token_logps(model, ids, completion).squeeze(0).float().cpu().tolist()


def transfer(args, root):
    manifest, cohort = load_run(root)
    c = manifest["config"]
    hints = candidates(root, manifest, cohort)
    cache = cache_for(root, "transfer", manifest, args.force)
    student_cache = cache_for(root, "student_logps", manifest, args.force)
    questions = {q["question_id"]: q for q in cohort}
    model = tokenizer = None
    rows, in_memory = [], {}
    for hint in hints:
        if not hint["valid"]:
            continue
        q = questions[hint["question_id"]]
        messages = teacher_messages(q["question"], hint["hint"], c["dataset"])
        means = []
        for position, completion_ids in enumerate(q["student_completion_ids"]):
            key = cache.key([hint["hint_id"], messages, completion_ids])
            result = cache.load(key)
            if result is None:
                if model is None:
                    import torch
                    from transformers import AutoModelForCausalLM, AutoTokenizer

                    tokenizer = AutoTokenizer.from_pretrained(c["model"], revision=c["revision"], trust_remote_code=True)
                    model = AutoModelForCausalLM.from_pretrained(
                        c["model"], revision=c["revision"], dtype=getattr(torch, c["dtype"]), trust_remote_code=True,
                    ).eval().to("cuda")
                teacher_ids = render(tokenizer, messages)
                if max(len(teacher_ids), len(q["student_prompt_ids"])) + len(completion_ids) > c["max_model_len"]:
                    raise ValueError(f"Transfer sequence does not fit for {hint['hint_id']}; no truncation allowed")
                student_key = student_cache.key([q["student_prompt_ids"], completion_ids])
                if student_key not in in_memory:
                    values = student_cache.load(student_key)
                    if values is None:
                        values = score_completion(model, q["student_prompt_ids"], completion_ids)
                        student_cache.save(student_key, values)
                    in_memory[student_key] = values
                teacher_values = score_completion(model, teacher_ids, completion_ids)
                student_values = in_memory[student_key]
                if len(student_values) != len(completion_ids) or len(teacher_values) != len(completion_ids):
                    raise ValueError("Token log-probability alignment mismatch")
                value = statistics.mean(s - t for s, t in zip(student_values, teacher_values, strict=True))
                if not math.isfinite(value):
                    raise ValueError("Non-finite transfer score")
                result = {"raw_transfer": value}
                cache.save(key, result)
            means.append(result["raw_transfer"])
        rows.append({"hint_id": hint["hint_id"], "raw_transfer": statistics.mean(means),
                     "rollout_means": means, "n_rollouts": len(means)})
    save_artifact(root, "transfer", score_identity(manifest, hints), rows)
    print(f"Transfer: {cache.stats()}", flush=True)


def choose_hint(hints, scores, epsilon, gamma, hint_budget, clamp_transfer=True):
    """Literal per-question empirical best-minus-epsilon rule, with stable ties."""
    if not 0 <= epsilon <= 1 or not math.isfinite(gamma) or gamma < 0 or hint_budget <= 0:
        raise ValueError("Invalid selection settings")
    valid = [h for h in hints if h["valid"]]
    if not valid:
        return {"status": "no_valid_hints", "selected": None, "n_valid": 0}
    for h in valid:
        score = scores[h["hint_id"]]
        if not 0 <= score["sufficiency"] <= 1 or not math.isfinite(score["raw_transfer"]):
            raise ValueError("Invalid sufficiency/transfer score")
    best = max(scores[h["hint_id"]]["sufficiency"] for h in valid)
    threshold = max(0.0, best - epsilon)
    eligible = []
    for hint in valid:
        s = scores[hint["hint_id"]]
        if s["sufficiency"] + 1e-12 < threshold:
            continue
        transfer_cost = max(0.0, s["raw_transfer"]) if clamp_transfer else s["raw_transfer"]
        cost = min(hint["n_tokens"] / hint_budget, 1.0)
        eligible.append({**hint, "sufficiency": s["sufficiency"], "raw_transfer": s["raw_transfer"],
                         "length_cost": cost, "transfer_cost": transfer_cost,
                         "objective": cost + gamma * transfer_cost})
    selected = min(eligible, key=lambda h: (h["objective"], -h["sufficiency"], h["n_tokens"], h["sample_idx"]))
    return {"status": "selected" if best > 0 else "selected_zero_success", "selected": selected,
            "best_sufficiency": best, "threshold": threshold, "n_valid": len(valid),
            "n_eligible": len(eligible), "eligible_hint_ids": [h["hint_id"] for h in eligible]}


def select(args, root):
    from datasets import Dataset, load_from_disk

    manifest, cohort = load_run(root)
    hints = candidates(root, manifest, cohort)
    identity = score_identity(manifest, hints)
    srows = load_artifact(root, "sufficiency", identity)
    trows = load_artifact(root, "transfer", identity)
    valid_ids = {h["hint_id"] for h in hints if h["valid"]}
    for name, rows in (("sufficiency", srows), ("transfer", trows)):
        if len(rows) != len(valid_ids) or {r["hint_id"] for r in rows} != valid_ids:
            raise ValueError(f"Missing or duplicate {name} scores")
    c = manifest["config"]
    if any(r["n_samples"] != c["teacher_rollouts"] or len(r["correct"]) != r["n_samples"] or
           sum(r["correct"]) != r["n_correct"] or r["sufficiency"] != r["n_correct"] / r["n_samples"] for r in srows):
        raise ValueError("Teacher score sample counts or success fractions differ")
    if any(r["n_rollouts"] != c["transfer_rollouts"] or len(r["rollout_means"]) != r["n_rollouts"] or
           not all(math.isfinite(v) for v in r["rollout_means"]) or
           not math.isclose(statistics.mean(r["rollout_means"]), r["raw_transfer"], abs_tol=1e-12) for r in trows):
        raise ValueError("Transfer score rollout counts differ")
    scores = {r["hint_id"]: r for r in srows}
    for r in trows:
        scores[r["hint_id"]] = {**scores[r["hint_id"]], **r}
    selection_config = {"epsilon": args.epsilon, "gamma": args.gamma, "clamp_transfer": args.clamp_transfer,
                        "scores_fingerprint": digest([srows, trows]), "input_identity": identity,
                        "rule_version": VERSION}
    out = root / "selections" / f"epsilon_{args.epsilon:g}_gamma_{args.gamma:g}_{digest(selection_config)[:10]}"
    records, exported = [], []
    for q in cohort:
        group = [h for h in hints if h["question_id"] == q["question_id"]]
        result = choose_hint(group, scores, args.epsilon, args.gamma, c["hint_budget"], args.clamp_transfer)
        records.append({"question_id": q["question_id"], **result})
        if result["selected"] is None:
            continue
        h = result["selected"]
        exported.append({"question": q["question"], "final_answer": q["final_answer"], "hint": h["hint"],
                         "gen_model": c["model"], "dataset": c["dataset"], "question_id": q["question_id"],
                         "selection_method": METHOD, "selection_id": digest(selection_config),
                         "hint_id": h["hint_id"], "hint_sample_idx": h["sample_idx"],
                         "sufficiency": h["sufficiency"], "best_sufficiency": result["best_sufficiency"],
                         "threshold": result["threshold"], "raw_transfer": h["raw_transfer"],
                         "n_tokens": h["n_tokens"], "length_cost": h["length_cost"],
                         "transfer_cost": h["transfer_cost"],
                         "objective": h["objective"], "selection_status": result["status"]})
    summary = {"config": selection_config, "n_questions": len(cohort), "n_selected": len(exported),
               "status_counts": dict(Counter(r["status"] for r in records)),
               "candidate_invalid_reasons": dict(Counter(h["invalid_reason"] for h in hints if h["invalid_reason"])),
               "candidate_truncated": sum(h["truncated"] for h in hints),
               "selected_fingerprint": digest(exported),
               "mean_selected_tokens": statistics.mean(h["n_tokens"] for h in exported) if exported else None,
               "mean_selected_raw_transfer": statistics.mean(h["raw_transfer"] for h in exported) if exported else None,
               "mean_selected_objective": statistics.mean(h["objective"] for h in exported) if exported else None,
               "mean_selected_sufficiency": statistics.mean(h["sufficiency"] for h in exported) if exported else None,
               "note": "Scores were used for selection; they are not independent evaluation of selected hints."}
    # The content-derived directory preserves previous epsilon/gamma/score experiments.
    if (out / "summary.json").exists():
        if read_json(out / "summary.json") != summary:
            raise ValueError("Existing selection summary differs")
        if read_json(out / "selection.json") != records or (exported and
                digest(list(load_from_disk(str(out / "hints")))) != digest(exported)):
            raise ValueError("Existing selection export is incomplete or changed")
        print(f"Reusing selection -> {out}", flush=True)
        return
    out.mkdir(parents=True, exist_ok=True)
    if exported:
        temporary = out / f".hints-{os.getpid()}"
        Dataset.from_list(exported).save_to_disk(str(temporary))
        # A directory without summary.json is an interrupted export; preserve it.
        if (out / "hints").exists():
            (out / "hints").rename(out / f"interrupted-hints-{os.getpid()}")
        temporary.rename(out / "hints")
    write_json(out / "selection.json", records)
    write_json(out / "summary.json", summary)
    print(f"Selected {len(exported)}/{len(cohort)} questions: {summary['status_counts']}", flush=True)
    print(f"SDFT --hint-cache {out / 'hints'}" if exported else "No training cache exported: all hints invalid", flush=True)


def worker(phase, args_dict, root):
    globals()[phase](argparse.Namespace(**args_dict), Path(root))


def run_phase(phase, args, root):
    process = multiprocessing.get_context("spawn").Process(target=worker, args=(phase, vars(args), str(root)))
    process.start()
    try:
        process.join()
    except BaseException:
        process.terminate()
        process.join()
        raise
    if process.exitcode != 0:
        raise RuntimeError(f"{phase} failed with exit code {process.exitcode}; completed condition caches are reusable")


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase", choices=["sweep", "prepare", "generate", "sufficiency", "transfer", "select"], default="sweep")
    p.add_argument("--output-dir", help="Dedicated experiment directory. Later phases use its saved configuration.")
    p.add_argument("--model", default="Qwen/Qwen3-1.7B")
    p.add_argument("--revision", default=None)
    p.add_argument("--dataset", default="deepmath")
    p.add_argument("--cohort-dir", help="Optional saved HF dataset containing question/solution/final_answer")
    p.add_argument("--rollout-root", default="data/rollouts")
    p.add_argument("--num-questions", type=int, default=128, help="0 means all eligible questions")
    p.add_argument("--num-hints", type=int, default=8)
    p.add_argument("--temperature", type=float, default=1.4)
    p.add_argument("--hint-budget", type=int, default=128)
    p.add_argument("--teacher-rollouts", type=int, default=8)
    p.add_argument("--teacher-max-tokens", type=int, default=8192)
    p.add_argument("--teacher-temperature", type=float, default=1.0)
    p.add_argument("--teacher-top-k", type=int, default=20)
    p.add_argument("--teacher-seed", type=int, default=43)
    p.add_argument("--save-teacher-text", action="store_true")
    p.add_argument("--transfer-rollouts", type=int, default=4)
    p.add_argument("--epsilon", type=float, default=1 / 8)
    p.add_argument("--gamma", type=float, default=6.0)
    p.add_argument("--clamp-transfer", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    p.add_argument("--max-model-len", type=int, default=32768)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--teacher-batch-size", type=int, default=8)
    p.add_argument("--force", action="store_true", help="Recompute requested inference stages; preparation remains immutable")
    return p


def main():
    p = build_parser()
    args = p.parse_args()
    if args.num_questions < 0 or min(args.num_hints, args.hint_budget, args.teacher_rollouts,
                                    args.teacher_max_tokens, args.transfer_rollouts,
                                    args.tensor_parallel_size, args.teacher_batch_size) < 1:
        p.error("Counts must be positive (num-questions may be zero)")
    if any(not math.isfinite(t) or t <= 0 for t in (args.temperature, args.teacher_temperature)):
        p.error("Temperatures must be finite and positive")
    if not 0 <= args.epsilon <= 1 or not math.isfinite(args.gamma) or args.gamma < 0:
        p.error("Require 0 <= epsilon <= 1 and finite gamma >= 0")
    if not 0 < args.gpu_memory_utilization <= 1 or args.teacher_top_k < 0:
        p.error("Invalid GPU utilization or teacher top-k")
    if args.max_model_len <= max(args.hint_budget, args.teacher_max_tokens):
        p.error("Context length must exceed generation budgets")
    if args.force and args.phase in ("prepare", "select"):
        p.error("--force only applies to inference stages")
    root = run_root(args)
    protect_output(root, (args.cohort_dir, args.rollout_root))
    if args.phase in ("prepare", "sweep"):
        prepare(args, root)
    if args.phase == "sweep":
        for phase in ("generate", "sufficiency", "transfer"):
            run_phase(phase, args, root)
        select(args, root)
    elif args.phase != "prepare":
        globals()[args.phase](args, root)


if __name__ == "__main__":
    main()
