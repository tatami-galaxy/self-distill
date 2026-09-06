"""Compare OPD, OPSD and outcome MC advantages on one frozen student's rollouts.

CUDA_VISIBLE_DEVICES=0,1 uv run python -m eval.advantage_comparison \
  --student /path/to/student/checkpoint-100 \
  --opd-teacher Qwen/Qwen3-30B-A3B-Thinking-2507 \
  --opsd-teacher Qwen/Qwen3-1.7B \
  --dataset deepmath \
  --pi-modes answer full \
  --num-problems 8 --n 2 \
  --selection-modes uniform steps \
  --num-token-samples 16 --mc-samples 32 \
  --min-segment-tokens 32 --max-segment-tokens 256 \
  --max-completion-length 8192 --max-model-len 16384 \
  --student-device cuda:0 --teacher-device cuda:1 \
  --output-dir results/advantage_comparison/student100

"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import random
import re
import subprocess


SCHEMA_VERSION = 1
PI_MODES = ("none", "answer", "full", "hint", "rollout")


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=True).encode()).hexdigest()


def keyed_seed(seed, *parts):
    return int(fingerprint([seed, *parts])[:8], 16) % (2**31)


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def load_rows(path):
    path = Path(path)
    if path.is_dir():
        from datasets import load_from_disk
        return list(load_from_disk(str(path)))
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    rows = read_json(path)
    if not isinstance(rows, list):
        raise ValueError(f"{path} must contain a JSON list of rows")
    return rows


def tokenizer_identity(tokenizer):
    return {"vocab": fingerprint(tokenizer.get_vocab()),
            "special_ids": sorted(tokenizer.all_special_ids),
            "chat_template": fingerprint(tokenizer.chat_template)}


def check_tokenizers(student, teacher):
    a, b = tokenizer_identity(student), tokenizer_identity(teacher)
    if a["vocab"] != b["vocab"] or a["special_ids"] != b["special_ids"]:
        raise ValueError("Student and teacher must have identical token-to-ID mappings and special IDs")
    return b


def decode(tokenizer, ids, *, special=False):
    return tokenizer.decode(ids, skip_special_tokens=not special,
                            clean_up_tokenization_spaces=False)


def delimiter_boundaries(tokenizer, ids):
    """Map non-overlapping blank-line delimiters to original token boundaries.

    Locate each character endpoint with prefix decodes, never retokenizing a prefix.
    A delimiter inside a multi-character token snaps right to the end of that token.
    Counting complete delimiters also avoids partial UTF-8 replacement-character offsets.
    """
    text = decode(tokenizer, ids, special=True)
    count = len(list(re.finditer("\n\n", text)))
    boundaries, lower = [], 0
    for ordinal in range(1, count + 1):
        lo, hi = lower, len(ids)
        while lo < hi:
            mid = (lo + hi) // 2
            seen = decode(tokenizer, ids[:mid], special=True).count("\n\n")
            if seen >= ordinal:
                hi = mid
            else:
                lo = mid + 1
        boundaries.append(lo)
        lower = lo
    return sorted(set(boundaries))


def make_segments(length, boundaries, minimum, maximum):
    """Prefer delimiter endpoints; merge short spans and hard-split long spans.

    Only the final span may be shorter than minimum. All tokens occur exactly once.
    """
    if not 1 <= minimum <= maximum:
        raise ValueError("Require 1 <= min segment tokens <= max segment tokens")
    if length < 1:
        raise ValueError("Cannot segment an empty completion")
    ends = sorted(set([b for b in boundaries if 0 < b < length] + [length]))
    segments, start = [], 0
    for end in ends:
        while end - start > maximum:
            segments.append((start, start + maximum))
            start += maximum
        if end - start >= minimum or end == length:
            if end > start:
                segments.append((start, end))
                start = end
    return segments


def uniform_positions(length, samples, seed, rollout_id):
    if samples < 1 or length < 1:
        raise ValueError("Token sample count and completion length must be positive")
    return sorted(random.Random(keyed_seed(seed, "positions", rollout_id)).sample(
        range(length), min(samples, length)))


def prefix_key(row, end):
    return fingerprint([row["question_id"], row["student_prompt_ids"],
                        row["completion_ids"][:end]])


def make_plan(rows, tokenizer, modes, samples, minimum, maximum, seed):
    plans, prefixes = [], {}
    for row in rows:
        length = len(row["completion_ids"])
        tokens = uniform_positions(length, samples, seed, row["rollout_id"]) if "uniform" in modes else []
        segments = make_segments(length, delimiter_boundaries(tokenizer, row["completion_ids"]),
                                 minimum, maximum) if "steps" in modes else []
        spans = [(t, t + 1) for t in tokens] + segments
        endpoints = sorted({p for span in spans for p in span})
        keys = {}
        for end in endpoints:
            if end == length:
                # End of the original trajectory is terminal, with zero future value.
                keys[str(end)] = None
                continue
            key = prefix_key(row, end)
            keys[str(end)] = key
            spec = {"key": key, "question_id": row["question_id"],
                    "student_prompt_ids": row["student_prompt_ids"],
                    "prefix_ids": row["completion_ids"][:end],
                    "final_answer": row["final_answer"]}
            if key in prefixes and prefixes[key] != spec:
                raise ValueError("Conflicting prefix provenance")
            prefixes[key] = spec
        plans.append({"rollout_id": row["rollout_id"], "uniform_positions": tokens,
                      "segments": [list(span) for span in segments], "prefix_keys": keys})
    return {"rollouts": plans, "prefixes": list(prefixes.values())}


def wilson_interval(successes, samples, z=1.959963984540054):
    if not 0 <= successes <= samples or samples < 1:
        raise ValueError("Invalid MC success/sample counts")
    p = successes / samples
    denominator = 1 + z * z / samples
    center = (p + z * z / (2 * samples)) / denominator
    radius = z * math.sqrt(p * (1 - p) / samples + z * z / (4 * samples**2)) / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


def mc_advantage(before, after, terminal_reward=None):
    """gamma=1, zero intermediate reward; terminal transition carries observed reward.

    CI is a conservative difference of Bonferroni-adjusted Wilson intervals. Unlike a
    plug-in standard error it remains nonzero when all sampled rewards are identical.
    """
    b = before["successes"] / before["samples"]
    bi = wilson_interval(before["successes"], before["samples"], z=2.241402727604947)
    if terminal_reward is None:
        a = after["successes"] / after["samples"]
        ai = wilson_interval(after["successes"], after["samples"], z=2.241402727604947)
    else:
        a = float(terminal_reward)
        ai = [a, a]
    return {"advantage": a - b, "ci95": [ai[0] - bi[1], ai[1] - bi[0]],
            "value_before": b, "value_after": 0.0 if terminal_reward is not None else a,
            "transition_reward": 0.0 if terminal_reward is None else a}


def experiment_config(args):
    # K can increase on resume without changing rollouts, positions or earlier MC draws.
    excluded = {"phase", "output_dir", "mc_samples", "mc_batch_size", "bootstrap_samples"}
    models = {}
    for name in ("student", "opd_teacher", "opsd_teacher"):
        path = Path(getattr(args, name))
        if path.is_dir():
            models[name] = [[str(p.relative_to(path)), p.stat().st_size, p.stat().st_mtime_ns]
                            for p in sorted(path.rglob("*")) if p.is_file()
                            and p.suffix in (".json", ".safetensors", ".bin", ".model", ".jinja")]
    return {"schema_version": SCHEMA_VERSION, "local_model_files": models,
            **{k: v for k, v in vars(args).items() if k not in excluded}}


def ensure_manifest(args):
    path = Path(args.output_dir) / "manifest.json"
    config = experiment_config(args)
    if path.exists():
        if read_json(path)["config"] != config:
            raise ValueError("Output directory has different experiment settings; use a new --output-dir")
    else:
        try:
            commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            commit = None
        write_json(path, {"config": config, "git_commit": commit})


def build_messages(problem, mode):
    from utils import (compose_pi_messages, format_prompt_math, PI_ANSWER, PI_FULL,
                       PI_HINT, PI_ROLLOUT)
    messages = format_prompt_math(problem["question"])
    templates = {"answer": (PI_ANSWER, "answer", "final_answer"),
                 "full": (PI_FULL, "demo", "solution"),
                 "hint": (PI_HINT, "hint", "hint"),
                 "rollout": (PI_ROLLOUT, "attempt", "rollout")}
    if mode == "none":
        return messages
    template, field, column = templates[mode]
    return compose_pi_messages(messages, template.format(**{field: problem[column]}))


def chat_ids(tokenizer, messages):
    return list(tokenizer.apply_chat_template([messages], tokenize=True,
                add_generation_prompt=True, return_dict=True)["input_ids"][0])


def load_source(args):
    """Attach cached self-teacher hints to registered datasets by exact problem identity."""
    from utils import DATASET_REGISTRY_EVAL, load_hint_cache, load_train_dataset

    if args.cohort:
        return load_rows(args.cohort)
    if args.dataset in DATASET_REGISTRY_EVAL:
        source = list(DATASET_REGISTRY_EVAL[args.dataset]())
    else:
        source = list(load_train_dataset(args.dataset, require_solution="full" in args.pi_modes))
    if "hint" in args.pi_modes:
        hints = load_hint_cache(args.opsd_teacher, args.dataset)
        by_problem = {}
        for row in hints:
            key = (str(row["question"]), str(row["final_answer"]))
            by_problem.setdefault(key, row["hint"])
        source = [{**row, "hint": by_problem.get((str(row["question"]), str(row["final_answer"])))}
                  for row in source]
    return source


def prepare(args):
    from transformers import AutoConfig, AutoTokenizer

    out = Path(args.output_dir)
    if (out / "cohort.json").exists():
        return
    student = AutoTokenizer.from_pretrained(args.student, trust_remote_code=True)
    opd = AutoTokenizer.from_pretrained(args.opd_teacher, trust_remote_code=True)
    opsd = AutoTokenizer.from_pretrained(args.opsd_teacher, trust_remote_code=True)

    tokenizers = {"student": tokenizer_identity(student), "opd": check_tokenizers(student, opd),
                  "opsd": check_tokenizers(student, opsd)}
    configs = {name: AutoConfig.from_pretrained(path, trust_remote_code=True)
               for name, path in (("student", args.student), ("opd", args.opd_teacher),
                                  ("opsd", args.opsd_teacher))}
    revisions = {name: getattr(config, "_commit_hash", None) for name, config in configs.items()}
    eos = configs["student"].eos_token_id
    eos_ids = set(eos if isinstance(eos, list) else ([] if eos is None else [eos]))
    if student.eos_token_id is not None:
        eos_ids.add(student.eos_token_id)
    if not eos_ids:
        raise ValueError("Student must define an EOS token")
    
    source = load_source(args)
    source_hash = fingerprint(source)
    order = list(range(len(source)))
    random.Random(args.seed).shuffle(order)
    needed = {"answer": "final_answer", "full": "solution", "hint": "hint", "rollout": "rollout"}
    rows, seen = [], set()
    missing = too_long = 0

    for index in order:
        problem = source[index]
        columns = {"question", "final_answer"} | {needed[m] for m in args.pi_modes if m in needed}
        if any(problem.get(c) is None or not str(problem[c]).strip() for c in columns):
            missing += 1
            continue
        problem = {c: str(problem[c]) for c in columns}
        qid = fingerprint([problem["question"], problem["final_answer"]])[:24]
        if qid in seen:
            continue
        student_ids = chat_ids(student, build_messages(problem, "none"))
        teacher_ids = {"opd": chat_ids(opd, build_messages(problem, "none"))}
        teacher_ids.update({f"opsd_{mode}": chat_ids(opsd, build_messages(problem, mode))
                            for mode in args.pi_modes})
        if any(len(ids) + args.max_completion_length > args.max_model_len
               for ids in [student_ids, *teacher_ids.values()]):
            too_long += 1
            continue
        rows.append({**problem, "question_id": qid, "source_index": index,
                     "student_prompt_ids": student_ids, "teacher_prompt_ids": teacher_ids})
        seen.add(qid)
        if len(rows) == args.num_problems:
            break

    if len(rows) != args.num_problems:
        raise ValueError(f"Only {len(rows)}/{args.num_problems} eligible problems; {missing} missing PI, "
                         f"{too_long} too long. Check hint-cache coverage or supply --cohort with the required PI columns.")
    
    write_json(out / "cohort.json", {"rows": rows, "source_fingerprint": source_hash,
               "tokenizers": tokenizers, "eos_ids": sorted(eos_ids), "model_revisions": revisions,
               "missing_context": missing, "prompt_too_long": too_long})


def load_llm(args):
    from vllm import LLM
    revision = read_json(Path(args.output_dir) / "cohort.json")["model_revisions"]["student"]
    return LLM(model=args.student, dtype=args.dtype, max_model_len=args.max_model_len,
               tensor_parallel_size=args.tensor_parallel_size, seed=args.seed,
               gpu_memory_utilization=args.gpu_memory_utilization, trust_remote_code=True,
               generation_config="vllm", revision=revision, tokenizer_revision=revision)


def sampling_params(args, remaining, seed, eos_ids):
    from vllm import SamplingParams
    return SamplingParams(n=1, max_tokens=remaining, temperature=1.0, top_p=1.0,
                          top_k=-1, min_p=0.0, repetition_penalty=1.0, seed=seed,
                          stop_token_ids=eos_ids)


def validate_completion(ids, budget, eos_ids, finish_reason):
    if not ids or len(ids) > budget:
        raise ValueError("Empty completion or completion exceeds remaining budget")
    if any(t in eos_ids for t in ids[:-1]):
        raise ValueError("Completion contains tokens after EOS")
    if ids[-1] not in eos_ids and len(ids) != budget:
        raise ValueError(f"Unsupported nonterminal completion (finish_reason={finish_reason})")


def generate(args):
    from utils import grade
    out = Path(args.output_dir)
    cohort = read_json(out / "cohort.json")
    path = out / "rollouts.json"
    if path.exists():
        return
    llm = load_llm(args)
    tokenizer = llm.get_tokenizer()
    if tokenizer_identity(tokenizer) != cohort["tokenizers"]["student"]:
        raise ValueError("Generation tokenizer changed since cohort preparation")
    jobs = [(p, i) for p in cohort["rows"] for i in range(args.n)]
    params = [sampling_params(args, args.max_completion_length,
              keyed_seed(args.seed, "rollout", p["question_id"], i), cohort["eos_ids"])
              for p, i in jobs]
    outputs = llm.generate([{"prompt_token_ids": p["student_prompt_ids"]} for p, _ in jobs], params)
    rows = []
    for (problem, sample), output in zip(jobs, outputs, strict=True):
        completion = output.outputs[0]
        ids = list(completion.token_ids)
        validate_completion(ids, args.max_completion_length, cohort["eos_ids"], completion.finish_reason)
        text = decode(tokenizer, ids)
        rows.append({**problem, "sample_idx": sample, "completion_ids": ids,
                     "completion_text": text, "reward": float(grade(text, problem["final_answer"])[1]),
                     "truncated": ids[-1] not in cohort["eos_ids"],
                     "finish_reason": completion.finish_reason,
                     "rollout_id": fingerprint([args.student, problem["question_id"], sample, ids, args.seed])})
    write_json(path, {"cohort_fingerprint": fingerprint(cohort), "rows": rows})


def plan(args):
    from transformers import AutoTokenizer
    out = Path(args.output_dir)
    rollouts = read_json(out / "rollouts.json")
    cohort = read_json(out / "cohort.json")
    if rollouts["cohort_fingerprint"] != fingerprint(cohort):
        raise ValueError("Rollouts do not match the cached cohort")
    tokenizer = AutoTokenizer.from_pretrained(args.student, trust_remote_code=True)
    if tokenizer_identity(tokenizer) != cohort["tokenizers"]["student"]:
        raise ValueError("Student tokenizer changed since generation")
    spec = make_plan(rollouts["rows"], tokenizer, args.selection_modes, args.num_token_samples,
                     args.min_segment_tokens, args.max_segment_tokens, args.seed)
    spec["rollout_fingerprint"] = fingerprint(rollouts)
    path = out / "plan.json"
    if path.exists() and read_json(path) != spec:
        raise ValueError("Existing token selection plan does not match rollouts/settings")
    write_json(path, spec)
    print(f"{len(spec['rollouts'])} rollouts; {len(spec['prefixes'])} distinct nonterminal prefixes; "
          f"{len(spec['prefixes']) * args.mc_samples} MC continuations at K={args.mc_samples}", flush=True)


def mc(args):
    from utils import grade
    out = Path(args.output_dir)
    spec, cohort = read_json(out / "plan.json"), read_json(out / "cohort.json")
    caches, jobs = {}, []
    for prefix in spec["prefixes"]:
        path = out / "mc" / f"{prefix['key']}.json"
        cached = read_json(path) if path.exists() else {"prefix": prefix, "draws": []}
        if cached["prefix"] != prefix:
            raise ValueError(f"MC prefix cache mismatch at {path}")
        caches[prefix["key"]] = cached
        jobs.extend((prefix, i) for i in range(len(cached["draws"]), args.mc_samples))
    if not jobs:
        return
    llm = load_llm(args)
    tokenizer = llm.get_tokenizer()
    if tokenizer_identity(tokenizer) != cohort["tokenizers"]["student"]:
        raise ValueError("MC tokenizer changed since generation")
    for start in range(0, len(jobs), args.mc_batch_size):
        batch = jobs[start:start + args.mc_batch_size]
        prompts = [{"prompt_token_ids": p["student_prompt_ids"] + p["prefix_ids"]} for p, _ in batch]
        params = [sampling_params(args, args.max_completion_length - len(p["prefix_ids"]),
                  keyed_seed(args.seed, "mc", p["key"], i), cohort["eos_ids"]) for p, i in batch]
        outputs = llm.generate(prompts, params, use_tqdm=False)
        touched = set()
        for (prefix, i), output in zip(batch, outputs, strict=True):
            completion = output.outputs[0]
            ids = list(completion.token_ids)
            validate_completion(ids, args.max_completion_length - len(prefix["prefix_ids"]),
                                cohort["eos_ids"], completion.finish_reason)
            text = decode(tokenizer, prefix["prefix_ids"] + ids)
            draw = {"sample_idx": i, "seed": keyed_seed(args.seed, "mc", prefix["key"], i),
                    "reward": int(grade(text, prefix["final_answer"])[1]),
                    "num_tokens": len(ids), "finish_reason": completion.finish_reason}
            if args.save_mc_completions:
                draw["continuation_ids"] = ids
            cache = caches[prefix["key"]]
            if len(cache["draws"]) != i:
                raise ValueError("MC samples are not in canonical order")
            cache["draws"].append(draw)
            touched.add(prefix["key"])
        for key in touched:
            cache = caches[key]
            cache.update(samples=len(cache["draws"]), successes=sum(d["reward"] for d in cache["draws"]))
            write_json(out / "mc" / f"{key}.json", cache)
        print(f"MC continuations {min(start + len(batch), len(jobs))}/{len(jobs)}", flush=True)


def causal_logit_chunks(model, prompt_ids, completion_ids, chunk_size):
    """Stream batch-one causal logits using KV cache, without a full T x V allocation.

    Prefill prompt[:-1], then [prompt[-1]] + completion[:-1]; emitted row t predicts
    completion[t]. Both models follow the same completion chunk schedule.
    """
    import torch
    device = model.get_input_embeddings().weight.device
    cache = None
    with torch.inference_mode():
        for start in range(0, len(prompt_ids) - 1, chunk_size):
            ids = prompt_ids[start:min(start + chunk_size, len(prompt_ids) - 1)]
            result = model(input_ids=torch.tensor([ids], device=device), past_key_values=cache,
                           use_cache=True, logits_to_keep=1)
            cache = result.past_key_values
            if cache is None:
                raise ValueError("Scoring requires a causal LM with a working KV cache")
            del result
        inputs = [prompt_ids[-1]] + completion_ids[:-1]
        for start in range(0, len(inputs), chunk_size):
            ids = inputs[start:start + chunk_size]
            result = model(input_ids=torch.tensor([ids], device=device), past_key_values=cache,
                           use_cache=True, logits_to_keep=len(ids))
            cache = result.past_key_values
            if cache is None or result.logits.shape[1] != len(ids):
                raise ValueError("Model did not return the expected cached causal logits")
            yield start, result.logits
            del result


def distillation_scores(student_logits, teacher_logits, token_ids):
    """FP32 full-vocabulary normalization; exact per-prefix reverse KL baseline."""
    import torch
    from utils.model_scoring import _selective_logps_fp32
    if student_logits.shape != teacher_logits.shape:
        raise ValueError("Student and teacher output vocabulary sizes must match")
    # Move only this bounded chunk; retain the model's lower-precision forward logits.
    teacher_logits = teacher_logits.to(student_logits.device)
    ids = torch.tensor([token_ids], device=student_logits.device)
    student_selected = _selective_logps_fp32(student_logits, ids)
    teacher_selected = _selective_logps_fp32(teacher_logits, ids)
    p = torch.log_softmax(student_logits.float(), dim=-1)
    q = torch.log_softmax(teacher_logits.float(), dim=-1)
    kl = (p.exp() * (p - q)).sum(-1)
    raw = teacher_selected - student_selected
    values = {"student_logps": student_selected, "teacher_logps": teacher_selected,
              "raw": raw, "reverse_kl": kl, "centered": raw + kl}
    if any(not torch.isfinite(v).all() for v in values.values()):
        raise ValueError("Non-finite distillation scores")
    return {name: value[0].cpu().tolist() for name, value in values.items()}


def load_hf(path, device, dtype, revision=None):
    import torch
    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM.from_pretrained(path, dtype=getattr(torch, dtype),
             device_map=device, trust_remote_code=True, revision=revision).eval()


def score_condition(args, condition, teacher_path):
    import torch
    from transformers import AutoTokenizer
    out = Path(args.output_dir)
    rollouts = read_json(out / "rollouts.json")
    cohort = read_json(out / "cohort.json")
    provenance = fingerprint(rollouts)
    tokenizer = AutoTokenizer.from_pretrained(args.student, trust_remote_code=True)
    teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_path, trust_remote_code=True)
    if tokenizer_identity(tokenizer) != cohort["tokenizers"]["student"]:
        raise ValueError("Scoring student tokenizer changed")
    teacher_meta = check_tokenizers(tokenizer, teacher_tokenizer)
    if teacher_meta != cohort["tokenizers"]["opd" if condition == "opd" else "opsd"]:
        raise ValueError("Scoring teacher tokenizer changed")
    pending = []
    for row in rollouts["rows"]:
        path = out / "scores" / condition / f"{row['rollout_id']}.json"
        if path.exists():
            cached = read_json(path)
            if cached["rollout_fingerprint"] != provenance:
                raise ValueError("Cached scores belong to different rollouts")
        else:
            pending.append((row, path))
    if not pending:
        return
    revisions = cohort["model_revisions"]
    student = load_hf(args.student, args.student_device, args.dtype, revisions["student"])
    teacher = student if teacher_path == args.student else load_hf(
        teacher_path, args.teacher_device, args.dtype, revisions["opd" if condition == "opd" else "opsd"])
    with torch.inference_mode():
        for index, (row, path) in enumerate(pending):
            ids = row["completion_ids"]
            student_chunks = causal_logit_chunks(student, row["student_prompt_ids"], ids, args.score_chunk_size)
            teacher_chunks = causal_logit_chunks(teacher, row["teacher_prompt_ids"][condition], ids,
                                                 args.score_chunk_size)
            scores = defaultdict(list)
            for (start, p), (other_start, q) in zip(student_chunks, teacher_chunks, strict=True):
                if start != other_start:
                    raise ValueError("Teacher/student causal positions differ")
                chunk = distillation_scores(p, q, ids[start:start + p.shape[1]])
                for name, values in chunk.items():
                    scores[name].extend(values)
            write_json(path, {"rollout_id": row["rollout_id"], "rollout_fingerprint": provenance,
                             "condition": condition, **scores})
            print(f"Scored {condition}: {index + 1}/{len(pending)}", flush=True)


def counts_for_prefix(out, key, k):
    cache = read_json(out / "mc" / f"{key}.json")
    if cache["prefix"]["key"] != key or len(cache["draws"]) < k:
        raise ValueError(f"Missing MC samples for prefix {key}")
    draws = cache["draws"][:k]
    if [d["sample_idx"] for d in draws] != list(range(k)):
        raise ValueError("MC cache contains missing or reordered samples")
    return {"successes": sum(d["reward"] for d in draws), "samples": k}


def join_comparisons(row, plan_row, score_rows, counts):
    """Join exclusively by canonical rollout ID and zero-based completion index."""
    length = len(row["completion_ids"])
    if plan_row["rollout_id"] != row["rollout_id"]:
        raise ValueError("Plan rollout ID mismatch")
    for score in score_rows.values():
        if score["rollout_id"] != row["rollout_id"] or any(
            len(score[name]) != length for name in ("student_logps", "teacher_logps", "raw", "centered", "reverse_kl")
        ):
            raise ValueError("Scores are not aligned to the canonical completion")
    reference = next(iter(score_rows.values()))["student_logps"]
    for score in score_rows.values():
        if any(abs(a - b) > 1e-5 for a, b in zip(reference, score["student_logps"], strict=True)):
            raise ValueError("Student log probabilities changed across teacher conditions")
    rows = []
    for selection, spans in (("uniform", [(t, t + 1) for t in plan_row["uniform_positions"]]),
                             ("steps", plan_row["segments"])):
        for start, end in spans:
            before_key, after_key = plan_row["prefix_keys"][str(start)], plan_row["prefix_keys"][str(end)]
            before = counts[before_key]
            after = counts[after_key] if after_key is not None else None
            vine = mc_advantage(before, after, row["reward"] if end == length else None)
            rows.append({"rollout_id": row["rollout_id"], "question_id": row["question_id"],
                         "selection": selection, "start": start, "end": end,
                         "token_indices": list(range(start, end)),
                         "token_ids": row["completion_ids"][start:end],
                         "reward": row["reward"], "truncated": row["truncated"],
                         "vine": {**vine, "before_key": before_key, "after_key": after_key,
                                  "before_counts": before, "after_counts": after,
                                  "credit": "single_token" if selection == "uniform" else "segment_broadcast"},
                         "distillation": {name: {field: score[field][start:end]
                             for field in ("student_logps", "teacher_logps", "raw", "centered", "reverse_kl")}
                             for name, score in score_rows.items()}})
    return rows


def paired_metrics(x, y, resolved):
    import numpy as np
    from scipy.stats import spearmanr
    x, y, resolved = np.asarray(x), np.asarray(y), np.asarray(resolved, dtype=bool)
    nonconstant = len(x) > 1 and np.ptp(x) > 0 and np.ptp(y) > 0
    agreement = np.sign(x) == np.sign(y)
    return {"num_observations": len(x), "mean_distillation": float(x.mean()),
            "mean_vine": float(y.mean()),
            "pearson": float(np.corrcoef(x, y)[0, 1]) if nonconstant else None,
            "spearman": float(spearmanr(x, y).statistic) if nonconstant else None,
            "sign_agreement": float(agreement.mean()), "num_resolved": int(resolved.sum()),
            "resolved_sign_agreement": float(agreement[resolved].mean()) if resolved.any() else None}


def summarize(rows, bootstrap_samples, seed):
    """Keep uniform-token and broadcast-step views separate; cluster by question."""
    import numpy as np
    groups = defaultdict(list)
    for row in rows:
        vine = row["vine"]["advantage"]
        lo, hi = row["vine"]["ci95"]
        for condition, values in row["distillation"].items():
            for signal in ("raw", "centered"):
                # Step-token view deliberately weights segments by their number of tokens.
                for value in values[signal]:
                    groups[(row["selection"], "token", condition, signal)].append(
                        (row["question_id"], value, vine, lo > 0 or hi < 0))
                if row["selection"] == "steps":
                    groups[("steps", "segment_mean", condition, signal)].append(
                        (row["question_id"], sum(values[signal]) / len(values[signal]), vine, lo > 0 or hi < 0))
    summaries = []
    for key, observations in sorted(groups.items()):
        qids, x, y, resolved = zip(*observations)
        metrics = paired_metrics(x, y, resolved)
        clusters = defaultdict(list)
        for index, qid in enumerate(qids):
            clusters[qid].append(index)
        rng = random.Random(keyed_seed(seed, "bootstrap", key))
        intervals = defaultdict(list)
        if len(clusters) >= 2:
            q = sorted(clusters)
            for _ in range(bootstrap_samples):
                indices = [i for chosen in rng.choices(q, k=len(q)) for i in clusters[chosen]]
                sample = paired_metrics([x[i] for i in indices], [y[i] for i in indices],
                                        [resolved[i] for i in indices])
                for name in ("pearson", "spearman", "sign_agreement", "resolved_sign_agreement"):
                    if sample[name] is not None:
                        intervals[name].append(sample[name])
        metrics["question_bootstrap_ci95"] = {name: np.quantile(values, [0.025, 0.975]).tolist()
                                              for name, values in intervals.items()}
        metrics["num_questions"] = len(clusters)
        summaries.append(dict(zip(("selection", "unit", "condition", "signal"), key)) | metrics)
    return summaries


def aggregate(args):
    out = Path(args.output_dir)
    spec, rollouts = read_json(out / "plan.json"), read_json(out / "rollouts.json")
    if spec["rollout_fingerprint"] != fingerprint(rollouts):
        raise ValueError("Plan does not match canonical rollouts")
    counts = {p["key"]: counts_for_prefix(out, p["key"], args.mc_samples) for p in spec["prefixes"]}
    plans = {p["rollout_id"]: p for p in spec["rollouts"]}
    conditions = ["opd"] + [f"opsd_{m}" for m in args.pi_modes]
    comparisons = []
    for row in rollouts["rows"]:
        scores = {condition: read_json(out / "scores" / condition / f"{row['rollout_id']}.json")
                  for condition in conditions}
        if any(s["rollout_fingerprint"] != fingerprint(rollouts) for s in scores.values()):
            raise ValueError("Scores do not match canonical rollouts")
        comparisons.extend(join_comparisons(row, plans[row["rollout_id"]], scores, counts))
    target = out / f"k-{args.mc_samples}"
    target.mkdir(parents=True, exist_ok=True)
    temp = target / f".comparisons.{os.getpid()}.tmp"
    with temp.open("w") as file:
        for row in comparisons:
            file.write(json.dumps(row, allow_nan=False) + "\n")
    temp.replace(target / "comparisons.jsonl")
    write_json(target / "summary.json", {
        "mc_samples": args.mc_samples, "num_rollouts": len(rollouts["rows"]),
        "num_prefixes": len(counts), "rollout_fingerprint": fingerprint(rollouts),
        "pass_at_1": sum(r["reward"] for r in rollouts["rows"]) / len(rollouts["rows"]),
        "bootstrap_samples": args.bootstrap_samples,
        "uncertainty_note": "Question bootstrap conditions on the cached MC draws; shared prefix noise "
                            "and segment broadcasts are not independent observations.",
        "comparisons": summarize(comparisons, args.bootstrap_samples, args.seed)})
    print(f"Saved paired comparisons and summary to {target}", flush=True)


def run_spawned(target, *args):
    process = multiprocessing.get_context("spawn").Process(target=target, args=args)
    process.start()
    process.join()
    if process.exitcode != 0:
        raise RuntimeError(f"{target.__name__} failed with exit code {process.exitcode}")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student", required=True, help="Frozen student checkpoint/model path")
    parser.add_argument("--opd-teacher", required=True, help="Frozen external teacher checkpoint/model path")
    parser.add_argument("--opsd-teacher", help="Frozen self-teacher; defaults to --student")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--phase", choices=("all", "prepare", "generate", "plan", "mc", "score", "aggregate"), default="all")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--dataset", default="deepmath", help="Registered training/eval dataset")
    source.add_argument("--cohort", help="JSON/JSONL rows or saved HF Dataset; hint/rollout PI require corresponding columns")
    parser.add_argument("--pi-modes", nargs="+", choices=PI_MODES, default=["answer", "full", "hint"],
                        help="OPSD PI modes (default: answer full hint)")
    parser.add_argument("--selection-modes", nargs="+", choices=("uniform", "steps"), default=["uniform", "steps"])
    parser.add_argument("--num-problems", type=int, default=32)
    parser.add_argument("--n", type=int, default=4, help="Original student rollouts per problem")
    parser.add_argument("--num-token-samples", type=int, default=32, help="Uniform samples per rollout, without replacement")
    parser.add_argument("--mc-samples", "--K", dest="mc_samples", type=int, default=16, help="Continuations per distinct prefix")
    parser.add_argument("--min-segment-tokens", type=int, default=32)
    parser.add_argument("--max-segment-tokens", type=int, default=256)
    parser.add_argument("--max-completion-length", type=int, default=8192)
    parser.add_argument("--max-model-len", type=int, default=32000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--mc-batch-size", type=int, default=32)
    parser.add_argument("--save-mc-completions", action="store_true")
    parser.add_argument("--student-device", default="cuda:0", help="HF device/device_map, including cpu or auto")
    parser.add_argument("--teacher-device", default="cuda:1", help="HF device/device_map; may equal student device if memory permits")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--score-chunk-size", type=int, default=128)
    parser.add_argument("--bootstrap-samples", type=int, default=200)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.opsd_teacher = args.opsd_teacher or args.student
    for name in ("num_problems", "n", "num_token_samples", "mc_samples", "min_segment_tokens",
                 "max_segment_tokens", "max_completion_length", "max_model_len", "tensor_parallel_size",
                 "mc_batch_size", "score_chunk_size"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.min_segment_tokens > args.max_segment_tokens:
        parser.error("--min-segment-tokens must not exceed --max-segment-tokens")
    if args.max_model_len <= args.max_completion_length:
        parser.error("--max-model-len must exceed --max-completion-length")
    if not 0 < args.gpu_memory_utilization < 1 or args.bootstrap_samples < 0:
        parser.error("Require 0 < gpu-memory-utilization < 1 and bootstrap-samples >= 0")
    args.pi_modes = sorted(set(args.pi_modes))
    args.selection_modes = sorted(set(args.selection_modes))
    ensure_manifest(args)
    phases = ("prepare", "generate", "plan", "mc", "score", "aggregate") if args.phase == "all" else (args.phase,)
    for phase in phases:
        if phase == "score":
            run_spawned(score_condition, args, "opd", args.opd_teacher)
            for mode in args.pi_modes:
                run_spawned(score_condition, args, f"opsd_{mode}", args.opsd_teacher)
        else:
            run_spawned(globals()[phase], args)


if __name__ == "__main__":
    main()
