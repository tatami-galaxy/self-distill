"""Small temperature sweep using the unchanged question + demonstration hint prompt.

CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m eval.hint_diversity \
    --model Qwen/Qwen3-1.7B --temperatures 1.0 1.2 1.4

Defaults: 10 seeded random DeepMath questions, 8 hints/question/temperature,
128 generated tokens, thinking disabled, top-p=1 and no top-k filtering.
The same questions and demonstrations are used at every temperature. No solver,
judge, sufficiency scoring, or training runs. Lexical metrics are descriptive:
inspect report.html to assess mathematical/semantic diversity.

--dry-run prepares the cohort with a tokenizer but no model weights.
--cohort-dir can read an existing Hugging Face dataset saved to disk, with
question/final_answer/solution columns, instead of loading the source dataset.
--phase summarize rebuilds the report using only saved JSON, without GPU imports.
Completed temperatures are reusable; interrupted temperatures are regenerated.
Use a new output directory when changing settings, or --force to resample the
same experiment. GPU layout can change; bitwise regeneration is not guaranteed.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import unicodedata
from collections import Counter
from html import escape
from itertools import combinations
from pathlib import Path

from eval.hint_compare_cache import digest, model_identity


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)
    )
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text())


def normalized(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def lexical_tokens(text):
    # Keep numbers, LaTeX commands, and mathematical symbols in the comparison.
    return re.findall(r"\\[a-z]+|\w+|[^\w\s]", normalized(text))


def mean(values):
    values = [v for v in values if v is not None]
    return statistics.mean(values) if values else None


def diversity(rows):
    """Within-question lexical metrics; fewer than two hints gives no pair estimate."""
    sets = []
    for row in rows:
        tokens = lexical_tokens(row["hint"])
        sets.append({tuple(tokens[i : i + 3]) for i in range(len(tokens) - 2)})
    overlaps = [len(a & b) / len(a | b) for a, b in combinations(sets, 2) if a | b]
    texts = [normalized(row["hint"]) for row in rows]
    pairs = list(combinations(texts, 2))
    return {
        "n_hints": len(rows),
        "n_unique_normalized": len(set(texts)),
        "unique_fraction": len(set(texts)) / len(rows) if rows else None,
        "duplicate_pair_fraction": mean([float(a == b) for a, b in pairs]),
        "mean_pairwise_trigram_jaccard": mean(overlaps),
        "n_trigram_pairs": len(overlaps),
        "mean_tokens": mean([row["n_tokens"] for row in rows]),
    }


def summarize(rows):
    per_question = {}
    for qid in sorted({r["question_id"] for r in rows}):
        group = [r for r in rows if r["question_id"] == qid]
        valid = [r for r in group if not r["invalid_reason"] and not r["truncated"]]
        per_question[qid] = {
            "all": diversity(group),
            "valid_complete": diversity(valid),
        }
    result = {"per_question": per_question}
    for subset in ("all", "valid_complete"):
        groups = [q[subset] for q in per_question.values()]
        result[subset] = {
            "n_hints": sum(g["n_hints"] for g in groups),
            "n_questions_with_hints": sum(g["n_hints"] > 0 for g in groups),
            "n_questions_with_pairs": sum(g["n_hints"] >= 2 for g in groups),
            **{
                key: mean([g[key] for g in groups])
                for key in (
                    "unique_fraction",
                    "duplicate_pair_fraction",
                    "mean_pairwise_trigram_jaccard",
                    "mean_tokens",
                )
            },
        }
    result["invalid_reasons"] = dict(
        Counter(r["invalid_reason"] for r in rows if r["invalid_reason"])
    )
    result["truncation_fraction"] = mean([float(r["truncated"]) for r in rows])
    return result


def temperature_path(root, temperature):
    return root / f"temperature_{temperature:g}.json"


def load_samples(path, manifest, temperature):
    data = read_json(path)
    if (
        data["manifest_fingerprint"] != digest(manifest)
        or data["temperature"] != temperature
    ):
        raise ValueError(f"Incompatible samples: {path}")
    rows = data["hints"]
    expected = {
        (q["question_id"], i)
        for q in manifest["cohort"]
        for i in range(manifest["config"]["hints_per_question"])
    }
    keys = [(r["question_id"], r["sample_idx"]) for r in rows]
    if (
        len(keys) != len(expected)
        or set(keys) != expected
        or digest(rows) != data["hints_fingerprint"]
    ):
        raise ValueError(
            f"Incomplete or changed samples: {path}; use --force to regenerate"
        )
    return rows


def report(root, manifest):
    summaries, samples = {}, {}
    for temperature in manifest["config"]["temperatures"]:
        rows = load_samples(temperature_path(root, temperature), manifest, temperature)
        samples[temperature] = rows
        summaries[str(temperature)] = summarize(rows)
    write_json(
        root / "summary.json",
        {
            "metric_note": "Question-balanced lexical statistics, not semantic diversity. "
            "Lower Jaccard means less shared text. Valid statistics condition on validity; "
            "check support and truncation before comparing temperatures.",
            "temperatures": summaries,
        },
    )
    html = [
        '<!doctype html><meta charset="utf-8"><title>Hint diversity</title>',
        ("<style>body{font:16px system-ui;margin:2rem;max-width:1600px} "
        ".grid{display:flex;gap:1rem;align-items:flex-start;overflow-x:auto} "
        ".arm{flex:1;min-width:280px} pre{white-space:pre-wrap;overflow-wrap:anywhere} "
        "article{padding:.7rem;border:1px solid #ddd;margin:.5rem 0} "
        ".flag{color:#a22} summary{cursor:pointer} table{border-collapse:collapse} "
        "td,th{padding:.5rem;border:1px solid #ddd}</style>"),
        "<h1>Hint diversity by temperature</h1>",
        ("<p>Same question, demonstration and generator prompt at every temperature. "
        "Metrics measure text overlap; different wording does not establish different mathematical help. "
        "Invalid and truncated outputs are retained below.</p>"),
        ("<table><tr><th>Temperature</th><th>Valid complete</th><th>Unique fraction (all)</th>"
        "<th>Trigram Jaccard (all / valid)</th><th>Mean tokens</th><th>Truncated</th></tr>"),
    ]
    fmt = lambda x: "n/a" if x is None else f"{x:.3f}"
    for temp, s in summaries.items():
        a, v = s["all"], s["valid_complete"]
        values = [
            temp,
            f"{v['n_hints']}/{a['n_hints']}",
            fmt(a["unique_fraction"]),
            f"{fmt(a['mean_pairwise_trigram_jaccard'])} / {fmt(v['mean_pairwise_trigram_jaccard'])}",
            fmt(a["mean_tokens"]),
            fmt(s["truncation_fraction"]),
        ]
        html.append("<tr>" + "".join(f"<td>{escape(x)}</td>" for x in values) + "</tr>")
        print(
            f"T={temp}: valid={values[1]}, unique={values[2]}, Jaccard={values[3]}, "
            f"tokens={values[4]}, truncated={values[5]}"
        )
    html.append("</table>")
    for index, question in enumerate(manifest["cohort"], 1):
        html.append(
            f"<h2>Question {index}</h2><pre>{escape(question['question'])}</pre>"
            f"<details><summary>Reference answer and demonstration</summary>"
            f"<pre>{escape(str(question['final_answer']))}\n\n{escape(question['solution'])}</pre></details>"
            '<div class="grid">'
        )
        for temperature, rows in samples.items():
            html.append(f'<section class="arm"><h3>Temperature {temperature:g}</h3>')
            for row in rows:
                if row["question_id"] != question["question_id"]:
                    continue
                flags = ", ".join(
                    filter(
                        None,
                        [
                            row["invalid_reason"],
                            "truncated" if row["truncated"] else "",
                        ],
                    )
                )
                html.append(
                    f"<article>Hint {row['sample_idx'] + 1} · {row['n_tokens']} tokens "
                    f'<span class="flag">{escape(flags)}</span><pre>{escape(row["hint"])}</pre></article>'
                )
            html.append("</section>")
        html.append("</div>")
    (root / "report.html").write_text("\n".join(html))


def prepare(args, config):
    from transformers import AutoTokenizer

    from utils import load_train_dataset
    from utils.gen_hints import build_messages
    from utils.model_adapters import resolve_model_adapter

    tokenizer = AutoTokenizer.from_pretrained(
        resolve_model_adapter(args.model).base_model, trust_remote_code=True
    )
    if args.cohort_dir:
        from datasets import load_from_disk

        source = load_from_disk(args.cohort_dir)
        required = {"question", "final_answer", "solution"}
        if not required.issubset(source.column_names):
            raise ValueError(f"Local cohort needs columns {sorted(required)}")
    else:
        source = load_train_dataset(args.dataset, require_solution=True)
    source = source.shuffle(seed=args.seed)
    cohort, seen, skipped = [], set(), 0
    for row in source:
        if any(
            row[key] is None or not str(row[key]).strip()
            for key in ("question", "final_answer", "solution")
        ):
            continue
        qid = digest([row["question"], str(row["final_answer"])])[:24]
        if qid in seen:
            continue
        seen.add(qid)
        messages = build_messages(row["question"], row["solution"], args.dataset)
        ids = tokenizer.apply_chat_template(
            [messages],
            tokenize=True,
            return_dict=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )["input_ids"][0]
        if len(ids) + args.max_tokens > args.max_model_len:
            skipped += 1
            continue
        cohort.append(
            {
                "question_id": qid,
                "question": row["question"],
                "final_answer": str(row["final_answer"]),
                "solution": row["solution"],
                "messages": messages,
                "prompt_ids": list(ids),
            }
        )
        if len(cohort) == args.num_questions:
            break
    if len(cohort) != args.num_questions:
        raise ValueError(
            f"Only {len(cohort)} unique questions fit; requested {args.num_questions}"
        )
    return {
        "config": config,
        "cohort": cohort,
        "skipped_overlong": skipped,
        "tokenizer": tokenizer.name_or_path,
    }


def build_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--model", default="Qwen/Qwen3-1.7B")
    p.add_argument("--dataset", default="deepmath")
    p.add_argument(
        "--cohort-dir",
        help="Optional saved HF cohort with question/final_answer/solution columns; --dataset identifies its task.",
    )
    p.add_argument("--num-questions", type=int, default=10)
    p.add_argument("--hints-per-question", type=int, default=8)
    p.add_argument("--temperatures", nargs="+", type=float, default=[1.0, 1.2, 1.4])
    p.add_argument(
        "--max-tokens",
        type=int,
        default=128,
        help="Matches constrained hint generation; truncation is reported.",
    )
    p.add_argument("--max-model-len", type=int, default=32768)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--phase", choices=["all", "summarize"], default="all")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--force", action="store_true")
    return p


def main():
    parser = build_parser()
    args = parser.parse_args()
    if (
        min(
            args.num_questions,
            args.hints_per_question,
            args.max_tokens,
            args.tensor_parallel_size,
        )
        < 1
    ):
        parser.error("Question, hint, token and parallelism counts must be positive")
    if (
        args.max_model_len <= args.max_tokens
        or not 0 < args.gpu_memory_utilization <= 1
    ):
        parser.error("Invalid context length or GPU memory utilization")
    if any(not math.isfinite(t) or t <= 0 for t in args.temperatures):
        parser.error("Temperatures must be finite and positive")
    args.temperatures = list(dict.fromkeys(args.temperatures))
    if args.cohort_dir:
        args.cohort_dir = str(Path(args.cohort_dir).resolve())
    if len({f"{t:g}" for t in args.temperatures}) != len(args.temperatures):
        parser.error("Temperatures are too close to distinguish in output filenames")
    root = Path(
        args.output_dir
        or f"results/hint_diversity/{args.model.rstrip('/').split('/')[-1]}"
    )
    manifest_path = root / "manifest.json"
    if args.phase == "summarize":
        if args.force or args.dry_run:
            parser.error("--force and --dry-run apply only to --phase all")
        report(root, read_json(manifest_path))
        return

    from utils.gen_hints import HINT_VALIDATION_VERSION, build_messages, leaks_answer

    config = {
        key: getattr(args, key)
        for key in (
            "model",
            "dataset",
            "num_questions",
            "hints_per_question",
            "temperatures",
            "max_tokens",
            "max_model_len",
            "seed",
            "cohort_dir",
        )
    }
    config.update(
        schema_version=1,
        model_identity=model_identity(args.model),
        validation_version=HINT_VALIDATION_VERSION,
        prompt_template=build_messages("{question}", "{solution}", args.dataset),
        top_p=1.0,
        top_k=-1,
        min_p=0.0,
        repetition_penalty=1.0,
        enable_thinking=False,
    )
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        if manifest["config"] != config:
            raise ValueError(
                "Run settings differ from the saved manifest; use a new --output-dir"
            )
    else:
        manifest = prepare(args, config)
        write_json(manifest_path, manifest)
    print(
        f"{len(manifest['cohort'])} questions × {args.hints_per_question} hints × "
        f"{len(args.temperatures)} temperatures; max {args.max_tokens} tokens/hint"
    )
    pending = []
    for temperature in args.temperatures:
        path = temperature_path(root, temperature)
        if path.exists() and not args.force:
            load_samples(path, manifest, temperature)
        else:
            pending.append(temperature)
    if args.dry_run:
        print(f"Pending temperatures: {pending}; manifest: {manifest_path}")
        return
    if pending:
        from vllm import LLM, SamplingParams

        from utils.model_adapters import vllm_model_and_adapter

        kwargs, adapter, _ = vllm_model_and_adapter(args.model)
        llm = LLM(
            **kwargs,
            generation_config="vllm",
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            tensor_parallel_size=args.tensor_parallel_size,
            seed=args.seed,
            trust_remote_code=True,
        )
        prompts = [{"prompt_token_ids": q["prompt_ids"]} for q in manifest["cohort"]]
        for temperature in pending:
            sampling = SamplingParams(
                n=args.hints_per_question,
                temperature=temperature,
                top_p=1.0,
                top_k=-1,
                min_p=0.0,
                repetition_penalty=1.0,
                max_tokens=args.max_tokens,
                seed=args.seed,
            )
            outputs = llm.generate(prompts, sampling, lora_request=adapter)
            rows = []
            for question, output in zip(manifest["cohort"], outputs, strict=True):
                if len(output.outputs) != args.hints_per_question:
                    raise RuntimeError("vLLM returned the wrong number of hints")
                for index, candidate in enumerate(output.outputs):
                    hint = candidate.text
                    reason = (
                        "empty"
                        if not hint.strip()
                        else "thinking"
                        if "<think>" in hint or "</think>" in hint
                        else "answer_leak"
                        if leaks_answer(hint, question["final_answer"], args.dataset)
                        else ""
                    )
                    rows.append(
                        {
                            "question_id": question["question_id"],
                            "sample_idx": index,
                            "hint": hint,
                            "token_ids": list(candidate.token_ids),
                            "n_tokens": len(candidate.token_ids),
                            "invalid_reason": reason,
                            "truncated": candidate.finish_reason == "length",
                            "finish_reason": candidate.finish_reason,
                        }
                    )
            write_json(
                temperature_path(root, temperature),
                {
                    "manifest_fingerprint": digest(manifest),
                    "temperature": temperature,
                    "hints": rows,
                    "hints_fingerprint": digest(rows),
                },
            )
    report(root, manifest)
    print(
        f"Inspect {root / 'report.html'}; machine-readable statistics: {root / 'summary.json'}"
    )


if __name__ == "__main__":
    main()
