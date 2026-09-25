# Self-teacher pass@k across PI types

Compare the same frozen model under seven conditions on a common question set:

| CLI condition | Privileged information |
| --- | --- |
| `none` | Question only |
| `answer` | Gold final answer |
| `rollout` | One fixed, unverified cached attempt from the same model |
| `full` | Complete reference demonstration, including thinking and final solution |
| `hint_short` | Cached short self-generated hint |
| `hint_medium` | Cached medium self-generated hint |
| `hint_detailed` | Cached detailed self-generated hint |

These are **free-running accuracy** evaluations. The model generates its own
thinking and final response; no reference solution tokens are teacher-forced.
Thinking is enabled in every condition. This differs from `demo_gain`, which
measures reference final-solution likelihood with an empty thinking prefix.

Each question gets eight independent sampled responses per condition, using
one fixed PI text throughout. Report pass@1, pass@2, pass@4, and pass@8 using
`1 - C(n-c, k) / C(n, k)`, where `c` is the number of correct responses among
`n=8`. The existing dataset-specific grader checks each response against the gold
answer. Results average equally over questions.

## Inputs and common cohort

Reuse the completed demo-gain artifacts:

- `results/demo_gain/solution/Qwen3-1.7B/{manifest.json,cohort.jsonl,hints/}`
- `results/demo_gain/solution/Qwen3-4B/{manifest.json,cohort.jsonl,hints/}`

Both `hints/manifest.json` and `hints/hints.jsonl` must exist. A directory containing
only incremental generation caches is not complete. Finish the demo-gain hint
generation stage before running this evaluator; see [demo_gain.md](demo_gain.md).
The pass@k evaluator does not generate or repair hints.

The evaluator checks cohort and hint checksums, demonstration identities, model
identity, model revision, and tokenizer identity. The hint generator must be the
same model as the self-teacher.

For every requested hint condition, select `--hint-sample-idx 0` before inspecting
validity. Exclude a question from **every arm** if any selected hint is invalid or
truncated. Missing hint artifacts are an error. No replacement hint is selected
based on validity or accuracy. The rollout is taken directly from the prepared
cohort, with its original fixed sample index; correctness does not select it.

Then require every PI prompt to fit within
`max_model_len - max_tokens`, reserving the full response budget. No prompt is
truncated. Every condition within a model uses identical questions and a shared
denominator. The cohort manifest's order is preserved; `--num-problems 0` uses all
valid questions, while a positive value caps the valid list before context filtering.

Preflight reports the actual retained count from the completed artifacts. This
is a selected training-cache diagnostic cohort. Each model applies its own hint
validity filters, so their question sets can differ; comparing their scores then
does not isolate a model-size effect.

## Run

Run from the repository root. Set `CUDA_VISIBLE_DEVICES` to the GPU chosen at
launch; the commands below use GPU 7 and can be run sequentially.

Settings are explicit: eight samples, an 8,192-token response budget including
thinking, temperature 0.6, top-p 0.95, top-k 20, seed 42, and BF16 weights.
Model-provided generation defaults are disabled in favor of these settings.

### Qwen3-1.7B

```sh
CUDA_VISIBLE_DEVICES=7 .venv/bin/python -m eval.passk_pi \
  --model Qwen/Qwen3-1.7B \
  --cohort-dir results/demo_gain/solution/Qwen3-1.7B \
  --pi-modes none answer rollout full hint_short hint_medium hint_detailed \
  --num-problems 0 --hint-sample-idx 0 \
  --n 8 --k 1 2 4 8 \
  --enable-thinking --max-tokens 8192 --max-model-len 40000 \
  --temperature 0.6 --top-p 0.95 --top-k 20 --seed 42 \
  --batch-size 16 --gpu-memory-utilization 0.9 \
  --output-dir results/passk_pi/demo_gain
```

### Qwen3-4B

```sh
CUDA_VISIBLE_DEVICES=7 .venv/bin/python -m eval.passk_pi \
  --model Qwen/Qwen3-4B \
  --cohort-dir results/demo_gain/solution/Qwen3-4B \
  --pi-modes none answer rollout full hint_short hint_medium hint_detailed \
  --num-problems 0 --hint-sample-idx 0 \
  --n 8 --k 1 2 4 8 \
  --enable-thinking --max-tokens 8192 --max-model-len 40000 \
  --temperature 0.6 --top-p 0.95 --top-k 20 --seed 42 \
  --batch-size 16 --gpu-memory-utilization 0.9 \
  --output-dir results/passk_pi/demo_gain
```

### CPU preflight

Append `--prepare-only` to either command to validate inputs and context lengths
and write `run_meta.json` without loading model weights or generating responses.
For example, check both models:

```sh
for model in Qwen3-1.7B Qwen3-4B; do
  CUDA_VISIBLE_DEVICES='' .venv/bin/python -m eval.passk_pi \
    --model "Qwen/$model" \
    --cohort-dir "results/demo_gain/solution/$model" \
    --pi-modes none answer rollout full hint_short hint_medium hint_detailed \
    --num-problems 0 --hint-sample-idx 0 \
    --n 8 --k 1 2 4 8 \
    --enable-thinking --max-tokens 8192 --max-model-len 40000 \
    --temperature 0.6 --top-p 0.95 --top-k 20 --seed 42 \
    --output-dir results/passk_pi/demo_gain --prepare-only
done
```

A small pilot can use `--num-problems 4 --n 2 --k 1 2` with a separate output
root such as `results/passk_pi/demo_gain_pilot`.

## Resume and outputs

Rerun the same command to reuse completed generations. Each completed batch is
cached per prompt, so an interruption only loses unfinished batch work. Cache
keys include the exact prompt, gold answer, dataset, model revision and identity,
tokenizer identity, thinking mode, sampling settings, response budget, and vLLM
version. Raw responses and grading outcomes are retained. GPU placement and batch
size can change on resume; bitwise regeneration across execution layouts is not
guaranteed.

Use a new output directory if the cohort, PI conditions, or generation settings
change. `--force` recomputes generations under the same run settings. Changing
`--k` or `--paired-bootstrap-samples` can reuse the sampled responses. A resumed
run still initializes the vLLM model.

Each model writes its own directory:

```text
results/passk_pi/demo_gain/Qwen_Qwen3-1.7B/
results/passk_pi/demo_gain/Qwen_Qwen3-4B/
```

Files:

- `run_meta.json`: cohort identities, retained question IDs/source indices,
  validity/context exclusions, and generation settings.
- `<condition>_results.json`: question IDs, correct/sample counts, truncation
  counts, and every response's text, correctness, finish reason, and token count.
  Written after each condition completes.
- `score_cache/passk_generation/`: checksummed, resumable per-prompt responses.
- `passk_pi_summary.json`: accuracy table, question-bootstrap 95% intervals,
  paired differences against `none`, truncation rates, and run metadata. Written
  after all conditions complete.

`paired_against_none[condition]["pass@k"]` contains `mean`, `ci95`,
`delta_vs_none`, and `delta_ci95`. Values are fractions; multiply differences by
100 for percentage points. The bootstrap resamples questions with all conditions
paired, using 10,000 draws by default. It does not resample hint generation or
independent generation seeds. `--save-samples` additionally embeds compact
per-question correct counts in the summary; response files are always saved.

A response that hits the token limit remains in the denominator and is graded
normally. `truncation_rate` reports the fraction with finish reason `length`;
it does not automatically mark such responses wrong.

`answer` and `full` expose the gold answer, so success can reflect copying or
following supplied information. These conditions measure privileged self-teacher
behavior, not unaided mathematical capability. The fixed hints' source model
matches the evaluated model, but their source demonstration remains the external
reference used by the existing hint-generation experiment.
