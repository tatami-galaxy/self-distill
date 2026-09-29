# Self-teacher pass@k across PI types

Compare the same frozen model under six conditions on a common question set:

| CLI condition | Privileged information |
| --- | --- |
| `none` | Question only |
| `answer` | Gold final answer |
| `rollout` | One fixed, unverified cached attempt from the same model |
| `full` | Complete reference demonstration, including thinking and final solution |
| `solution` | Complete reference response after `</think>`, without the thinking trace |
| `hint` | Standard self-generated hint cached by `utils.gen_hints` |

`solution` uses the same worked-solution prompt wrapper as `full`, but removes
all reference text through `</think>`. It keeps the entire worked final response,
not just the boxed answer. Missing, repeated, or empty thinking boundaries are
rejected rather than falling back to the complete trace.

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

The standard hint comes from `utils.gen_hints`, the same cache used by SDFT's
`--pi-mode hint`. Generate it first if it is not already available:

```sh
CUDA_VISIBLE_DEVICES=7 .venv/bin/python -m utils.gen_hints \
  --model Qwen/Qwen3-1.7B --dataset deepmath --max-samples 20000
```

Repeat for `Qwen/Qwen3-4B`. By default the cache lives at
`data/pi/hint/deepmath/<model-slug>/`. Without `--cohort-dir`, pass@k samples this
cache using `--seed`, validates its generator/model and dataset stamps, rejoins
reference solutions, and attaches fixed rollout PI from `--rollout-pi-root`.
Use a positive `--num-problems` in this mode.

With `--cohort-dir`, reuse the prepared demo-gain artifacts:

- `results/demo_gain/solution/Qwen3-1.7B/{manifest.json,cohort.jsonl}`
- `results/demo_gain/solution/Qwen3-4B/{manifest.json,cohort.jsonl}`

Cohort preparation already copied the standard cache's hint into each row's
`hint` field. Pass@k uses that saved text directly; it does not require a `hints/`
subdirectory or a separate hint-generation stage. A missing or empty standard
hint is an error when the `hint` arm is requested. The evaluator checks the
cohort checksum, model identity, and tokenizer identity, and loads the model at
the cohort's recorded revision. The original hint cache records the generator's
model identifier, not an independently pinned revision.

The rollout comes directly from the prepared cohort with its original fixed
sample index; correctness does not select it. The evaluator never generates or
repairs PI. Hint-variant validity filters and `--hint-sample-idx` no longer apply.

Then require every PI prompt to fit within
`max_model_len - max_tokens`, reserving the full response budget. No prompt is
truncated. Every condition within a model uses identical questions and a shared
denominator. The cohort manifest's order is preserved; `--num-problems 0` uses all
valid questions, while a positive value caps the valid list before context filtering.

Preflight reports the actual retained count from the completed artifacts. This
is a selected training-cache diagnostic cohort. The source caches and context
filters can yield different question sets for each model; comparing their scores then
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
  --pi-modes none answer rollout full solution hint \
  --num-problems 0 \
  --n 8 --k 1 2 4 8 \
  --enable-thinking --max-tokens 8192 --max-model-len 40000 \
  --temperature 0.6 --top-p 0.95 --top-k 20 --seed 42 \
  --batch-size 16 --gpu-memory-utilization 0.9 \
  --output-dir results/passk_pi/default_hint
```

### Qwen3-4B

```sh
CUDA_VISIBLE_DEVICES=7 .venv/bin/python -m eval.passk_pi \
  --model Qwen/Qwen3-4B \
  --cohort-dir results/demo_gain/solution/Qwen3-4B \
  --pi-modes none answer rollout full solution hint \
  --num-problems 200 \
  --n 8 --k 1 2 4 8 \
  --enable-thinking --max-tokens 8192 --max-model-len 40000 \
  --temperature 0.6 --top-p 0.95 --top-k 20 --seed 42 \
  --batch-size 16 --gpu-memory-utilization 0.9 \
  --output-dir results/passk_pi/default_hint
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
    --pi-modes none answer rollout full solution hint \
    --num-problems 0 \
    --n 8 --k 1 2 4 8 \
    --enable-thinking --max-tokens 8192 --max-model-len 40000 \
    --temperature 0.6 --top-p 0.95 --top-k 20 --seed 42 \
    --output-dir results/passk_pi/default_hint --prepare-only
done
```

A small pilot can use `--num-problems 4 --n 2 --k 1 2` with a separate output
root such as `results/passk_pi/default_hint_pilot`.

## Resume and outputs

Rerun the same command to reuse completed generations. Each completed batch is
cached per prompt, so an interruption only loses unfinished batch work. Cache
keys include the exact prompt, gold answer, dataset, model revision and identity,
tokenizer identity, thinking mode, sampling settings, response budget, and vLLM
version. Raw responses and grading outcomes are retained. GPU placement and batch
size can change on resume; bitwise regeneration across execution layouts is not
guaranteed.

Use a new output directory if the cohort, PI conditions, or generation settings
change. In particular, use a new directory for this six-condition run instead
of a previous hint-variant run. `--force` recomputes generations under the same
run settings. Changing `--k` or `--paired-bootstrap-samples` can reuse the sampled
responses. A resumed run still initializes the vLLM model.

Each model writes its own directory:

```text
results/passk_pi/default_hint/Qwen_Qwen3-1.7B/
results/passk_pi/default_hint/Qwen_Qwen3-4B/
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

`answer`, `full`, and `solution` expose the gold answer, so success can reflect copying or
following supplied information. These conditions measure privileged self-teacher
behavior, not unaided mathematical capability. The fixed hints' source model
matches the evaluated model, but their source demonstration remains the external
reference used by the existing hint-generation experiment.
