Choose checkpoints using one fixed DeepMath set, then evaluate only the chosen
checkpoint on each OOD benchmark. The existing `eval/run_eval.py` is unchanged.

1. Create the shared 128-question split once. List every exact run directory you
   intend to compare so the split excludes the union of their training pools:

   ```sh
   .venv/bin/python -m eval.deepmath_validation \
     --exclude-run /mnt/data/ujan/self-distill/outputs/sdft/Qwen3-1.7B/deepmath_hint \
     --output data/validation/deepmath_128.json
   ```

   Multiple paths can follow `--exclude-run`. Selection is random with seed 42,
   without replacement, and deduplicated by whitespace-normalized question text.
   The saved JSON contains the actual questions, answers, identities, and exclusion
   provenance. Existing manifests are never overwritten. Use the **same manifest**
   for all compared runs; the selector checks each new run for overlap.

2. Sweep one run/hyperparameter setting:

   ```sh
   CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m eval.select_checkpoint \
     --run-dir /mnt/data/ujan/self-distill/outputs/sdft/Qwen3-1.7B/deepmath_hint \
     --validation-file data/validation/deepmath_128.json \
     --output-dir results/validation/Qwen3-1.7B/deepmath_hint
   ```

   Each numeric `checkpoint-N` gets one sampled solution per question. Exact ties
   favor the earliest checkpoint. `final` is omitted to avoid counting the last
   checkpoint twice. A fresh process per checkpoint releases GPU resources.
   Rerunning reuses completed compatible scores; `--phase summarize` only reads
   cached scores. Missing or incompatible scores prevent a partial selection.
   The output `selection.json` records all candidates and the chosen model path.

3. Run avg@16 on the selected checkpoint:

   ```sh
   CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m eval.run_avg16 \
     --selection results/validation/Qwen3-1.7B/deepmath_hint/selection.json \
     --dataset aime24 --algo sdft --variant hint
   ```

   Repeat with `--dataset aime25`. For a direct model evaluation, use `--model`
   and the explicit arm identifiers supported by `run_eval` (`--model-name`,
   `--train-dataset`, `--variant`, `--run`). Selected runs infer model name,
   training dataset, and run label from the selection record. Results go under
   `results/selected_ood/`, separate from historical checkpoint sweeps, and report
   only `accuracy` with `metric: "avg@16"`. This equals the historical pass@1
   estimator using 16 samples, not best-of-16 accuracy.

Both evaluation scripts reuse `run_eval.evaluate_model` without changing its
prompt, verifier, or decoding. They default to the same 32,000-token budget;
`--max-tokens` can change it, but keep it fixed across checkpoints. Dataset seed
42 fixes the validation questions; generation uses the existing evaluator's
vLLM defaults, including its engine seed, without promising bitwise reproducibility.
The evaluation cache records code, library versions, checkpoint file stamps,
question fingerprints, and generation settings. No training or GPU evaluation
is launched by preparing a split.

Exclusion is deliberately conservative. For hint/rollout SDFT it uses the recorded
hint cache and prefix; for other supported DeepMath runs it excludes their entire
configured source prefix before length filtering. Historical metadata does not
record actual consumed question IDs or dataset hashes. Current data/caches must
therefore still match the training sources, and a run configured on **all DeepMath**
leaves no verifiable held-out pool. The scripts fail rather than call overlapping
data held out. They do not modify future training loaders; future runs must also
reserve this set explicitly. DeepMath versus benchmark duplicates and pretraining
exposure are outside this audit.

With 128 questions, one correct answer changes selection accuracy by 0.78 percentage
points. In addition to sampling noise, repeated checkpoint/hyperparameter selection
can overfit this small set, and DeepMath rankings may differ from OOD rankings.
Use OOD results only after selection; do not choose a different checkpoint for each
AIME benchmark. Reporting only avg@16 does not itself reduce generation cost—the
savings come from evaluating fewer checkpoints on OOD data.

For a known missing historical hint cache, `--allow-missing-cache` permits split
creation while recording that run's overlap as **unverified**. Other available
training pools are still excluded. The selector permits this exception only for
runs explicitly recorded in that manifest. It does not bypass known overlap or
other provenance errors. The initial split in this workspace uses this exception
for the Qwen3-1.7B full-weight `t0.7_g6/checkpoint-100` hint source.
