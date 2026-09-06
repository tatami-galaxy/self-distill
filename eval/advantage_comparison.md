OPSD PI defaults to `answer full hint`. With `--dataset`, hints are loaded from
the existing cache for `--opsd-teacher` (which defaults to `--student`) and matched
by both question and final answer. The cache must already exist; missing hints
are excluded from the common cohort. With `--cohort`, provide `question`,
`final_answer`, `solution`, and `hint` for these default modes. Override
`--pi-modes` to select a different set.

## Definitions

Let p be the frozen, unprivileged student, q the external OPD teacher or the
frozen OPSD teacher with PI, and y the original student completion.

- Raw distillation coefficient: `d_t = log q(y_t | prefix) - log p(y_t | prefix)`.
- Exact prefix-centered coefficient: `d_t + KL(p(. | prefix) || q(. | prefix))`.
  The baseline is over the entire vocabulary at that prefix. It is not a
  mean over the trajectory or the selected token positions.
- Vine MC: sample K independent student continuations from each distinct
  nonterminal prefix; estimate V by binary final-answer correctness.
  For an interior interval [a,b), `A = V(b) - V(a)`. For a terminal interval,
  `A = observed_original_reward - V(a)`, with terminal V=0.

These OPD coefficients expose the sampled-action reverse-KL signal; they do not
reproduce GOLD's complete full-vocabulary training gradient. Vine estimates
expected correctness under this student and completion budget, not logical
validity or optimal future behavior.

Original rollouts and MC continuations use temperature=1, top_p=1, no top-k
restriction, and no repetition penalty. There is no KL reward penalty,
discounting, or advantage whitening. A prefix of t completion tokens receives
only B-t additional tokens. Grading sees the complete original prefix plus the
new continuation. EOS and the total budget are terminal. MC never uses PI.

The MC definition follows [VinePPO](https://arxiv.org/html/2410.01679v2).
Our blank-line segmentation is an explicit experiment choice rather than an
exact reproduction of the paper's MATH segmentation heuristic.

## Token selection and matching

Both selection modes run by default:

- `uniform`: `--num-token-samples` positions per rollout, uniformly without
  replacement. Short rollouts use all positions. Each selected token branches
  immediately before and after that exact token.
- `steps`: split at `\n\n`, merge short spans, and hard-split long spans using
  `--min-segment-tokens` and `--max-segment-tokens`. Lengths are in tokens.
  Only the final remainder may be shorter than the minimum. Delimiters stay in
  the preceding segment; a delimiter inside a token snaps to that token's end.

All tokens are preserved. Every record carries its canonical rollout ID,
zero-based completion token indices, and exact token IDs. OPD and every OPSD
condition score these same IDs. Token-to-ID vocabularies and special IDs must
match across models. Each model uses its own chat template; only OPSD receives
PI. Stored student prefixes are passed to vLLM as token IDs without rerendering
or retokenizing the conversation.

A step's Vine advantage is explicitly labeled `segment_broadcast`: that value
applies to each token in the segment, while distillation values remain individual
token coefficients. Summaries include both the token-weighted broadcast view
and a separate segment view comparing mean distillation coefficient with the
segment's Vine advantage. Uniform-token summaries remain separate.

## Phases and resuming

`--phase all` runs prepare, generate, plan, mc, score, and aggregate in separate
spawned processes. In particular, HF scoring never shares vLLM's process-global
Torch state. Each phase can also be requested individually.

Preparation, original generation, per-prefix MC draws, and per-rollout teacher
scores are cached. The manifest rejects changed experiment settings; local
checkpoint file sizes/mtimes are recorded, and remote model commits are pinned
after preparation. Use a new output directory for a different student, cohort,
selection seed, token count, segmentation, PI setup, or completion budget.

Increase `--mc-samples` (alias `--K`) and rerun the same command to append MC
draws without regenerating original rollouts or rescoring. Sampling seeds are
derived separately from the prefix ID and draw index. Every shared prefix uses
one cached value estimate across uniform/step views and across rollouts.
`--mc-batch-size` and `--bootstrap-samples` may also change on resume.
A smaller K aggregates the first K cached draws. Generation is stochastic;
per-request seeds do not promise bitwise equality across hardware or engine
versions.

To inspect branching cost before starting MC, run prepare, generate, and plan
individually. The plan prints distinct-prefix count and total K continuations.
The original rollout generation cache is committed as a whole; interrupted MC
and scoring resume from completed prefix draws and rollout score files.

## Artifacts

- `manifest.json`: experiment configuration and code commit.
- `cohort.json`: selected questions, PI, exact prompt IDs, tokenizer identities,
  source fingerprint, and remote model revisions.
- `rollouts.json`: fixed student token IDs, decoded completions, rewards and IDs.
- `plan.json`: selected tokens, segments and shared nonterminal prefixes.
- `mc/<prefix-id>.json`: draw seeds, rewards, lengths, finish reasons and total
  success/sample counts. `--save-mc-completions` additionally retains continuation
  IDs; the complete text can be reconstructed from prefix plus continuation.
- `scores/<condition>/<rollout-id>.json`: aligned student/teacher logps, raw
  coefficients, exact reverse KL, and centered coefficients for every token.
- `k-<K>/comparisons.jsonl`: one record per selected token or segment, including
  the shared MC prefix keys/counts, Vine value, approximate 95% interval, and
  all paired distillation coefficients.
- `k-<K>/summary.json`: original-rollout pass@1, Pearson/Spearman correlations,
  sign agreement, agreement on MC effects whose interval excludes zero, and
  question-cluster bootstrap intervals.

MC difference intervals subtract Bonferroni-adjusted Wilson intervals. They
remain nonzero for all-success/all-failure samples and are approximate intervals,
not simultaneous guarantees across all selected positions. Bootstrap summaries
condition on the cached MC estimates. Shared prefixes and broadcast segment
values create dependence; token counts are not independent MC sample counts.
Increasing K improves the estimate but does not turn it into ground truth.

## Validation

```sh
.venv/bin/python -m unittest tests.test_advantage_comparison tests.test_model_scoring -v
```

The tests run on CPU, including causal alignment against a full tiny-Qwen
forward pass, FP32 normalization, exact centering, token/segment matching,
terminal handling, remaining budgets, and MC cache extension.
