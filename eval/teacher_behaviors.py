r"""
Count the four COGNITIVE BEHAVIORS of Gandhi et al. (arXiv 2503.01307, "Cognitive Behaviors
that Enable Self-Improving Reasoners") in the self-teacher completions that
eval/teacher_uncertainty.py already wrote to disk, per privileged context (PI).

WHY. teacher_uncertainty.py measures E(y), a 10-token epistemic-marker regex. That proxy
covers roughly two of the four behaviors -- `wait/hmm/actually/alternatively` reads as
backtracking, `check/seems/likely/might` as verification -- and NOTHING for subgoal setting
or backward chaining. Gandhi et al. show it is precisely backtracking and verification that
gate whether RL self-improvement works, that priming them in is causal, and (their strongest
control) that reasoning-behavior content matters more than whether the primed solution was
even correct. That makes the behavior counts the natural second instrument on the PI ladder,
where the 16k readings already split BIMODALLY rather than monotonically:

    PI        tokens   E/1k    pass@1   unclosed    step-0 advantage
    none        7684   10.9      78.8         6%            0
    answer      6461   11.0      95.2         4%       -0.015
    hint        6866   10.3      81.0         5%       -0.015
    rollout     1063    2.7      81.6        93%       -0.117
    full         888    1.7      99.8        98%       -0.110

`answer`/`hint` hold the marker RATE at `none`'s level and only shorten; `full`/`rollout`
drop the rate 4-6x and mostly never emit `</think>` at all. This script asks whether that
rate collapse is specifically a loss of verification and backtracking.

THIS IS A RE-PARSE, NOT A RE-GENERATION. It reads `completions_{pi}.jsonl` and never samples
from a teacher. `question_idx`/`sample_idx` are carried through unchanged, so every row joins
back to the source record -- and, via `char_start`/`char_end`, forward to the per-token
`advantages` arrays under results/advantage_dynamics/*/step-*/scores.

CHUNKED, NOT WHOLE-TRACE. The `none` arm averages ~83 epistemic markers over ~7.7k tokens;
asking a classifier to count ~30 distinct backtracking events in one pass produces an error
that is itself length-correlated -- exactly the confound the per-1k normalization exists to
remove. Segments are packed to `--chunk-tokens`, ALIGNED TO PARAGRAPH BREAKS (the long arms
carry 100+ of them, so packing is clean). The rubric is a constant prefix across every call,
so `enable_prefix_caching` makes the per-call overhead close to the segment itself.

THE GRAMMAR IS WHAT DISABLES THINKING. The Qwen3.x-27B judges reason by default.
`enable_thinking=False` is passed via chat_template_kwargs, but the real guarantee is the JSON schema: structured
decoding constrains the FIRST generated token, so a `<think>` block is unrepresentable. The
defensive strip in `_parse_response` exists only for the case where that is ever relaxed --
utils/gen_hints.py documents what a thinking model does when it ignores the kwarg.

WHAT IS REPORTED, and why all three:
  rate_per_1k          the headline. The only figure that separates "shorter" from
                       "behaviourally different" -- see the E/1k column above.
  mean_per_trajectory  Gandhi's own unit, for comparability with the paper.
  prevalence           fraction of trajectories with >=1 occurrence. Robust to counting
                       error, and closest to the paper's load-bearing binary framing.
Confidence intervals resample QUESTIONS, not trajectories: with 8 samples over 128 problems
the effective n is ~128. Reuses advantage_dynamics.question_cluster_bootstrap_ci.

Analysis is over the WHOLE completion. The <think>/post split is degenerate for full/rollout
(93-98% never close the block), so it is reported as a secondary, `closed`-conditioned cut.

# smoke: 8 trajectories per arm, placeholder rubric, no GPU cost estimate needed
CUDA_VISIBLE_DEVICES=0 uv run python -m eval.teacher_behaviors \
    --teacher-model Qwen/Qwen3-1.7B --limit 8 --allow-placeholder-examples

# plan the sweep without loading the engine
uv run python -m eval.teacher_behaviors --teacher-model Qwen/Qwen3-1.7B --dry-run

# the sweep
CUDA_VISIBLE_DEVICES=0 uv run python -m eval.teacher_behaviors \
    --teacher-model Qwen/Qwen3-1.7B --samples-per-problem 4
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from eval.advantage_dynamics import (
    fingerprint_ids,
    question_cluster_bootstrap_ci,
    read_json,
    write_json_atomic,
)
from eval.teacher_uncertainty import EPISTEMIC_MARKERS, count_epistemic, split_think


PI_MODES = ("none", "rollout", "answer", "hint", "full")

BEHAVIORS = ("verification", "backtracking", "subgoal_setting", "backward_chaining")

# Bump on any change to BEHAVIOR_DEFINITIONS, COUNTING_RULES or the example set that alters
# what a count MEANS -- the same role TEACHER_VERSION plays for the self-teacher's weights and
# VALUE_PROMPT_VERSION for the critic's state. Ordinary knobs (chunk size, temperature) are
# recorded individually and do not need a bump. `rubric_fingerprint` in the run meta catches
# an edit that forgot to bump this one; the version string is what makes the bump intentional.
BEHAVIOR_RUBRIC_VERSION = "gandhi_four_behaviors_v2"


# ---------------------------------------------------------------------------
# The rubric
#
# Definitions follow Gandhi et al. section 3, re-worded for step-by-step mathematics rather
# than Countdown. The re-wording is not cosmetic: Countdown supplies a literal target number,
# which makes backward chaining both frequent and unambiguous there. In DeepMath it is neither,
# and the paper's own interrater reliability is weakest on exactly that behavior (ICC3 0.55
# against humans, vs up to 0.88 elsewhere). Treat backward_chaining as the noisy column.
# ---------------------------------------------------------------------------

BEHAVIOR_DEFINITIONS = {
    "verification": (
        "Checking a result against a criterion. Any of these count: substituting a value back "
        "into an equation, testing a candidate and comparing the outcome against what was "
        "predicted or required, checking a boundary or special case, confirming a claim by "
        "unfolding a definition, sanity-checking a magnitude or sign. It counts whether the "
        "check confirms or refutes. Simply recomputing a quantity, with nothing tested against "
        "anything, is not verification."
    ),
    "backtracking": (
        "Abandoning an approach already taken and replacing it. The segment drops a line of "
        "work and moves to a different one -- because it failed, or because it got unwieldy. "
        "An option that is floated and never actually pursued has not been abandoned, and "
        "correcting a slip while continuing down the same path is not a replacement."
    ),
    "subgoal_setting": (
        "Naming an intermediate target on the way to the final answer, thereby decomposing the "
        "problem. Both forms count: planning a sequence ('first find the discriminant, "
        "then...'), and splitting a problem or expression into parts to be handled separately "
        "('break this into two cases', 'this reduces to computing X'). An ordinary next "
        "computation ('let me compute 3 * 7') is not a subgoal unless it is named as an "
        "objective to reach."
    ),
    "backward_chaining": (
        "Reasoning from the desired outcome back toward what is given: starting from the goal, "
        "or from a property the answer must have, and deriving what would then have to be "
        "true. All of these count: 'for the sum to be an integer, the denominator must "
        "divide...'; 'we want U*AU to be zero-diagonal, so the diagonal entries must "
        "vanish, hence b = c = 0'; 'x^2 + x + 1 is always positive, so m must be positive'. "
        "The test is the DIRECTION of inference -- from a requirement to a constraint. Forward "
        "derivation that merely mentions the goal is not."
    ),
}

COUNTING_RULES = (
    "Count DISTINCT occurrences, not sentences: one continuous act of checking is one "
    "verification even if it spans several lines.\n"
    "Count only what occurs inside the CURRENT segment. Text shown as preceding context is "
    "for disambiguation only and must never be counted.\n"
    "A segment may contain several behaviors at once. Score each behavior "
    "independently.\n"
    "Judge the reasoning as written. Do not evaluate whether the mathematics is correct."
)


@dataclass(frozen=True)
class RubricExample:
    """One few-shot segment and its gold counts.

    `note` is rendered to the classifier as the reason for the labelling; it is where the
    boundary cases in BEHAVIOR_DEFINITIONS are made concrete, and it is doing most of the work
    in a few-shot rubric. Keep the segments SHORT (a few hundred characters) -- they sit in the
    cached prefix of every one of ~47k calls.
    """

    segment: str
    counts: dict[str, int]
    note: str


# The sentinel that marks an unfilled example. `examples_are_filled` refuses to run a sweep
# while any survive: a 47M-token pass under a placeholder rubric is an expensive
# mistake, and every arm would come back looking the same.
EXAMPLE_PLACEHOLDER = "<<TODO: FILL FROM A LABELLED DeepMath TRACE>>"


# Every segment below is a VERBATIM excerpt from a DeepMath-103K r1_solution_3 trace, with the
# source row recorded above it. R1 traces are the closest thing on hand to a `none`-PI
# self-teacher rollout, which is the arm where all four behaviors actually occur -- a rubric
# built from `full`-arm excerpts would have almost no positive instances to show.
#
# Kept grouped BY BEHAVIOR for editing. The order the classifier sees is EXAMPLE_ORDER below,
# which is a separate decision for a separate reason.
_BY_ID: dict[str, RubricExample] = {
    # DeepMath-103K row 95889, r1_solution_3 ("greatest common divisor of abcd + dcba").
    "vf_counterexample": RubricExample(
        segment=(
            "Wait, interesting. So regardless of whether the digits are increasing or "
            "decreasing, abcd + dcba is 5555? Wait, let's check another example. Take 2345, "
            "its reverse is 5432. Sum is 2345 + 5432 = 7777. Wait, that's not 5555."
        ),
        counts={
            "verification": 1,
            "backtracking": 0,
            "subgoal_setting": 0,
            "backward_chaining": 0,
        },
        note=(
            "A prediction is floated, a concrete case is built to test it, and the computed "
            "value is compared against the prediction. The comparison against a criterion is "
            "what makes this verification -- refuting and confirming both count. It is ONE "
            "occurrence, not three, because it is a single continuous act of checking. It is "
            "NOT backtracking: a floated conjecture is discarded, but the approach (compute "
            "sums for examples and look for structure) is unchanged."
        ),
    ),
    # DeepMath-103K row 65233, r1_solution_3 (rewrite a summation via its EGF).
    "vf_expand_identity": RubricExample(
        segment=(
            "Let me check if that makes sense. Let's verify by expanding both sides.\n\n"
            "Original summation when n=2: term is 2*1*a_0*z^2/2! = 2*1*a0*z^2/2 = a0 z^2. If "
            "we compute z^2 Ahat(z), the coefficient of z^2 in Ahat(z) is a0/0! + a1 z/1! + "
            "a2 z^2/2! + ... So multiplying by z^2 gives a0 z^2 + a1 z^3/1! + a2 z^4/2! + "
            "..., which matches the original summation when we shift indices. Wait, but when "
            "n=2, the term in the original sum is a0 z^2, which corresponds to k=0 in the "
            "shifted sum: a0 z^{2}/0! = a0 z^2. Similarly, when n=3, original term is 3*2*a1 "
            "z^3/3! = 6 a1 z^3 /6 = a1 z^3. In the shifted sum, when k=1: a1 z^{3}/1! = a1 "
            "z^3. So that matches."
        ),
        counts={
            "verification": 1,
            "backtracking": 0,
            "subgoal_setting": 0,
            "backward_chaining": 0,
        },
        note=(
            "A derived identity is checked by expanding both sides and comparing coefficients. "
            "ONE occurrence, not two or three: checking n=2 and then n=3 is a single continuous "
            "act of verification, not one per case. 'Let's verify by expanding both sides' is "
            "the METHOD of the check and not a separate subgoal -- announcing how you will "
            "verify is part of verifying. The mid-check 'Wait, but when n=2...' is a "
            "clarification inside the same check, not an abandoned approach."
        ),
    ),
    # DeepMath-103K row 38512, r1_solution_3 (sum of squares of order-m minors).
    "vf_definitional": RubricExample(
        segment=(
            "If the rows are orthonormal, then the matrix A multiplied by its transpose should "
            "be the identity matrix. Let me verify that. If A is m x n with orthonormal rows, "
            "then A * A^T is indeed the m x m identity matrix. Because each entry (i,j) of "
            "A * A^T is the dot product of row i and row j of A, which is 1 if i = j and 0 "
            "otherwise. So, A * A^T = I_m."
        ),
        counts={
            "verification": 1,
            "backtracking": 0,
            "subgoal_setting": 0,
            "backward_chaining": 0,
        },
        note=(
            "Verification does not have to be arithmetic. A claim is asserted, then tested by "
            "unfolding the relevant definition until it either holds or fails -- here the "
            "entries of A * A^T are computed symbolically and matched against the identity "
            "matrix. Counting only numeric checks would miss most verification in algebra and "
            "analysis."
        ),
    ),
    # DeepMath-103K row 73729, r1_solution_3 (degree-11 polynomial agreeing with 1/(x+1)).
    "bt_invalid_setup": RubricExample(
        segment=(
            "Wait, but f(x) is a polynomial, and 1/(x+1) is a rational function. So their "
            "difference isn't a polynomial. So maybe that approach is invalid. Hmm, scratch "
            "that.\n\n"
            "Alternatively, consider that since f(x) is a polynomial that agrees with 1/(x+1) "
            "at x=0,...,11, perhaps we can use the concept of polynomial interpolation through "
            "those points. Since f(x) is the unique polynomial of degree 11 passing through "
            "all these points, then to find f(12), we can use some interpolation formula."
        ),
        counts={
            "verification": 1,
            "backtracking": 1,
            "subgoal_setting": 0,
            "backward_chaining": 0,
        },
        note=(
            "TWO behaviors in one segment, and both must be counted. Testing the stated "
            "intermediate result (that f(x) - 1/(x+1) is a polynomial) against the definition "
            "and finding it fails is VERIFICATION; dropping that approach and taking up "
            "interpolation instead is BACKTRACKING. Real backtracking is usually preceded by "
            "the check that condemned the approach -- do not stop at one label."
        ),
    ),
    # DeepMath-103K row 10652, r1_solution_3 (inversion-parity cancellation in S_n).
    "bt_too_convoluted": RubricExample(
        segment=(
            "Wait, maybe this is getting too convoluted. Let's try a different approach.\n\n"
            "Alternatively, note that for each position i, the sum T_i = ∑_{σ ∈ S_n} "
            "(-1)^{ℓ(σ)} σ(i) must be the same for all i due to symmetry."
        ),
        counts={
            "verification": 0,
            "backtracking": 1,
            "subgoal_setting": 0,
            "backward_chaining": 0,
        },
        note=(
            "No error is found here. An approach is dropped because it has become unwieldy, "
            "and a different one is taken up. Backtracking does not require a detected "
            "mistake -- being bogged down is enough, provided the approach is actually "
            "abandoned and replaced. Nothing is checked against a criterion, so verification "
            "is 0."
        ),
    ),
    # DeepMath-103K row 73729, r1_solution_3 -- a later passage in the same trace.
    "bt_floated_vs_pursued": RubricExample(
        segment=(
            "Alternatively, maybe there's a clever trick using the fact that 1/(x+1) is "
            "related to the harmonic series or something else. Wait, but how?\n\n"
            "Alternatively, consider constructing the polynomial explicitly. Let me think. "
            "Let’s denote that for each x in 0,1,...,11, f(x) = 1/(x+1). So, "
            "f(x) - 1/(x+1) = 0 for x=0,1,...,11. But since f(x) is a polynomial of degree 11 "
            "and 1/(x+1) is not a polynomial, their difference is not a polynomial. Therefore, "
            "this doesn't help."
        ),
        counts={
            "verification": 1,
            "backtracking": 1,
            "subgoal_setting": 0,
            "backward_chaining": 0,
        },
        note=(
            "ONE backtracking, not two. The harmonic-series idea is floated and dropped in the "
            "same breath without ever being pursued -- considering an option and not "
            "committing to it is not backtracking. The explicit-construction approach IS taken "
            "up, worked on, and then abandoned ('this doesn't help'), so that one counts. The "
            "check that kills it is again verification."
        ),
    ),
    # DeepMath-103K row 26724, r1_solution_3 (contour integral via the residue theorem).
    "sg_two_acts": RubricExample(
        segment=(
            "So, in this case, I need to find all the singularities of the integrand inside "
            "the circle |z| = π/4 and compute their residues, then sum them up and multiply "
            "by 2πi.\n\n"
            "The integrand is 1 + z + 1/tan z. Let me break this down into three separate "
            "terms: 1, z, and 1/tan z."
        ),
        counts={
            "verification": 0,
            "backtracking": 0,
            "subgoal_setting": 2,
            "backward_chaining": 0,
        },
        note=(
            "TWO occurrences, and the arithmetic of that is the point. The first sentence "
            "names a four-step route (find singularities, compute residues, sum, multiply) but "
            "is ONE continuous act of planning, not four -- the same rule that makes a "
            "multi-part check one verification. Splitting the integrand into three terms is a "
            "SECOND, separate act of decomposition, so the segment scores 2."
        ),
    ),
    # DeepMath-103K row 22157, r1_solution_3 (radius of convergence of sum P(n) x^n).
    "sg_split_expression": RubricExample(
        segment=(
            "Let's break this into two parts. First, |c₀|^(1/n): since |c₀| is a constant, the "
            "limit of |c₀|^(1/n) as n→∞ is 1, because any positive number raised to the power "
            "of 1/n approaches 1. \n\nNext, n^(N/n)."
        ),
        counts={
            "verification": 0,
            "backtracking": 0,
            "subgoal_setting": 1,
            "backward_chaining": 0,
        },
        note=(
            "Decomposing an expression into sub-problems that will be solved separately is "
            "subgoal setting, just as much as planning a sequence of steps. The justification "
            "that follows ('because any positive number raised to the power of 1/n approaches "
            "1') is a derivation, not a check against a criterion, so verification is 0."
        ),
    ),
    # DeepMath-103K row 102406, r1_solution_3 (arithmetic mean of five expressions).
    "sg_plain_plan": RubricExample(
        segment=(
            "So, maybe I can set up an equation where the sum of these five terms divided by 5 "
            "equals 24, and then solve for x. Let me write that out."
        ),
        counts={
            "verification": 0,
            "backtracking": 0,
            "subgoal_setting": 1,
            "backward_chaining": 0,
        },
        note=(
            "A plain two-step plan: build the equation, then solve it. What makes it subgoal "
            "setting is that the equation is named as an intermediate target distinct from the "
            "final one (x). Two things that are NOT subgoal setting and look similar: an "
            "ordinary next computation ('now add these five terms'), and an orientation "
            "opener such as 'first, I need to make sure I understand all the components', "
            "which names no mathematical target at all."
        ),
    ),
    # DeepMath-103K row 92351, r1_solution_3 (Z(n): least m with 1+...+m divisible by n).
    "bc_divisibility": RubricExample(
        segment=(
            "For this sum to be divisible by p, we need m(m + 1)/2 ≡ 0 mod p. Since p is an "
            "odd prime, 2 has an inverse modulo p. So, equivalently, m(m + 1) ≡ 0 mod p. This "
            "means that either m ≡ 0 mod p or m + 1 ≡ 0 mod p. Therefore, the minimal m should "
            "be the smallest positive integer such that m ≡ 0 mod p or m ≡ -1 mod p."
        ),
        counts={
            "verification": 0,
            "backtracking": 0,
            "subgoal_setting": 0,
            "backward_chaining": 1,
        },
        note=(
            "The plain form. The desired property (the sum is divisible by p) is the starting "
            "point, and the reasoning runs backwards from it to a condition m must satisfy. "
            "The several rewriting steps are ONE continuous act of backward reasoning, not "
            "one per line. Nothing is checked against a criterion here, so verification is 0."
        ),
    ),
    # DeepMath-103K row 29715, r1_solution_3 (values of y giving three distinct real roots).
    "bc_three_roots": RubricExample(
        segment=(
            "So, maybe I need to find the critical points of the function and then ensure that "
            "between these critical points, the function changes sign. Wait, actually, for "
            "there to be three real roots, the function must have a local maximum and a local "
            "minimum, and the y-values at these critical points must straddle the x-axis. In "
            "other words, the local maximum must be positive and the local minimum must be "
            "negative."
        ),
        counts={
            "verification": 0,
            "backtracking": 0,
            "subgoal_setting": 1,
            "backward_chaining": 1,
        },
        note=(
            "Two behaviors again. Naming critical points as the thing to find first is SUBGOAL "
            "SETTING; deriving, from the goal of three real roots, what the function must "
            "therefore look like is BACKWARD CHAINING. 'Wait, actually' here refines the "
            "previous sentence rather than discarding an approach, so it is NOT backtracking "
            "-- the trace goes on to do exactly what it just proposed."
        ),
    ),
    # DeepMath-103K row 37710, r1_solution_3 (find b/a for a PDE with u(1,2) = u(3,6)).
    "bc_pde_then_forward": RubricExample(
        segment=(
            "Since u(1, 2) = u(3, 6), it must be that f(b - 2a) = f(3b - 6a). For this to "
            "hold, the arguments of f must be the same (assuming f is not a constant function, "
            "which would trivially satisfy the PDE but not necessarily the given condition). "
            "Therefore, we require that:\n\nb - 2a = 3b - 6a.\n\nLet me solve this equation "
            "for b/a. Let's rearrange the terms:\n\nb - 2a = 3b - 6a.\n\nSubtract b from both "
            "sides:\n\n-2a = 2b - 6a.\n\nAdd 6a to both sides:\n\n4a = 2b."
        ),
        counts={
            "verification": 0,
            "backtracking": 0,
            "subgoal_setting": 0,
            "backward_chaining": 1,
        },
        note=(
            "ONE occurrence, located in the first half. Backward chaining is the move from the "
            "required condition to what must therefore be true of the parameters. Everything "
            "after 'Let me solve this equation' is ORDINARY FORWARD ALGEBRA driving toward the "
            "same goal, and adds nothing to the count -- what matters is the direction of "
            "inference, not whether the goal is in view. 'Let me solve this equation for b/a' "
            "is also not subgoal setting: solving it IS the final step, not an intermediate "
            "target on the way to one."
        ),
    ),
    # DeepMath-103K row 102406, r1_solution_3 -- the passage that EXECUTES the plan shown two
    # examples above. Same problem and same voice, scoring zero: the pairing isolates what the
    # label is actually responding to.
    "neg_busy_arithmetic": RubricExample(
        segment=(
            "So for the x terms: x + 2x + 2x. Let me compute that. x is 1x, so 1x + 2x + 2x. "
            "That's 1 + 2 + 2 = 5x. Okay, so the x terms add up to 5x.\n\n"
            "Now the constant terms: 8 + 15 + 13 + 4. Let's add those step by step. 8 + 15 is "
            "23. Then 23 + 13. Hmm, 23 + 10 is 33, and then +3 is 36. Then 36 + 4 is 40. So "
            "the constants add up to 40."
        ),
        counts={
            "verification": 0,
            "backtracking": 0,
            "subgoal_setting": 0,
            "backward_chaining": 0,
        },
        note=(
            "THE NEGATIVE EXAMPLE -- all four are zero, and zero is a common, correct answer. "
            "This segment is busy but empty: 'Let me compute that' and 'Let's add those step "
            "by step' are ordinary next computations, not named intermediate targets; the "
            "regrouping of 23 + 13 into 23 + 10 + 3 is a computation, not a check against any "
            "criterion; nothing is abandoned; nothing is derived from the goal. Note the "
            "hedging word 'Hmm' -- hedges, 'wait', and 'let me' are NOT what you are counting. "
            "Score the reasoning move, never the vocabulary."
        ),
    ),
    # DeepMath-103K row 95889, r1_solution_3 -- the same trace as vf_counterexample.
    "bc_digit_constraint": RubricExample(
        segment=(
            "But then, if a is 7, then d would be 10, which is not a digit. Wait, hold on. If "
            "a is a digit (0-9), but since abcd is a four-digit number, a must be from 1-9, "
            "and d must also be from 1-9 (since dcba is a four-digit number). So if a, b, c, d "
            "are consecutive digits in increasing order, then a can be at most 6, because a+3 "
            "must be ≤9. So a can be 1-6."
        ),
        counts={
            "verification": 1,
            "backtracking": 0,
            "subgoal_setting": 0,
            "backward_chaining": 1,
        },
        note=(
            "Backward chaining does not need formal 'for X to hold, Y must' phrasing. Here a "
            "requirement on the OUTPUT (d has to be a digit) is propagated back to a "
            "constraint on the INPUT (a is at most 6) -- that is the same move, written "
            "casually. Testing a=7 and finding it gives a non-digit is also a verification, so "
            "both are counted."
        ),
    ),
}


# The order the classifier sees, which is a different question from how the examples are
# grouped above. Three properties are deliberate:
#
#   * NO TWO ADJACENT examples share a dominant behavior. Grouped by behavior, the sequence
#     itself becomes a pattern an in-context learner can read instead of reading the content.
#   * The first five are the crisp single-behavior cases, one per behavior plus a negative, so
#     the simplest reading of each label is established before any co-occurrence or counting
#     subtlety appears.
#   * The three NEGATIVES are spread, and none is last. Twelve of fifteen examples carry at
#     least one behavior, which already suggests a base rate of "something is here"; clustering
#     the zeros, or burying them where recency discounts them, would push that further. The
#     segments most likely to be scored wrong by an over-firing classifier are the short terse
#     ones -- which is exactly what the `full` and `rollout` arms are made of, so the error
#     would land on the arms whose low rates the study is trying to measure.
EXAMPLE_ORDER: tuple[str, ...] = (
    "vf_counterexample",       # verification, crisp
    "sg_plain_plan",           # subgoal setting, crisp
    "bc_divisibility",         # backward chaining, crisp
    "bt_too_convoluted",       # backtracking, crisp
    "neg_busy_arithmetic",     # the one all-zero example
    "vf_definitional",         # verification need not be numeric
    "bc_digit_constraint",     # backward chaining written casually, not formally
    "sg_two_acts",             # counting: one act naming four steps, plus a second act
    "bt_invalid_setup",        # co-occurrence: backtracking with the check that caused it
    "vf_expand_identity",      # counting: a multi-part check is still one
    "bc_pde_then_forward",     # backward then forward, in one segment, scoring one
    "sg_split_expression",     # decomposing an expression, not a plan
    "bc_three_roots",          # co-occurrence: backward chaining with subgoal setting
    "bt_floated_vs_pursued",   # counting: a floated-but-unpursued idea is not backtracking
)

EXAMPLES: tuple[RubricExample, ...] = tuple(_BY_ID[key] for key in EXAMPLE_ORDER)


def examples_are_filled() -> bool:
    """False while any few-shot segment is still the placeholder sentinel."""
    return all(example.segment != EXAMPLE_PLACEHOLDER for example in EXAMPLES)


def render_system_prompt() -> str:
    """The constant classifier prefix: definitions, counting rules, worked examples.

    Identical for every call, which is what `enable_prefix_caching` needs to make the rubric
    free after the first segment.
    """
    parts = [
        "You are to label excerpts of mathematical reasoning for four cognitive behaviors. For the "
        "segment given, report how many times each behavior occurs.",
        "",
        "BEHAVIORS",
    ]
    for name in BEHAVIORS:
        parts.append(f"- {name}: {BEHAVIOR_DEFINITIONS[name]}")
    parts += ["", "RULES", COUNTING_RULES, "", "EXAMPLES"]
    for index, example in enumerate(EXAMPLES, start=1):
        counts = json.dumps({name: example.counts[name] for name in BEHAVIORS})
        parts += [
            f"Example {index}",
            "<segment>",
            example.segment,
            "</segment>",
            f"-> {counts}",
            f"({example.note})",
            "",
        ]
    parts.append("Answer with a JSON object holding one integer per behavior.")
    return "\n".join(parts)


def rubric_fingerprint() -> str:
    """Hash of everything that defines what a count means.

    Recorded in the run meta so an edited rubric cannot silently share a
    BEHAVIOR_RUBRIC_VERSION with the numbers it invalidates.
    """
    payload = json.dumps(
        {
            "version": BEHAVIOR_RUBRIC_VERSION,
            "definitions": BEHAVIOR_DEFINITIONS,
            "rules": COUNTING_RULES,
            "examples": [
                {"segment": e.segment, "counts": e.counts, "note": e.note} for e in EXAMPLES
            ],
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def response_schema(with_evidence: bool) -> dict:
    """JSON schema handed to vLLM's structured-output backend.

    Flat rather than nested per behavior: a shallower grammar decodes faster and leaves less
    room for the model to emit a well-formed object with the wrong shape. Evidence quotes are
    capped at 3 -- they exist to be audited and to give `quote_match_rate`, not to be complete.
    """
    properties: dict[str, dict] = {
        name: {"type": "integer", "minimum": 0} for name in BEHAVIORS
    }
    if with_evidence:
        properties |= {
            f"{name}_evidence": {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": 3,
            }
            for name in BEHAVIORS
        }
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

_PARAGRAPH_BREAK = re.compile(r"\n{2,}")


def paragraph_spans(text: str) -> list[tuple[int, int]]:
    """Character spans of the paragraphs in `text`, excluding the blank-line separators.

    Offsets are into the ORIGINAL string, so a chunk built from these spans can be mapped back
    to the source completion (and from there to token positions under whichever tokenizer the
    caller cares about -- the classifier's vocabulary is not the teacher's).
    """
    spans, cursor = [], 0
    for match in _PARAGRAPH_BREAK.finditer(text):
        if match.start() > cursor:
            spans.append((cursor, match.start()))
        cursor = match.end()
    if cursor < len(text):
        spans.append((cursor, len(text)))
    return spans or [(0, len(text))]


def _split_oversized(
    text: str, start: int, end: int, tokenizer, target_tokens: int
) -> list[tuple[int, int]]:
    """Hard-split one paragraph that alone exceeds the budget, on token boundaries.

    Rare (a single unbroken derivation), but it must not silently produce a segment longer
    than the context. A fast tokenizer gives exact character offsets; without one we fall back
    to a proportional character split, which can land mid-token and is only ever approximate.
    """
    body = text[start:end]
    if not getattr(tokenizer, "is_fast", False):
        n_pieces = math.ceil(len(body) / max(target_tokens * 3, 1))
        width = math.ceil(len(body) / n_pieces)
        return [
            (start + offset, min(start + offset + width, end))
            for offset in range(0, len(body), width)
        ]
    encoded = tokenizer(body, add_special_tokens=False, return_offsets_mapping=True)
    offsets = encoded["offset_mapping"]
    pieces = []
    for first in range(0, len(offsets), target_tokens):
        window = offsets[first : first + target_tokens]
        if window:
            pieces.append((start + window[0][0], start + window[-1][1]))
    return pieces or [(start, end)]


def chunk_completion(text: str, tokenizer, target_tokens: int) -> list[dict]:
    """Pack `text` into paragraph-aligned segments of at most ~`target_tokens`.

    Greedy: paragraphs accumulate until the next one would overflow. A paragraph is never
    split across two segments unless it exceeds the budget on its own, because a behavior
    straddling a boundary is either double-counted or missed, and paragraph breaks are where
    reasoning steps already end. The long PI arms carry 100+ breaks per trace, so the packing
    is tight; `full`/`rollout` yield one or two segments, which is the honest representation
    of how little reasoning they contain.
    """
    if target_tokens < 1:
        raise ValueError("target_tokens must be >= 1")
    spans = paragraph_spans(text)
    bodies = [text[start:end] for start, end in spans]
    lengths = [
        len(ids) for ids in tokenizer(bodies, add_special_tokens=False)["input_ids"]
    ]

    chunks: list[dict] = []
    current: list[tuple[int, int]] = []
    current_tokens = 0

    def flush() -> None:
        nonlocal current, current_tokens
        if not current:
            return
        start, end = current[0][0], current[-1][1]
        chunks.append(
            {
                "chunk_idx": len(chunks),
                "char_start": start,
                "char_end": end,
                "n_classifier_tokens": current_tokens,
                "text": text[start:end],
            }
        )
        current, current_tokens = [], 0

    for (start, end), n_tokens in zip(spans, lengths, strict=True):
        if n_tokens > target_tokens:
            flush()
            for piece_start, piece_end in _split_oversized(
                text, start, end, tokenizer, target_tokens
            ):
                piece = text[piece_start:piece_end]
                chunks.append(
                    {
                        "chunk_idx": len(chunks),
                        "char_start": piece_start,
                        "char_end": piece_end,
                        "n_classifier_tokens": len(
                            tokenizer(piece, add_special_tokens=False)["input_ids"]
                        ),
                        "text": piece,
                    }
                )
            continue
        if current and current_tokens + n_tokens > target_tokens:
            flush()
        current.append((start, end))
        current_tokens += n_tokens
    flush()
    return chunks


# ---------------------------------------------------------------------------
# Source completions
# ---------------------------------------------------------------------------


def completions_path(root: str, teacher_model: str, pi_mode: str) -> str:
    """Where teacher_uncertainty.py wrote this arm (same slug convention: '/' -> '_')."""
    return os.path.join(
        root, teacher_model.replace("/", "_"), f"completions_{pi_mode}.jsonl"
    )


def load_completions(
    path: str, samples_per_problem: int | None, limit: int | None
) -> list[dict]:
    """Read one arm's completions, optionally thinned.

    Thinning drops SAMPLES (`sample_idx < k`), never problems: `problem_idx`/`question_idx` are
    shared across PI arms and are what makes the comparison paired, whereas the 8 samples of a
    problem are independent draws whose marginal value falls off fast once the confidence
    interval already resamples questions.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"No completions at {path}. Generate them first with\n"
            "  python -m eval.teacher_uncertainty --teacher-model <model> --pi-modes ..."
        )
    rows = []
    with open(path) as handle:
        for line in handle:
            record = json.loads(line)
            if samples_per_problem is not None and int(record["sample_idx"]) >= samples_per_problem:
                continue
            rows.append(record)
    if limit is not None:
        rows = rows[:limit]
    return rows


def trajectory_key(record: dict) -> str:
    return f"{int(record['question_idx'])}:{int(record['sample_idx'])}"


def source_fingerprint(rows: list[dict]) -> str:
    """Identity of the exact set of trajectories classified, in order."""
    return fingerprint_ids(trajectory_key(row) for row in rows)


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

_THINK_CLOSE = "</think>"


def build_messages(system_prompt: str, chunk_text_: str, context: str | None) -> list[dict]:
    """One classification conversation. The system turn is byte-identical everywhere, so it is
    the cached prefix; only the user turn varies."""
    user = ""
    if context:
        user += (
            "Preceding context (for disambiguation only -- do NOT count anything in it):\n"
            f"<context>\n{context}\n</context>\n\n"
        )
    user += f"<segment>\n{chunk_text_}\n</segment>"
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user},
    ]


def _parse_response(text: str, with_evidence: bool) -> dict | None:
    """Parse one classifier response, or None if it is unusable.

    Structured decoding should make this total -- the grammar admits nothing else, and it is
    also why a `<think>` block cannot appear. The strip and the try/except are here for the
    case where the schema is ever relaxed, and because a generation truncated at
    --max-output-tokens yields a prefix of valid JSON, which is not valid JSON.
    """
    if _THINK_CLOSE in text:
        text = text.rsplit(_THINK_CLOSE, 1)[1]
    text = text.strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    out: dict = {}
    for name in BEHAVIORS:
        value = parsed.get(name)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            return None
        out[name] = value
    if with_evidence:
        for name in BEHAVIORS:
            quotes = parsed.get(f"{name}_evidence") or []
            out[f"{name}_evidence"] = [q for q in quotes if isinstance(q, str)]
    return out


def classify_chunks(llm, sampling_params, plan: list[dict], system_prompt: str,
                    with_evidence: bool) -> list[dict]:
    """Run the classifier over a prepared chunk plan and attach the parsed counts.

    `plan` rows carry their own provenance (question_idx, sample_idx, chunk_idx, char span), so
    the returned rows are self-describing and order-independent.
    """
    conversations = [
        build_messages(system_prompt, row["text"], row.get("context")) for row in plan
    ]
    outputs = llm.chat(
        conversations,
        sampling_params,
        # Belt and braces: the JSON grammar already makes a reasoning block unrepresentable.
        chat_template_kwargs={"enable_thinking": False},
    )
    rows = []
    for row, output in zip(plan, outputs, strict=True):
        completion = output.outputs[0]
        parsed = _parse_response(completion.text, with_evidence)
        scored = {
            "question_idx": row["question_idx"],
            "sample_idx": row["sample_idx"],
            "chunk_idx": row["chunk_idx"],
            "char_start": row["char_start"],
            "char_end": row["char_end"],
            "n_classifier_tokens": row["n_classifier_tokens"],
            "parse_failed": parsed is None,
        }
        if parsed is None:
            # Recorded, and excluded from BOTH numerator and denominator at aggregation.
            # Coercing a failure to zero would bias every rate downward, silently and in
            # proportion to how hard the segment was to read.
            scored["raw_response"] = completion.text[:400]
        else:
            for name in BEHAVIORS:
                scored[name] = parsed[name]
            if with_evidence:
                quotes = {name: parsed[f"{name}_evidence"] for name in BEHAVIORS}
                matched = sum(q in row["text"] for qs in quotes.values() for q in qs)
                total = sum(len(qs) for qs in quotes.values())
                scored["evidence"] = quotes
                scored["n_quotes"] = total
                scored["n_quotes_matched"] = matched
        rows.append(scored)
    return rows


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    """Pearson r, or None when either side is constant."""
    n = len(xs)
    if n < 2:
        return None
    mean_x, mean_y = sum(xs) / n, sum(ys) / n
    dx = [x - mean_x for x in xs]
    dy = [y - mean_y for y in ys]
    denominator = math.sqrt(sum(v * v for v in dx) * sum(v * v for v in dy))
    return None if denominator == 0 else sum(a * b for a, b in zip(dx, dy)) / denominator


def collapse_to_trajectories(chunk_rows: list[dict], source_rows: list[dict]) -> list[dict]:
    """Sum chunk counts per trajectory and rejoin the source record's fields.

    A trajectory with ANY failed chunk is dropped whole rather than partially counted: a
    per-1k rate needs its numerator and its token denominator to describe the same text.
    """
    by_key: dict[str, list[dict]] = defaultdict(list)
    for row in chunk_rows:
        by_key[f"{row['question_idx']}:{row['sample_idx']}"].append(row)

    trajectories = []
    for record in source_rows:
        key = trajectory_key(record)
        chunks = by_key.get(key)
        if not chunks or any(chunk["parse_failed"] for chunk in chunks):
            continue
        merged = {
            "question_idx": int(record["question_idx"]),
            "sample_idx": int(record["sample_idx"]),
            "n_chunks": len(chunks),
            # The teacher's own token count, NOT the classifier's: per-1k rates have to share a
            # denominator with teacher_uncertainty.py's e_per_1k_tokens or the two instruments
            # cannot be read on one axis. The classifier's vocabulary differs (248,320 vs
            # 151,669) and is tracked separately, for cost only.
            "n_tokens": int(record["n_tokens"]),
            "n_classifier_tokens": sum(c["n_classifier_tokens"] for c in chunks),
            "correct": bool(record["correct"]),
            "truncated": bool(record["truncated"]),
            "unclosed": bool(record["unclosed"]),
            "e_think": int(record["e_think"]),
            "e_total": int(record["e_total"]),
        }
        for name in BEHAVIORS:
            merged[name] = sum(chunk[name] for chunk in chunks)
        trajectories.append(merged)
    return trajectories


def _ratio_ci(
    trajectories: list[dict], numerator, denominator, bootstrap_samples: int, seed: int
) -> list[float]:
    """Question-clustered bootstrap CI for a ratio of two per-trajectory quantities."""
    totals: dict[str, list[float]] = defaultdict(lambda: [0.0, 0])
    for row in trajectories:
        bucket = totals[str(row["question_idx"])]
        bucket[0] += numerator(row)
        bucket[1] += denominator(row)
    typed = {key: (float(value[0]), int(value[1])) for key, value in totals.items()}
    return question_cluster_bootstrap_ci(typed, bootstrap_samples, seed)


def summarize_arm(
    trajectories: list[dict],
    chunk_rows: list[dict],
    bootstrap_samples: int,
    seed: int,
) -> dict:
    """Per-PI behavior statistics, plus the health and validity readings.

    `marker_correlation` is the convergent-validity check: the epistemic-marker regex covers
    roughly backtracking plus verification and nothing else, so those two should track E(y)
    within an arm. A weak or negative correlation means the rubric is measuring something
    other than what it claims, and is worth more attention than any between-arm difference.
    """
    n = len(trajectories)
    failed = sum(row["parse_failed"] for row in chunk_rows)
    summary: dict = {
        "n_trajectories": n,
        "n_chunks": len(chunk_rows),
        "n_chunks_parse_failed": failed,
        "parse_failure_rate": failed / len(chunk_rows) if chunk_rows else 0.0,
        "n_trajectories_dropped": None,  # filled by the caller, which knows the source count
    }
    if not n:
        return summary

    total_tokens = sum(row["n_tokens"] for row in trajectories)
    summary |= {
        "mean_tokens": total_tokens / n,
        "mean_chunks_per_trajectory": sum(r["n_chunks"] for r in trajectories) / n,
        "pass@1": sum(r["correct"] for r in trajectories) / n,
        "unclosed_rate": sum(r["unclosed"] for r in trajectories) / n,
        "mean_e_think": sum(r["e_think"] for r in trajectories) / n,
    }

    for index, name in enumerate(BEHAVIORS):
        count = sum(row[name] for row in trajectories)
        present = sum(row[name] > 0 for row in trajectories)
        summary[name] = {
            "mean_per_trajectory": count / n,
            "rate_per_1k": 1000 * count / total_tokens if total_tokens else 0.0,
            "prevalence": present / n,
            "total_count": count,
            "rate_per_1k_ci95": _ratio_ci(
                trajectories,
                lambda row, key=name: 1000 * row[key],
                lambda row: row["n_tokens"],
                bootstrap_samples,
                seed + index * 10_000,
            ),
            "prevalence_ci95": _ratio_ci(
                trajectories,
                lambda row, key=name: float(row[key] > 0),
                lambda _row: 1,
                bootstrap_samples,
                seed + index * 10_000 + 5_000,
            ),
        }

    summary["marker_correlation"] = {
        "backtracking_plus_verification_vs_e_think": _pearson(
            [row["backtracking"] + row["verification"] for row in trajectories],
            [float(row["e_think"]) for row in trajectories],
        ),
    }

    closed = [row for row in trajectories if not row["unclosed"]]
    summary["closed_only"] = {
        "n_trajectories": len(closed),
        **{
            name: {"mean_per_trajectory": sum(r[name] for r in closed) / len(closed)}
            for name in BEHAVIORS
        },
    } if closed else {"n_trajectories": 0}

    quoted = [row for row in chunk_rows if "n_quotes" in row]
    if quoted:
        total_quotes = sum(row["n_quotes"] for row in quoted)
        summary["quote_match_rate"] = (
            sum(row["n_quotes_matched"] for row in quoted) / total_quotes
            if total_quotes
            else None
        )
    return summary


# ---------------------------------------------------------------------------
# Run provenance
# ---------------------------------------------------------------------------


def build_run_config(args, teacher_slug: str) -> dict:
    """Everything that determines what the counts MEAN. Compared on reuse; a difference is a
    hard error unless --force, the same contract as advantage_dynamics' cohort/score caches."""
    return {
        "method": "teacher_cognitive_behaviors",
        "classifier_model": args.classifier_model,
        "teacher_model": args.teacher_model,
        "teacher_slug": teacher_slug,
        "completions_root": args.completions_root,
        "behaviors": list(BEHAVIORS),
        "rubric_version": BEHAVIOR_RUBRIC_VERSION,
        "rubric_fingerprint": rubric_fingerprint(),
        "chunk_tokens": args.chunk_tokens,
        "context_paragraphs": args.context_paragraphs,
        "samples_per_problem": args.samples_per_problem,
        "limit": args.limit,
        "evidence": args.evidence,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_output_tokens": args.max_output_tokens,
        "seed": args.seed,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--teacher-model", default="Qwen/Qwen3-1.7B",
                   help="Whose completions to classify. Names the input and output subdirs; "
                        "this model is never loaded.")
    p.add_argument("--classifier-model", default="Qwen/Qwen3.8-27B",
                   help="The labelling model. Hybrid reasoner: thinking is disabled via "
                        "chat_template_kwargs AND structurally by the JSON grammar.")
    p.add_argument("--completions-root", default="results/teacher_uncertainty_16k",
                   help="Root eval/teacher_uncertainty.py wrote to (--output-dir there).")
    p.add_argument("--output-root", default="results/teacher_behaviors_16k",
                   help="Separate from --completions-root so --force can never overwrite "
                        "generation output.")
    p.add_argument("--pi-modes", nargs="+", default=list(PI_MODES), choices=list(PI_MODES),
                   help="Arms to classify. All five share one problem set, so all five should "
                        "normally be run together.")
    p.add_argument("--samples-per-problem", type=int, default=4,
                   help="Keep sample_idx < N. Thins SAMPLES, never problems: the CI resamples "
                        "questions, so the marginal sample is cheap to give up and the "
                        "problem-level pairing across arms is what must survive. 0/None keeps all.")
    p.add_argument("--limit", type=int, default=None,
                   help="Cap trajectories per arm (smoke runs). Applied after thinning.")
    # rubric / segmentation
    p.add_argument("--chunk-tokens", type=int, default=1000,
                   help="Target segment size, measured in CLASSIFIER tokens. Paragraphs are "
                        "never split below this.")
    p.add_argument("--context-paragraphs", type=int, default=0,
                   help="Trailing paragraphs of the previous segment shown as uncountable "
                        "context. Off by default: it mitigates boundary misses but gives a 27B "
                        "something extra to miscount.")
    p.add_argument("--evidence", action="store_true",
                   help="Also request up to 3 verbatim quotes per behavior and report "
                        "quote_match_rate. Decode-bound -- intended for validation subsamples, "
                        "not the full sweep.")
    p.add_argument("--allow-placeholder-examples", action="store_true",
                   help="Run while EXAMPLES still hold placeholders. Smoke runs only: the "
                        "counts are meaningless and every arm will look alike.")
    # sampling
    p.add_argument("--temperature", type=float, default=0.7,
                   help="The model card recommends 0.7 for non-thinking ")
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--max-output-tokens", type=int, default=None,
                   help="Defaults to 1024, or 2048 with --evidence. A truncated response is a "
                        "parse failure, not a zero -- and because one failed segment drops its "
                        "whole trajectory, a budget set too low costs far more rows than it "
                        "saves tokens.")
    p.add_argument("--seed", type=int, default=42,
                   help="Bounds run-to-run variation; vLLM's continuous batching is not "
                        "bitwise deterministic across batch compositions.")
    p.add_argument("--bootstrap-samples", type=int, default=10_000,
                   help="Question-clustered bootstrap replicates.")
    # vLLM
    p.add_argument("--max-model-len", type=int, default=16384,
                   help="Rubric + one segment + the response fit far inside this. The model "
                        "supports 262k natively; asking for it here would spend the KV cache "
                        "on context no call uses.")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    p.add_argument("--tensor-parallel-size", type=int, default=1,
                   help="27B in bf16 is ~54GB of weights; raise this if it does not fit "
                        "alongside the KV cache.")
    p.add_argument("--max-num-seqs", type=int, default=256,
                   help="Concurrent sequences. The Qwen3.x-27B judges are HYBRID models: their Gated "
                        "DeltaNet layers need one Mamba cache block per running sequence, and "
                        "vLLM refuses to capture CUDA graphs when max_num_seqs exceeds the "
                        "block count it could fit (at 0.9 utilization on one 96GB card that is "
                        "~570, against vLLM's default of 1024). 256 clears it with room to "
                        "spare and still batches these short responses well. Not part of the "
                        "cached run config -- it changes throughput, not the counts.")
    p.add_argument("--enable-prefix-caching", action=argparse.BooleanOptionalAction, default=True,
                   help="The rubric is a constant prefix on every call. Leaving this on is "
                        "most of why chunking is affordable.")
    p.add_argument("--dry-run", action="store_true",
                   help="Plan the chunking and print the token cost without loading the engine.")
    p.add_argument("--force", action="store_true",
                   help="Reclassify arms whose cached provenance differs from this run's.")
    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.chunk_tokens < 1:
        raise SystemExit("--chunk-tokens must be >= 1")
    if args.context_paragraphs < 0:
        raise SystemExit("--context-paragraphs must be >= 0")
    if args.samples_per_problem is not None and args.samples_per_problem <= 0:
        args.samples_per_problem = None
    if args.max_output_tokens is None:
        args.max_output_tokens = 2048 if args.evidence else 1024
    if not examples_are_filled() and not args.allow_placeholder_examples:
        raise SystemExit(
            f"EXAMPLES still contains the placeholder sentinel ({EXAMPLE_PLACEHOLDER!r}).\n"
            "Fill the few-shot segments from labelled DeepMath traces first -- a full sweep is "
            "tens of millions of tokens and a placeholder rubric would flatten every arm into "
            "the same numbers.\nPass --allow-placeholder-examples for a smoke run."
        )

    teacher_slug = args.teacher_model.replace("/", "_")
    out_dir = Path(args.output_root) / teacher_slug
    config = build_run_config(args, teacher_slug)

    from transformers import AutoTokenizer

    # Loaded before the engine so --dry-run costs nothing but CPU.
    tokenizer = AutoTokenizer.from_pretrained(args.classifier_model, trust_remote_code=True)
    system_prompt = render_system_prompt()
    system_tokens = len(tokenizer(system_prompt, add_special_tokens=False)["input_ids"])
    print(f"teacher: {args.teacher_model}  classifier: {args.classifier_model}")
    print(f"  rubric {BEHAVIOR_RUBRIC_VERSION} ({system_tokens} tokens, cached prefix) "
          f"fp={rubric_fingerprint()[:12]}")
    if not examples_are_filled():
        print("  WARNING: placeholder examples -- these counts mean nothing.")

    # -- plan every arm first, so the cost is known before any GPU time is spent ------
    plans: dict[str, dict] = {}
    for pi_mode in args.pi_modes:
        source_rows = load_completions(
            completions_path(args.completions_root, args.teacher_model, pi_mode),
            args.samples_per_problem,
            args.limit,
        )
        if not source_rows:
            raise SystemExit(f"No completions selected for pi={pi_mode}")
        chunk_plan = []
        for record in source_rows:
            chunks = chunk_completion(record["text"], tokenizer, args.chunk_tokens)
            previous: list[str] = []
            for chunk in chunks:
                entry = {
                    "question_idx": int(record["question_idx"]),
                    "sample_idx": int(record["sample_idx"]),
                    **chunk,
                }
                if args.context_paragraphs and previous:
                    entry["context"] = "\n\n".join(previous[-args.context_paragraphs :])
                chunk_plan.append(entry)
                previous = [chunk["text"][s:e] for s, e in paragraph_spans(chunk["text"])]
        plans[pi_mode] = {
            "source_rows": source_rows,
            "chunk_plan": chunk_plan,
            "source_fingerprint": source_fingerprint(source_rows),
        }
        segment_tokens = sum(row["n_classifier_tokens"] for row in chunk_plan)
        print(f"  {pi_mode:8s} {len(source_rows):5d} trajectories -> {len(chunk_plan):6d} "
              f"segments, {segment_tokens / 1e6:5.2f}M segment tokens "
              f"(+{len(chunk_plan) * system_tokens / 1e6:5.2f}M cached prefix)")

    if args.dry_run:
        total = sum(
            sum(row["n_classifier_tokens"] for row in plan["chunk_plan"])
            for plan in plans.values()
        )
        n_calls = sum(len(plan["chunk_plan"]) for plan in plans.values())
        print(f"\nDry run: {n_calls} calls, {total / 1e6:.2f}M segment tokens, "
              f"{n_calls * args.max_output_tokens / 1e6:.2f}M output tokens at worst.")
        return

    # -- which arms actually need the engine ------------------------------------------
    pending = []
    for pi_mode in args.pi_modes:
        meta_path = out_dir / f"behaviors_meta_{pi_mode}.json"
        rows_path = out_dir / f"behaviors_{pi_mode}.jsonl"
        expected = {**config, "source_fingerprint": plans[pi_mode]["source_fingerprint"]}
        if rows_path.is_file() and meta_path.is_file() and not args.force:
            if read_json(meta_path).get("config") == expected:
                print(f"  reusing cached {pi_mode} ({rows_path})")
                continue
            raise SystemExit(
                f"Cached {rows_path} was produced under different provenance. Pass --force to "
                "reclassify, or point --output-root somewhere else."
            )
        pending.append(pi_mode)

    if pending:
        from vllm import LLM, SamplingParams
        from vllm.sampling_params import StructuredOutputsParams

        llm = LLM(
            model=args.classifier_model,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            tensor_parallel_size=args.tensor_parallel_size,
            enable_prefix_caching=args.enable_prefix_caching,
            max_num_seqs=args.max_num_seqs,
            # NOT optional, and it has to be set HERE. A JSON grammar admits arbitrary
            # whitespace between members, and this model exploits it: it emits the four
            # counts correctly, then runs newlines and tabs until it hits
            # --max-output-tokens, so the response is truncated before the closing brace and
            # fails to parse. Measured at 110 of 138 segments before this was set.
            #
            # SamplingParams.structured_outputs also has a `disable_any_whitespace` field and
            # it is IGNORED: vllm/v1/structured_output/backend_xgrammar.py reads
            # `vllm_config.structured_outputs_config`, i.e. the engine-level config below,
            # never the per-request one. Setting it per request looks right, changes nothing,
            # and shows up as a parse-failure rate rather than an error.
            # `backend` must be named explicitly: the validator rejects
            # disable_any_whitespace while the backend is still "auto", even though auto
            # resolves to xgrammar for this schema anyway.
            structured_outputs_config={
                "backend": "xgrammar",
                "disable_any_whitespace": True,
            },
            seed=args.seed,
            trust_remote_code=True,
        )
        sampling_params = SamplingParams(
            n=1,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_output_tokens,
            seed=args.seed,
            structured_outputs=StructuredOutputsParams(
                json=response_schema(args.evidence)
            ),
        )
        for pi_mode in pending:
            plan = plans[pi_mode]
            print(f"\nClassifying {pi_mode}: {len(plan['chunk_plan'])} segments")
            chunk_rows = classify_chunks(
                llm, sampling_params, plan["chunk_plan"], system_prompt, args.evidence
            )
            out_dir.mkdir(parents=True, exist_ok=True)
            rows_path = out_dir / f"behaviors_{pi_mode}.jsonl"
            # Same staging convention as advantage_dynamics.write_json_atomic: a dot-prefixed
            # sibling, so a crashed run leaves no half-written behaviors_<pi>.jsonl behind.
            temporary = rows_path.with_name(f".{rows_path.name}.tmp-{os.getpid()}")
            with temporary.open("w") as handle:
                for row in chunk_rows:
                    handle.write(json.dumps(row) + "\n")
            os.replace(temporary, rows_path)
            write_json_atomic(
                out_dir / f"behaviors_meta_{pi_mode}.json",
                {
                    "status": "complete",
                    "config": {**config, "source_fingerprint": plan["source_fingerprint"]},
                    "pi_mode": pi_mode,
                    "n_trajectories": len(plan["source_rows"]),
                    "n_chunks": len(chunk_rows),
                },
            )
            print(f"  saved -> {rows_path}")

    # -- aggregate -------------------------------------------------------------------
    summary: dict = {}
    for pi_mode in args.pi_modes:
        chunk_rows = [
            json.loads(line)
            for line in (out_dir / f"behaviors_{pi_mode}.jsonl").open()
        ]
        source_rows = plans[pi_mode]["source_rows"]
        trajectories = collapse_to_trajectories(chunk_rows, source_rows)
        arm = summarize_arm(trajectories, chunk_rows, args.bootstrap_samples, args.seed)
        arm["n_trajectories_dropped"] = len(source_rows) - len(trajectories)
        summary[pi_mode] = arm

    write_json_atomic(
        out_dir / "behaviors_summary.json",
        {
            "method": "teacher_cognitive_behaviors",
            "paper": "arXiv:2503.01307",
            "config": config,
            "source_fingerprints": {
                pi_mode: plans[pi_mode]["source_fingerprint"] for pi_mode in args.pi_modes
            },
            "epistemic_markers": EPISTEMIC_MARKERS,
            "behavior": summary,
        },
    )

    header = "  ".join(f"{name[:9]:>9s}/1k" for name in BEHAVIORS)
    print(f"\n{'PI':8s}{'tokens':>8s}{'E/1k':>7s}  {header}   r(bt+vf,E)")
    for pi_mode in args.pi_modes:
        arm = summary[pi_mode]
        if not arm.get("n_trajectories"):
            print(f"{pi_mode:8s}  (no usable trajectories)")
            continue
        rates = "  ".join(f"{arm[name]['rate_per_1k']:12.2f}" for name in BEHAVIORS)
        marker_e = 1000 * arm["mean_e_think"] / arm["mean_tokens"] if arm["mean_tokens"] else 0.0
        correlation = arm["marker_correlation"]["backtracking_plus_verification_vs_e_think"]
        print(f"{pi_mode:8s}{arm['mean_tokens']:8.0f}{marker_e:7.2f}  {rates}   "
              f"{'n/a' if correlation is None else f'{correlation:+.2f}'}")
    print(f"\nSaved -> {out_dir / 'behaviors_summary.json'}")


if __name__ == "__main__":
    main()
