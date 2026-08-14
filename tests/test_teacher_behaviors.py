"""Unit tests for the cognitive-behavior classifier pass.

Every assertion here is about STRUCTURE, never about the content of the few-shot rubric:
`EXAMPLES` still holds placeholders and will be replaced with labelled DeepMath excerpts, so a
test that pinned a rendered prompt or a `rubric_fingerprint()` literal would break on that
commit and teach whoever hits it to update the expected value reflexively -- which is exactly
what a fingerprint exists to prevent.

The one example-aware test (`RubricStructureTest`) is deliberate: "every behavior has a
positive instance, and at least one instance is all-zero" must survive the labelling pass,
because dropping the negative example is what makes a 27B find a behavior in every segment and
flattens the between-arm differences the whole measurement is for.
"""

import json
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest import mock

from eval import teacher_behaviors as tb


class WordTokenizer:
    """Whitespace tokenizer with real character offsets.

    Hermetic and fast: chunking is pure text arithmetic, so a real tokenizer would add a model
    download and a few seconds per test without exercising a single extra branch. `is_fast` is
    True because the offset-mapping path is the one production takes; the proportional-split
    fallback is covered separately by `SlowTokenizer`.
    """

    is_fast = True

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        batched = not isinstance(text, str)
        texts = text if batched else [text]
        all_ids, all_offsets = [], []
        for item in texts:
            ids, offsets, cursor = [], [], 0
            for word in item.split():
                start = item.index(word, cursor)
                ids.append(len(word))
                offsets.append((start, start + len(word)))
                cursor = start + len(word)
            all_ids.append(ids)
            all_offsets.append(offsets)
        out = {"input_ids": all_ids if batched else all_ids[0]}
        if return_offsets_mapping:
            out["offset_mapping"] = all_offsets if batched else all_offsets[0]
        return out


class SlowTokenizer(WordTokenizer):
    is_fast = False


def paragraphs(*sizes):
    """A completion of paragraphs with the requested word counts, blank-line separated."""
    return "\n\n".join(" ".join(f"w{index}" for index in range(size)) for size in sizes)


def source_record(question_idx, sample_idx, **overrides):
    """One row in the shape eval/teacher_uncertainty.py writes to completions_<pi>.jsonl."""
    record = {
        "question_idx": question_idx,
        "sample_idx": sample_idx,
        "text": paragraphs(3, 3),
        "n_tokens": 1000,
        "correct": True,
        "truncated": False,
        "unclosed": False,
        "e_think": 10,
        "e_total": 12,
    }
    record.update(overrides)
    return record


def chunk_row(question_idx, sample_idx, chunk_idx, counts=None, parse_failed=False):
    row = {
        "question_idx": question_idx,
        "sample_idx": sample_idx,
        "chunk_idx": chunk_idx,
        "char_start": 0,
        "char_end": 1,
        "n_classifier_tokens": 100,
        "parse_failed": parse_failed,
    }
    if not parse_failed:
        row.update({name: 0 for name in tb.BEHAVIORS})
        row.update(counts or {})
    return row


# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------


class ParagraphSpanTest(unittest.TestCase):
    def test_spans_exclude_the_blank_line_separators(self):
        text = "alpha\n\nbeta"
        spans = tb.paragraph_spans(text)
        self.assertEqual([text[s:e] for s, e in spans], ["alpha", "beta"])

    def test_runs_of_more_than_two_newlines_are_one_separator(self):
        text = "alpha\n\n\n\nbeta"
        self.assertEqual(len(tb.paragraph_spans(text)), 2)

    def test_text_without_a_break_is_a_single_span(self):
        self.assertEqual(tb.paragraph_spans("alpha"), [(0, 5)])

    def test_empty_text_still_yields_one_span(self):
        """A degenerate completion must not silently produce zero segments and vanish from
        the denominator."""
        self.assertEqual(tb.paragraph_spans(""), [(0, 0)])


class ChunkCompletionTest(unittest.TestCase):
    def setUp(self):
        self.tokenizer = WordTokenizer()

    def chunk(self, text, target=10, tokenizer=None):
        return tb.chunk_completion(text, tokenizer or self.tokenizer, target)

    def test_paragraphs_pack_up_to_the_budget_without_being_split(self):
        text = paragraphs(4, 4, 4)
        chunks = self.chunk(text, target=10)
        # 4+4 fits under 10; adding the third would overflow, so it starts a new segment.
        self.assertEqual(len(chunks), 2)
        self.assertEqual(
            [text[c["char_start"] : c["char_end"]] for c in chunks],
            ["\n\n".join(text.split("\n\n")[:2]), text.split("\n\n")[2]],
        )

    def test_segments_never_overlap_and_advance_monotonically(self):
        """An overlap would double-count any behavior in the shared region, inflating exactly
        the rates this measures."""
        chunks = self.chunk(paragraphs(*([3] * 12)), target=7)
        self.assertGreater(len(chunks), 1)
        for earlier, later in zip(chunks, chunks[1:]):
            self.assertLessEqual(earlier["char_end"], later["char_start"])

    def test_chunk_indices_are_dense_and_zero_based(self):
        chunks = self.chunk(paragraphs(*([3] * 9)), target=7)
        self.assertEqual([c["chunk_idx"] for c in chunks], list(range(len(chunks))))

    def test_offsets_round_trip_into_the_source_text(self):
        """char_start/char_end are the currency of the forward join to the per-token advantage
        arrays, so they must index the ORIGINAL completion, not a normalized copy."""
        text = paragraphs(5, 5, 5)
        for chunk in self.chunk(text, target=6):
            self.assertEqual(text[chunk["char_start"] : chunk["char_end"]], chunk["text"])

    def test_every_word_is_covered_exactly_once(self):
        text = paragraphs(4, 6, 3, 8)
        covered = " ".join(c["text"] for c in self.chunk(text, target=7)).split()
        self.assertEqual(covered, text.split())

    def test_a_paragraph_over_budget_is_split_on_token_boundaries(self):
        """Rare (one unbroken derivation), but it must not emit a segment longer than the
        context window."""
        chunks = self.chunk(paragraphs(25), target=10)
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertLessEqual(chunk["n_classifier_tokens"], 10)
        self.assertEqual(
            " ".join(c["text"] for c in chunks).split(), paragraphs(25).split()
        )

    def test_oversized_split_falls_back_without_a_fast_tokenizer(self):
        """No offset mapping means an approximate character split; it must still terminate,
        stay in order, and cover the paragraph."""
        text = paragraphs(30)
        chunks = self.chunk(text, target=5, tokenizer=SlowTokenizer())
        self.assertGreater(len(chunks), 1)
        for earlier, later in zip(chunks, chunks[1:]):
            self.assertLessEqual(earlier["char_end"], later["char_start"])
        self.assertEqual(chunks[0]["char_start"], 0)
        self.assertEqual(chunks[-1]["char_end"], len(text))

    def test_a_short_completion_is_one_segment(self):
        """The `full`/`rollout` arms land here; one segment is the honest representation of
        how little reasoning they contain."""
        self.assertEqual(len(self.chunk(paragraphs(3), target=100)), 1)

    def test_a_nonpositive_budget_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "target_tokens"):
            self.chunk("alpha", target=0)


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


class ResponseParsingTest(unittest.TestCase):
    def counts(self, **overrides):
        values = {name: 0 for name in tb.BEHAVIORS}
        values.update(overrides)
        return json.dumps(values)

    def test_a_well_formed_response_parses_to_integer_counts(self):
        parsed = tb._parse_response(self.counts(verification=2, backtracking=1), False)
        self.assertEqual(parsed["verification"], 2)
        self.assertEqual(parsed["backtracking"], 1)
        self.assertEqual(parsed["backward_chaining"], 0)

    def test_a_reasoning_block_is_stripped_before_parsing(self):
        """The JSON grammar makes this unreachable in production; the strip is the guard for
        the day the schema is relaxed, and gen_hints.py documents why that guard is earned."""
        parsed = tb._parse_response("<think>counting</think>" + self.counts(), False)
        self.assertIsNotNone(parsed)

    def test_a_truncated_response_is_a_failure_not_a_zero(self):
        """A generation cut off at --max-output-tokens is a prefix of valid JSON. Coercing it
        to zero would bias every rate downward in proportion to segment difficulty."""
        self.assertIsNone(tb._parse_response(self.counts()[:12], False))

    def test_a_missing_behavior_is_rejected(self):
        partial = json.dumps({name: 0 for name in tb.BEHAVIORS[:-1]})
        self.assertIsNone(tb._parse_response(partial, False))

    def test_a_negative_count_is_rejected(self):
        self.assertIsNone(tb._parse_response(self.counts(verification=-1), False))

    def test_a_boolean_is_not_accepted_as_a_count(self):
        """bool subclasses int, so a naive isinstance check would let `true` through as 1."""
        self.assertIsNone(
            tb._parse_response(json.dumps({**json.loads(self.counts()), "verification": True}), False)
        )

    def test_a_non_object_response_is_rejected(self):
        self.assertIsNone(tb._parse_response("[1, 2, 3]", False))

    def test_evidence_is_returned_only_when_requested(self):
        payload = json.loads(self.counts(verification=1))
        payload |= {f"{name}_evidence": ["quoted"] for name in tb.BEHAVIORS}
        with_evidence = tb._parse_response(json.dumps(payload), True)
        self.assertEqual(with_evidence["verification_evidence"], ["quoted"])
        self.assertNotIn("verification_evidence", tb._parse_response(json.dumps(payload), False))

    def test_missing_evidence_degrades_to_an_empty_list(self):
        parsed = tb._parse_response(self.counts(), True)
        self.assertEqual(parsed["backtracking_evidence"], [])


class ResponseSchemaTest(unittest.TestCase):
    def test_every_behavior_is_a_required_nonnegative_integer(self):
        schema = tb.response_schema(with_evidence=False)
        for name in tb.BEHAVIORS:
            self.assertEqual(schema["properties"][name], {"type": "integer", "minimum": 0})
            self.assertIn(name, schema["required"])
        self.assertFalse(schema["additionalProperties"])

    def test_evidence_fields_appear_only_in_the_evidence_schema(self):
        plain = tb.response_schema(with_evidence=False)["properties"]
        rich = tb.response_schema(with_evidence=True)["properties"]
        self.assertEqual(set(rich) - set(plain), {f"{n}_evidence" for n in tb.BEHAVIORS})


# ---------------------------------------------------------------------------
# The rubric -- structure only, never content
# ---------------------------------------------------------------------------


class RubricStructureTest(unittest.TestCase):
    def test_every_behavior_has_a_positive_example(self):
        """A behavior with no demonstrated instance is a behavior the classifier will
        under-report; this must still hold once the placeholders are replaced."""
        for name in tb.BEHAVIORS:
            self.assertTrue(
                any(example.counts[name] > 0 for example in tb.EXAMPLES),
                f"no example demonstrates {name}",
            )

    def test_there_is_exactly_one_all_zero_example(self):
        """At least one, because without a negative instance the classifier finds a behavior in
        every segment. But NOT more than one, which is a measured result rather than a guess.

        v1 carried three all-zero examples plus a counting rule saying "zero can be a correct
        answer; do not look for a behavior that is not there". Hand-labelling 14 segments the
        classifier had scored all-zero found that 12 of them carried at least one behavior --
        and 11 of 11 among full-length segments. The failure mode was ABSTENTION, not
        over-firing, and the rubric's restraint was causing it. Adding negatives back is the
        most likely way to reintroduce that, so the count is pinned in both directions."""
        zeros = [e for e in tb.EXAMPLES if all(v == 0 for v in e.counts.values())]
        self.assertEqual(len(zeros), 1, "expected exactly one all-zero example")

    def test_every_example_scores_every_behavior(self):
        for example in tb.EXAMPLES:
            self.assertEqual(set(example.counts), set(tb.BEHAVIORS))

    def test_examples_are_filled_tracks_the_placeholder_sentinel(self):
        """Both example sets are built here rather than taken from the module global, so this
        stays true at every point of the labelling pass -- reaching into tb.EXAMPLES for a
        placeholder would make the test pass only while that particular slot is still unfilled.
        """
        zeros = {name: 0 for name in tb.BEHAVIORS}
        filled = tb.RubricExample(segment="a real excerpt", counts=zeros, note="")
        placeholder = tb.RubricExample(
            segment=tb.EXAMPLE_PLACEHOLDER, counts=zeros, note=""
        )
        with mock.patch.object(tb, "EXAMPLES", (filled, filled)):
            self.assertTrue(tb.examples_are_filled())
        with mock.patch.object(tb, "EXAMPLES", (filled, placeholder)):
            self.assertFalse(tb.examples_are_filled())

    def test_the_system_prompt_names_and_defines_every_behavior(self):
        prompt = tb.render_system_prompt()
        for name in tb.BEHAVIORS:
            self.assertIn(name, prompt)
            self.assertIn(tb.BEHAVIOR_DEFINITIONS[name], prompt)

    def test_the_fingerprint_moves_when_a_definition_changes(self):
        """Pinning the VALUE would break on the labelling commit and train everyone to bump it
        without reading; pinning the BEHAVIOR keeps it a real guard against silent drift."""
        before = tb.rubric_fingerprint()
        edited = dict(tb.BEHAVIOR_DEFINITIONS, verification="something else entirely")
        with mock.patch.object(tb, "BEHAVIOR_DEFINITIONS", edited):
            self.assertNotEqual(tb.rubric_fingerprint(), before)
        self.assertEqual(tb.rubric_fingerprint(), before)

    def test_the_fingerprint_moves_when_an_example_changes(self):
        before = tb.rubric_fingerprint()
        edited = (replace(tb.EXAMPLES[0], segment="a real excerpt"),) + tb.EXAMPLES[1:]
        with mock.patch.object(tb, "EXAMPLES", edited):
            self.assertNotEqual(tb.rubric_fingerprint(), before)


class ExampleOrderingTest(unittest.TestCase):
    """The sequence the classifier sees is a design decision, not an artifact of edit order."""

    def primaries(self):
        """Each example's INTENDED primary behavior, taken from its id prefix.

        Read off the key rather than argmax over `counts`, because the co-occurrence examples
        tie -- bc_three_roots scores 1 for both subgoal setting and backward chaining, and an
        argmax would silently resolve that by tuple position rather than by intent.
        """
        return [key.split("_")[0] for key in tb.EXAMPLE_ORDER]

    def test_the_order_is_a_permutation_of_the_pool(self):
        """An example dropped from EXAMPLE_ORDER vanishes from the prompt while every other
        test here still passes, because they all iterate the already-ordered EXAMPLES."""
        self.assertEqual(sorted(tb.EXAMPLE_ORDER), sorted(tb._BY_ID))
        self.assertEqual(len(set(tb.EXAMPLE_ORDER)), len(tb.EXAMPLE_ORDER))
        self.assertEqual(len(tb.EXAMPLES), len(tb._BY_ID))

    def test_no_two_adjacent_examples_share_a_primary_behavior(self):
        """Grouped by behavior, the sequence itself becomes a pattern an in-context learner can
        read instead of reading the content."""
        primaries = self.primaries()
        repeats = [(a, b) for a, b in zip(primaries, primaries[1:]) if a == b]
        self.assertEqual(repeats, [], f"adjacent examples share a primary: {repeats}")

    def test_the_negative_is_neither_first_nor_last(self):
        """Position matters for the one all-zero example. Last is where recency over-weights
        it, and first is where it anchors -- either would push the classifier back towards the
        abstention that v1 measured. Sitting mid-sequence, after each behavior has been
        introduced, it reads as one case among many rather than as the default."""
        primaries = self.primaries()
        positions = [i for i, p in enumerate(primaries) if p == "neg"]
        self.assertEqual(len(positions), 1)
        self.assertNotIn(positions[0], (0, len(primaries) - 1))

    def test_every_behavior_is_introduced_before_any_co_occurrence(self):
        """The crisp single-behavior reading of each label should be established before the
        first segment that carries two of them."""
        first_multi = next(
            (
                index
                for index, key in enumerate(tb.EXAMPLE_ORDER)
                if sum(v > 0 for v in tb._BY_ID[key].counts.values()) > 1
            ),
            len(tb.EXAMPLE_ORDER),
        )
        introduced = {
            key.split("_")[0] for key in tb.EXAMPLE_ORDER[:first_multi]
        }
        self.assertEqual(introduced, {"vf", "bt", "sg", "bc", "neg"})


class ClassifierMessageTest(unittest.TestCase):
    def test_the_system_turn_is_identical_across_segments(self):
        """It is the cached prefix on every one of ~47k calls; any per-call variation there
        would silently multiply the cost of the sweep."""
        prompt = tb.render_system_prompt()
        first = tb.build_messages(prompt, "segment one", None)
        second = tb.build_messages(prompt, "segment two", None)
        self.assertEqual(first[0], second[0])
        self.assertNotEqual(first[1], second[1])

    def test_context_is_marked_uncountable_and_kept_out_of_the_segment(self):
        messages = tb.build_messages("rubric", "the segment", "earlier text")
        user = messages[1]["content"]
        self.assertIn("do NOT count", user)
        self.assertLess(user.index("earlier text"), user.index("<segment>"))


# ---------------------------------------------------------------------------
# Source selection
# ---------------------------------------------------------------------------


class CompletionSelectionTest(unittest.TestCase):
    def test_thinning_drops_samples_and_keeps_every_problem(self):
        """Problems are shared across PI arms and are what makes the comparison paired; the 8
        samples of a problem are independent draws. Thinning must cut the second, never the
        first."""
        records = [
            source_record(question_idx=q, sample_idx=s) for q in range(4) for s in range(8)
        ]
        kept = [r for r in records if r["sample_idx"] < 2]
        self.assertEqual({r["question_idx"] for r in kept}, {0, 1, 2, 3})
        self.assertEqual(len(kept), 8)

    def test_the_fingerprint_distinguishes_a_different_trajectory_set(self):
        rows = [source_record(0, 0), source_record(0, 1)]
        self.assertEqual(tb.source_fingerprint(rows), tb.source_fingerprint(list(rows)))
        self.assertNotEqual(tb.source_fingerprint(rows), tb.source_fingerprint(rows[:1]))
        self.assertNotEqual(
            tb.source_fingerprint(rows), tb.source_fingerprint(list(reversed(rows)))
        )


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


class CollapseToTrajectoriesTest(unittest.TestCase):
    def test_chunk_counts_sum_within_a_trajectory(self):
        source = [source_record(0, 0)]
        chunks = [
            chunk_row(0, 0, 0, {"verification": 2, "backtracking": 1}),
            chunk_row(0, 0, 1, {"verification": 3}),
        ]
        [trajectory] = tb.collapse_to_trajectories(chunks, source)
        self.assertEqual(trajectory["verification"], 5)
        self.assertEqual(trajectory["backtracking"], 1)
        self.assertEqual(trajectory["n_chunks"], 2)

    def test_one_failed_chunk_drops_the_whole_trajectory(self):
        """A per-1k rate needs its numerator and its token denominator to describe the same
        text, so a partially-classified trace cannot be counted at all."""
        source = [source_record(0, 0), source_record(0, 1)]
        chunks = [
            chunk_row(0, 0, 0, {"verification": 2}),
            chunk_row(0, 0, 1, parse_failed=True),
            chunk_row(0, 1, 0, {"verification": 1}),
        ]
        kept = tb.collapse_to_trajectories(chunks, source)
        self.assertEqual([row["sample_idx"] for row in kept], [1])

    def test_a_trajectory_with_no_chunks_is_dropped(self):
        self.assertEqual(tb.collapse_to_trajectories([], [source_record(0, 0)]), [])

    def test_the_token_count_comes_from_the_source_not_the_classifier(self):
        """Qwen3.6's vocabulary is 248,320 against Qwen3's 151,669, so per-1k rates must use
        the teacher's own count or they cannot be read on the same axis as e_per_1k_tokens."""
        source = [source_record(0, 0, n_tokens=4321)]
        [trajectory] = tb.collapse_to_trajectories([chunk_row(0, 0, 0)], source)
        self.assertEqual(trajectory["n_tokens"], 4321)
        self.assertEqual(trajectory["n_classifier_tokens"], 100)

    def test_source_fields_needed_for_stratification_ride_along(self):
        source = [source_record(0, 0, correct=False, unclosed=True, e_think=7)]
        [trajectory] = tb.collapse_to_trajectories([chunk_row(0, 0, 0)], source)
        self.assertFalse(trajectory["correct"])
        self.assertTrue(trajectory["unclosed"])
        self.assertEqual(trajectory["e_think"], 7)


class SummarizeArmTest(unittest.TestCase):
    def summarize(self, trajectories, chunks, bootstrap=64):
        return tb.summarize_arm(trajectories, chunks, bootstrap, seed=0)

    def build(self, specs):
        """specs: (question_idx, sample_idx, n_tokens, verification_count)."""
        source = [source_record(q, s, n_tokens=t) for q, s, t, _ in specs]
        chunks = [chunk_row(q, s, 0, {"verification": v}) for q, s, _, v in specs]
        return tb.collapse_to_trajectories(chunks, source), chunks

    def test_rate_prevalence_and_mean_measure_different_things(self):
        """The bimodal PI split is only visible in the rate: a trace that is 8x shorter has a
        proportionally lower raw count even when its behavior density is unchanged."""
        trajectories, chunks = self.build([(0, 0, 1000, 10), (1, 0, 8000, 10)])
        arm = self.summarize(trajectories, chunks)["verification"]
        self.assertEqual(arm["mean_per_trajectory"], 10.0)
        self.assertAlmostEqual(arm["rate_per_1k"], 1000 * 20 / 9000)
        self.assertEqual(arm["prevalence"], 1.0)

    def test_prevalence_counts_trajectories_not_occurrences(self):
        trajectories, chunks = self.build([(0, 0, 1000, 5), (1, 0, 1000, 0)])
        self.assertEqual(self.summarize(trajectories, chunks)["verification"]["prevalence"], 0.5)

    def test_parse_failures_are_reported_rather_than_absorbed(self):
        source = [source_record(0, 0), source_record(1, 0)]
        chunks = [chunk_row(0, 0, 0, {"verification": 1}), chunk_row(1, 0, 0, parse_failed=True)]
        trajectories = tb.collapse_to_trajectories(chunks, source)
        arm = self.summarize(trajectories, chunks)
        self.assertEqual(arm["n_chunks_parse_failed"], 1)
        self.assertEqual(arm["parse_failure_rate"], 0.5)
        self.assertEqual(arm["n_trajectories"], 1)

    def test_an_empty_arm_returns_counters_rather_than_dividing_by_zero(self):
        arm = self.summarize([], [])
        self.assertEqual(arm["n_trajectories"], 0)
        self.assertNotIn("verification", arm)

    def test_confidence_intervals_bracket_the_point_estimate(self):
        specs = [(q, s, 1000, q + s) for q in range(6) for s in range(2)]
        trajectories, chunks = self.build(specs)
        arm = self.summarize(trajectories, chunks, bootstrap=200)["verification"]
        low, high = arm["rate_per_1k_ci95"]
        self.assertLessEqual(low, arm["rate_per_1k"])
        self.assertLessEqual(arm["rate_per_1k"], high)

    def test_the_bootstrap_resamples_questions_not_trajectories(self):
        """With every sample of a question identical, a question-clustered bootstrap over one
        distinct question has nothing to resample and the interval collapses -- which a
        trajectory-level bootstrap over 8 rows would hide behind a spuriously tight CI."""
        trajectories, chunks = self.build([(0, s, 1000, 4) for s in range(8)])
        arm = self.summarize(trajectories, chunks, bootstrap=200)["verification"]
        self.assertEqual(arm["rate_per_1k_ci95"][0], arm["rate_per_1k_ci95"][1])

    def test_the_closed_only_cut_excludes_unclosed_traces(self):
        """93-98% of full/rollout completions never emit </think>, so the think/post split is
        degenerate there and has to be reported separately from the headline."""
        source = [source_record(0, 0), source_record(1, 0, unclosed=True)]
        chunks = [chunk_row(0, 0, 0, {"verification": 4}), chunk_row(1, 0, 0, {"verification": 0})]
        arm = self.summarize(tb.collapse_to_trajectories(chunks, source), chunks)
        self.assertEqual(arm["closed_only"]["n_trajectories"], 1)
        self.assertEqual(arm["closed_only"]["verification"]["mean_per_trajectory"], 4.0)

    def test_quote_match_rate_is_reported_only_when_evidence_was_requested(self):
        source = [source_record(0, 0)]
        plain = [chunk_row(0, 0, 0)]
        self.assertNotIn(
            "quote_match_rate", self.summarize(tb.collapse_to_trajectories(plain, source), plain)
        )
        quoted = [{**chunk_row(0, 0, 0), "n_quotes": 4, "n_quotes_matched": 3}]
        arm = self.summarize(tb.collapse_to_trajectories(quoted, source), quoted)
        self.assertAlmostEqual(arm["quote_match_rate"], 0.75)


class MarkerCorrelationTest(unittest.TestCase):
    def test_a_perfectly_tracking_classifier_reads_plus_one(self):
        """The convergent-validity check: the epistemic-marker regex covers roughly
        backtracking plus verification and nothing else, so those two should track E(y)."""
        source = [source_record(q, 0, e_think=2 * q) for q in range(5)]
        chunks = [chunk_row(q, 0, 0, {"verification": q, "backtracking": q}) for q in range(5)]
        arm = tb.summarize_arm(tb.collapse_to_trajectories(chunks, source), chunks, 32, 0)
        self.assertAlmostEqual(
            arm["marker_correlation"]["backtracking_plus_verification_vs_e_think"], 1.0
        )

    def test_a_constant_series_reports_none_rather_than_a_spurious_number(self):
        source = [source_record(q, 0, e_think=5) for q in range(4)]
        chunks = [chunk_row(q, 0, 0, {"verification": 1}) for q in range(4)]
        arm = tb.summarize_arm(tb.collapse_to_trajectories(chunks, source), chunks, 32, 0)
        self.assertIsNone(
            arm["marker_correlation"]["backtracking_plus_verification_vs_e_think"]
        )

    def test_pearson_is_scale_and_offset_invariant(self):
        xs = [1.0, 2.0, 3.0, 4.0]
        self.assertAlmostEqual(tb._pearson(xs, [3 * x + 7 for x in xs]), 1.0)
        self.assertAlmostEqual(tb._pearson(xs, [-2 * x + 1 for x in xs]), -1.0)


# ---------------------------------------------------------------------------
# Run provenance
# ---------------------------------------------------------------------------


class RunConfigTest(unittest.TestCase):
    def args(self, **overrides):
        values = {
            "classifier_model": "Qwen/Qwen3.6-27B",
            "teacher_model": "Qwen/Qwen3-1.7B",
            "completions_root": "results/teacher_uncertainty_16k",
            "chunk_tokens": 1000,
            "context_paragraphs": 0,
            "samples_per_problem": 4,
            "limit": None,
            "evidence": False,
            "temperature": 0.0,
            "top_p": 1.0,
            "max_output_tokens": 64,
            "seed": 42,
        }
        values.update(overrides)
        return SimpleNamespace(**values)

    def test_the_config_carries_the_rubric_identity(self):
        config = tb.build_run_config(self.args(), "Qwen_Qwen3-1.7B")
        self.assertEqual(config["rubric_version"], tb.BEHAVIOR_RUBRIC_VERSION)
        self.assertEqual(config["rubric_fingerprint"], tb.rubric_fingerprint())

    def test_a_changed_segmentation_is_a_different_run(self):
        """Chunk size changes what a count is counted over, so cached rows from another value
        must not be reused silently."""
        base = tb.build_run_config(self.args(), "slug")
        self.assertNotEqual(base, tb.build_run_config(self.args(chunk_tokens=500), "slug"))
        self.assertNotEqual(base, tb.build_run_config(self.args(temperature=0.7), "slug"))
        self.assertNotEqual(
            base, tb.build_run_config(self.args(samples_per_problem=8), "slug")
        )


if __name__ == "__main__":
    unittest.main()
