"""SFT target formatting: the think-block wrapping and the prompt/completion boundary.

The load-bearing case is `test_unwrapped_trace_without_a_closing_tag_loses_its_think_channel`,
which pins the behavior the drop rule exists for: Qwen3's chat template silently inserts an
EMPTY `<think></think>` in front of assistant content carrying no `</think>`, moving the whole
trace into the answer channel while every metric -- loss, grad norm, even pass@k -- keeps
looking plausible. Content that DOES contain a `</think>` is instead split and emitted verbatim,
which is why the explicit opener is a no-op here and is asserted as such rather than claimed.

These assert against the real tokenizer and TRL's real `apply_chat_template`, since it is those
two libraries' behavior -- not ours -- that the formatting has to satisfy.
"""

import unittest

from tests.helpers import TOKENIZER_ID
from train.sft.train_sft import format_think_completion, to_sft_example
from utils import format_prompt_math


R1_TRACE = "Okay, let me think.\n\nSo the limit is 0.\n</think>\n\nThe answer is \\boxed{0}."


class FormatThinkCompletionTest(unittest.TestCase):
    def test_prepends_opener_to_r1_convention_trace(self):
        # DeepMath's traces: exactly one `</think>`, no opener.
        self.assertEqual(format_think_completion(R1_TRACE), f"<think>\n{R1_TRACE}")

    def test_leaves_an_already_wrapped_trace_alone(self):
        wrapped = f"<think>\n{R1_TRACE}"
        self.assertEqual(format_think_completion(wrapped), wrapped)

    def test_strips_surrounding_whitespace_before_wrapping(self):
        self.assertEqual(format_think_completion(f"  \n{R1_TRACE}\n  "), f"<think>\n{R1_TRACE}")

    def test_drops_a_trace_with_no_closing_tag(self):
        # A non-reasoning worked answer (most of DeepScaleR). Prepending an opener here would
        # train a think block the model never closes.
        self.assertIsNone(format_think_completion("By AM-GM, the minimum is \\boxed{4}."))

    def test_drops_a_trace_with_two_closing_tags(self):
        self.assertIsNone(format_think_completion("a</think>b</think>c"))

    def test_drops_a_trace_whose_opener_is_not_at_the_start(self):
        self.assertIsNone(format_think_completion("preamble <think> reasoning </think> answer"))

    def test_drops_empty_and_whitespace_traces(self):
        for empty in ("", "   \n  ", None):
            with self.subTest(solution=empty):
                self.assertIsNone(format_think_completion(empty))


class SftExampleTest(unittest.TestCase):
    def test_prompt_is_the_shared_eval_prompt(self):
        question = "What is 2 + 2?"
        example = to_sft_example(question, R1_TRACE)

        # Identical to what train_grpo / train_ppo / eval build, so the student is trained to
        # continue the exact tokens it is asked to continue at eval.
        self.assertEqual(example["prompt"], format_prompt_math(question))
        self.assertEqual(
            example["completion"],
            [{"role": "assistant", "content": f"<think>\n{R1_TRACE}"}],
        )

    def test_unusable_trace_yields_no_example(self):
        self.assertIsNone(to_sft_example("What is 2 + 2?", "no think block here"))


class RenderedTargetTest(unittest.TestCase):
    """Cross-library: what the tokenizer + TRL actually turn the example into.

    Tokenizer only -- no model weights are loaded.
    """

    @classmethod
    def setUpClass(cls):
        from transformers import AutoTokenizer

        try:
            cls.tokenizer = AutoTokenizer.from_pretrained(
                TOKENIZER_ID, trust_remote_code=True, local_files_only=True
            )
        except Exception as exc:  # no local cache / no network
            raise unittest.SkipTest(f"tokenizer {TOKENIZER_ID} unavailable: {exc}")

    def render(self, solution):
        from trl.data_utils import apply_chat_template

        example = to_sft_example("What is 2 + 2?", solution)
        return apply_chat_template(
            {"prompt": example["prompt"], "completion": example["completion"]}, self.tokenizer
        )

    def render_assistant(self, content):
        """The rendered assistant turn for raw `content`, bypassing our formatting."""
        full = self.tokenizer.apply_chat_template(
            [format_prompt_math("What is 2 + 2?") + [{"role": "assistant", "content": content}]],
            tokenize=False,
        )[0]
        return full[full.index("<|im_start|>assistant"):]

    def test_rendered_completion_keeps_the_reasoning_in_the_think_channel(self):
        rendered = self.render(R1_TRACE)

        # The reasoning sits INSIDE the think block, with no injected empty one.
        self.assertIn("<think>\nOkay, let me think.", rendered["completion"])
        self.assertNotIn("<think>\n\n</think>", rendered["completion"])
        self.assertEqual(rendered["completion"].count("</think>"), 1)

    def test_prepending_the_opener_is_a_no_op_under_this_template(self):
        # Qwen3's template splits assistant content on `</think>` and supplies the opener, so
        # our prepend changes nothing HERE. Pinned so the docstring's claim stays honest: if a
        # template stops splitting that way, this fails and the prepend starts doing real work.
        self.assertEqual(
            self.render_assistant(R1_TRACE),
            self.render_assistant(f"<think>\n{R1_TRACE}"),
        )

    def test_unwrapped_trace_without_a_closing_tag_loses_its_think_channel(self):
        # The corruption the drop rule exists for: no `</think>` anywhere in the content, so
        # the template injects an empty think block and the solution lands in the answer
        # channel. format_think_completion drops these rather than wrapping them.
        no_think = "By AM-GM, the minimum is \\boxed{4}."
        self.assertIn("<think>\n\n</think>", self.render_assistant(no_think))
        self.assertIsNone(format_think_completion(no_think))

    def test_prompt_and_completion_tokenize_independently_at_the_seam(self):
        """The invariant that puts the -100 boundary in the right place.

        SFTTrainer tokenizes the prompt alone and the combined prompt-plus-completion, then
        uses the rendered prompt length to construct the completion mask. This test makes
        the stronger seam assertion that separately tokenizing the two halves gives exactly
        the combined tokenization. If a token spanned the seam, the first supervised position
        could be the tail of the generation header rather than the model's first generated
        token.

        It holds because the seam falls on the `<|im_start|>assistant\\n` special-token
        boundary, but that is a property of the chat template, not something we control, so
        it is asserted rather than assumed. Verified against the real trainer separately:
        supervision starts at exactly the rendered prompt length, with `<|im_start|>assistant`
        masked and `<think>` the first supervised token.
        """
        rendered = self.render(R1_TRACE)

        prompt_ids = self.tokenizer(rendered["prompt"], add_special_tokens=False)["input_ids"]
        completion_ids = self.tokenizer(
            rendered["completion"], add_special_tokens=False
        )["input_ids"]
        joint_ids = self.tokenizer(
            rendered["prompt"] + rendered["completion"], add_special_tokens=False
        )["input_ids"]

        self.assertEqual(prompt_ids + completion_ids, joint_ids)
        # And the token supervision starts on is the completion's first, not the header's last.
        self.assertEqual(
            self.tokenizer.decode(joint_ids[len(prompt_ids):len(prompt_ids) + 2]),
            "<think>\n",
        )

    def test_prompt_ends_on_the_generation_header_and_completion_ends_on_eos(self):
        rendered = self.render(R1_TRACE)

        # Same boundary the RL arms sample from: add_generation_prompt=True on the prompt half.
        expected_prompt = self.tokenizer.apply_chat_template(
            [format_prompt_math("What is 2 + 2?")], add_generation_prompt=True, tokenize=False
        )[0]
        self.assertEqual(rendered["prompt"], expected_prompt)

        # TRL appends no EOS for conversational data (sft_trainer.py guards `add_eos` with
        # `not is_conversational`), so the chat template's turn terminator is the only thing
        # teaching the model to stop.
        self.assertIn(self.tokenizer.eos_token, rendered["completion"])


if __name__ == "__main__":
    unittest.main()
