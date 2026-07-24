import json
import os
import tempfile
import unittest

from utils import validate_resume


class ValidateResumeStrictKeysTest(unittest.TestCase):
    """`strict_keys` makes a key's ABSENCE from the prior meta disqualifying.

    Ordinary keys are skipped when the checkpoint predates them -- that is what lets a new
    key be added to a build_run_meta without invalidating every existing checkpoint. Keys
    that change what the saved tensors MEAN (train_ppo's `value_prompt_version`) need the
    opposite behaviour, or a new key could never refuse an old checkpoint on its own.
    """

    def make_checkpoint(self, tmp, prior_meta):
        """Lay out <run>/run_meta.json + <run>/checkpoint-20, returning the checkpoint dir."""
        with open(os.path.join(tmp, "run_meta.json"), "w") as f:
            json.dump(prior_meta, f)
        ckpt = os.path.join(tmp, "checkpoint-20")
        os.makedirs(ckpt)
        return ckpt

    def test_absent_strict_key_is_a_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = self.make_checkpoint(tmp, {"method": "ppo_vllm", "seed": 42})
            with self.assertRaisesRegex(ValueError, "value_prompt_version"):
                validate_resume(
                    ckpt,
                    {"method": "ppo_vllm", "seed": 42, "value_prompt_version": "verifier_v1"},
                    strict_keys=("value_prompt_version",),
                )

    def test_absent_key_is_skipped_when_not_strict(self):
        # The pre-existing contract: adding a key must not break old resumes.
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = self.make_checkpoint(tmp, {"method": "ppo_vllm", "seed": 42})
            validate_resume(ckpt, {"method": "ppo_vllm", "seed": 42, "optim": "adafactor"})

    def test_matching_strict_key_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            meta = {"method": "ppo_vllm", "seed": 42, "value_prompt_version": "verifier_v1"}
            ckpt = self.make_checkpoint(tmp, meta)
            validate_resume(ckpt, dict(meta), strict_keys=("value_prompt_version",))

    def test_differing_strict_key_is_a_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = self.make_checkpoint(
                tmp, {"seed": 42, "value_prompt_version": "verifier_v1"}
            )
            with self.assertRaisesRegex(ValueError, "value_prompt_version"):
                validate_resume(
                    ckpt,
                    {"seed": 42, "value_prompt_version": "verifier_v2"},
                    strict_keys=("value_prompt_version",),
                )

    def test_force_downgrades_strict_mismatch_to_a_warning(self):
        # `force` is the documented escape hatch everywhere else; making one key
        # un-forceable would be a surprise.
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = self.make_checkpoint(tmp, {"seed": 42})
            validate_resume(
                ckpt,
                {"seed": 42, "value_prompt_version": "verifier_v1"},
                force=True,
                strict_keys=("value_prompt_version",),
            )

    def test_ordinary_mismatch_still_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = self.make_checkpoint(tmp, {"seed": 42})
            with self.assertRaisesRegex(ValueError, "seed"):
                validate_resume(ckpt, {"seed": 7})


if __name__ == "__main__":
    unittest.main()
