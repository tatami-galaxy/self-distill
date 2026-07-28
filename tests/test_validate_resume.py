import json
import os
import tempfile
import types
import unittest

from train.ppo.train_ppo import build_run_meta as ppo_meta
from train.ppo.train_ppo_val import build_run_meta as ppo_val_meta
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


class CrossArmResumeTest(unittest.TestCase):
    """A critic cannot be resumed into an arm that feeds it a different prompt.

    The two arms' critics read different state representations, so `value_model.pt` and its
    optimizer state are not interchangeable -- and neither direction may fail quietly. Built
    from the REAL `build_run_meta`s rather than hand-written dicts, because the guard is only
    as good as what those functions actually stamp.
    """

    def make_args(self):
        return types.SimpleNamespace(
            model="Qwen/Qwen3-1.7B", dataset="deepmath", max_samples=None,
            gamma=1.0, lam=1.0, vf_coef=0.1, cliprange_value=0.2, critic_max_grad_norm=1.0,
            loss_type="dapo", vllm_mode="colocate", optim="paged_adamw_8bit",
            learning_rate=1e-6, critic_learning_rate=None, seed=42,
            per_device_train_batch_size=1, gradient_accumulation_steps=16, num_generations=1,
        )

    def make_checkpoint(self, tmp, prior_meta):
        with open(os.path.join(tmp, "run_meta.json"), "w") as f:
            json.dump(prior_meta, f)
        ckpt = os.path.join(tmp, "checkpoint-20")
        os.makedirs(ckpt)
        return ckpt

    def test_verifier_checkpoint_refused_by_the_baseline_trainer(self):
        # The direction `strict_keys` CANNOT catch: validate_resume only iterates the keys of
        # the meta it is handed, so the baseline has to stamp the key itself (as None) for the
        # comparison to fire at all. Without that, a verifier-trained critic would resume into
        # the baseline silently and be fed policy prompts.
        args = self.make_args()
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = self.make_checkpoint(tmp, ppo_val_meta(args, 100))
            with self.assertRaisesRegex(ValueError, "value_prompt_version"):
                validate_resume(ckpt, ppo_meta(args, 100))

    def test_baseline_checkpoint_refused_by_the_verifier_trainer(self):
        args = self.make_args()
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = self.make_checkpoint(tmp, ppo_meta(args, 100))
            with self.assertRaisesRegex(ValueError, "value_prompt_version"):
                validate_resume(
                    ckpt, ppo_val_meta(args, 100), strict_keys=("value_prompt_version",)
                )

    def test_legacy_baseline_checkpoint_still_resumes_into_the_baseline(self):
        # Checkpoints written before `value_prompt_version` existed have no such key. The
        # baseline stamps None, and an ABSENT key is skipped when it is not strict -- so
        # adding the key must not invalidate the runs already on disk.
        args = self.make_args()
        legacy = ppo_meta(args, 100)
        del legacy["value_prompt_version"]
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = self.make_checkpoint(tmp, legacy)
            validate_resume(ckpt, ppo_meta(args, 100))

    def test_each_arm_resumes_itself(self):
        args = self.make_args()
        for meta_fn, strict in ((ppo_meta, ()), (ppo_val_meta, ("value_prompt_version",))):
            with self.subTest(meta_fn.__module__):
                with tempfile.TemporaryDirectory() as tmp:
                    ckpt = self.make_checkpoint(tmp, meta_fn(args, 100))
                    validate_resume(ckpt, meta_fn(args, 100), strict_keys=strict)

    def test_the_arms_are_distinguishable_by_method_too(self):
        args = self.make_args()
        self.assertNotEqual(ppo_meta(args, 100)["method"], ppo_val_meta(args, 100)["method"])


if __name__ == "__main__":
    unittest.main()
