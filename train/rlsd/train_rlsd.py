"""RLSD on math: answer PI, default hints, or cached trained-generator hints.

The GRPO objective is inherited unchanged. Detached evidence weights are computed
from current-student and periodically refreshed teacher log probabilities on the
same sampled tokens. See docs/rlsd.md for commands and experimental conventions.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import time
from pathlib import Path

import torch
from accelerate.utils import is_peft_model
from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer
from trl.rewards import accuracy_reward
from trl.trainer.utils import pad

from train.opsd.train_sdft import build_sdft_dataset, resolve_hint_source
from utils import DATASET_REGISTRY_TRAIN, compose_pi_messages, validate_resume
from utils.model_scoring import per_token_logps

TEACHER_FILE = "rlsd_teacher.pt"
VERSION = "rlsd_v1"


def mixing_coefficient(step: int, initial: float, decay_steps: int) -> float:
    """step counts completed optimizer updates; zero decay means constant mixing."""
    return initial if decay_steps == 0 else initial * max(0.0, 1.0 - step / decay_steps)


@torch.no_grad()
def reweight_advantages(
    advantages, teacher_logps, student_logps, mask, mixing, clip_range
):
    if teacher_logps.shape != student_logps.shape or teacher_logps.shape != mask.shape:
        raise ValueError("Teacher/student log probabilities and mask must align")
    if advantages.ndim != 1 or advantages.shape[0] != mask.shape[0]:
        raise ValueError("RLSD requires one GRPO advantage per completion")
    valid = mask.bool()
    if (
        not torch.isfinite(teacher_logps[valid]).all()
        or not torch.isfinite(student_logps[valid]).all()
    ):
        raise ValueError("Non-finite RLSD log probabilities on valid tokens")
    if not torch.isfinite(advantages).all():
        raise ValueError("Non-finite GRPO advantage")
    delta = torch.where(valid, teacher_logps.float() - student_logps.float(), 0.0)
    signed_delta = advantages.sign().unsqueeze(1) * delta
    lo, hi = math.log1p(-clip_range), math.log1p(clip_range)
    weights = signed_delta.clamp(lo, hi).exp()
    multiplier = (1 - mixing) + mixing * weights
    result = advantages.float().unsqueeze(1) * multiplier * mask
    diagnostics = {
        "delta": delta,
        "multiplier": multiplier,
        "clip_low": (signed_delta < lo).float(),
        "clip_high": (signed_delta > hi).float(),
    }
    return result, diagnostics


def build_dataset(args, tokenizer):
    pi_mode = "hint" if args.pi_mode == "hint_trained" else args.pi_mode
    ds = build_sdft_dataset(
        pi_mode,
        dataset=args.dataset,
        max_samples=args.max_samples,
        model=args.model,
        hint_generator_model=args.hint_generator_model,
        hint_cache=args.hint_cache,
        include_reward_solution=True,
    )

    def fits(row):
        messages = compose_pi_messages(row["prompt"], row["privileged_context"])
        ids = tokenizer.apply_chat_template(
            [messages], tokenize=True, return_dict=True, add_generation_prompt=True
        )["input_ids"][0]
        return len(ids) <= args.max_prompt_length

    before = len(ds)
    ds = ds.filter(fits)
    if not len(ds):
        raise ValueError(
            "No RLSD examples remain after teacher prompt length filtering"
        )
    print(f"RLSD teacher prompt filter: {len(ds)}/{before} rows retained")
    return ds


def dataset_digest(dataset):
    digest = hashlib.sha256()
    for row in dataset:
        digest.update(json.dumps(row, sort_keys=True, ensure_ascii=False).encode())
        digest.update(b"\n")
    return digest.hexdigest()


class RLSDTrainer(GRPOTrainer):
    def __init__(
        self,
        *args,
        rlsd_lambda=0.5,
        rlsd_decay_steps=50,
        rlsd_clip_range=0.2,
        teacher_sync_steps=10,
        max_teacher_prompt_length=8192,
        **kwargs,
    ):
        if not 0 <= rlsd_lambda <= 1 or rlsd_decay_steps < 0:
            raise ValueError("Require lambda in [0, 1] and decay_steps >= 0")
        if not 0 <= rlsd_clip_range < 1 or teacher_sync_steps < 1:
            raise ValueError("Require clip_range in [0, 1) and teacher_sync_steps >= 1")
        super().__init__(*args, **kwargs)
        if (
            self.is_deepspeed_enabled
            or self.is_fsdp_enabled
            or is_peft_model(self.model)
        ):
            raise ValueError(
                "RLSD currently supports full-model single-GPU/DDP training only"
            )
        if (
            self.args.num_iterations != 1
            or self.args.steps_per_generation != self.args.gradient_accumulation_steps
        ):
            raise ValueError(
                "RLSD resume requires one generation batch per optimizer update"
            )
        if self.temperature != 1.0 or not self.args.disable_dropout:
            raise ValueError(
                "RLSD paired scoring requires temperature=1 and disabled dropout"
            )
        if self.beta != 0 or self.args.use_liger_kernel or self.aux_loss_enabled:
            raise ValueError("RLSD requires beta=0, no Liger, and no auxiliary loss")
        self.rlsd_lambda = rlsd_lambda
        self.rlsd_decay_steps = rlsd_decay_steps
        self.rlsd_clip_range = rlsd_clip_range
        self.teacher_sync_steps = teacher_sync_steps
        self.max_teacher_prompt_length = max_teacher_prompt_length
        self.last_teacher_sync_step = 0
        self.teacher_model = copy.deepcopy(self.accelerator.unwrap_model(self.model))
        self.teacher_model.requires_grad_(False).eval()
        self.teacher_model.gradient_checkpointing_disable()

    @property
    def mixing(self):
        return mixing_coefficient(
            self.state.global_step, self.rlsd_lambda, self.rlsd_decay_steps
        )

    def _generate_and_score_completions(self, inputs):
        output = super()._generate_and_score_completions(inputs)
        if self.mixing == 0:
            return output
        if len(inputs) != output["completion_ids"].size(0):
            raise RuntimeError("PI rows and sampled completions are misaligned")
        messages = [
            compose_pi_messages(row["prompt"], row["privileged_context"])
            for row in inputs
        ]
        ids = self.processing_class.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            add_generation_prompt=True,
            chat_template=self.chat_template,
            **self.chat_template_kwargs,
        )["input_ids"]
        if any(len(row) > self.max_teacher_prompt_length for row in ids):
            raise ValueError("Teacher prompt exceeds cap; refusing to truncate PI")
        device = output["completion_ids"].device
        tensors = [torch.tensor(row, device=device, dtype=torch.long) for row in ids]
        output["teacher_prompt_ids"] = pad(
            tensors, padding_value=self._tokenizer.pad_token_id, padding_side="left"
        )
        output["teacher_prompt_mask"] = pad(
            [torch.ones_like(row) for row in tensors],
            padding_value=0,
            padding_side="left",
        )
        return output

    @torch.no_grad()
    def _evidence_logps(self, model, inputs):
        """Paired unpadded FP32 scoring; no generation-backend logps enter Delta.

        A separate current-student forward keeps the inherited GRPO loss untouched
        and uses the same numerics as the teacher. Both passes precede the actor's
        gradient-bearing forward, avoiding simultaneous actor/teacher activations.
        """
        student = torch.zeros_like(inputs["completion_ids"], dtype=torch.float32)
        teacher = torch.zeros_like(student)
        for i, completion in enumerate(inputs["completion_ids"]):
            active = inputs["completion_mask"][i].bool()
            count = int(active.sum())
            if count == 0:
                continue
            if not active[:count].all() or active[count:].any():
                raise ValueError("RLSD expects right-padded completion tokens")
            completion = completion[:count].unsqueeze(0)
            for scorer, prefix, destination in (
                (model, "prompt", student),
                (self.teacher_model, "teacher_prompt", teacher),
            ):
                prompt = inputs[f"{prefix}_ids"][i][
                    inputs[f"{prefix}_mask"][i].bool()
                ].unsqueeze(0)
                ids = torch.cat([prompt, completion], dim=1)
                destination[i, :count] = per_token_logps(scorer, ids, completion)[0]
        return teacher, student

    def _compute_loss(self, model, inputs):
        mixing = self.mixing
        mode = "train" if self.model.training else "eval"
        self._metrics[mode]["rlsd/lambda"].append(mixing)
        if mixing == 0:
            return super()._compute_loss(model, inputs)
        start = time.monotonic()
        teacher, student = self._evidence_logps(model, inputs)
        mask = inputs["completion_mask"]
        if "tool_mask" in inputs:
            mask = mask * inputs["tool_mask"]
        advantages, stats = reweight_advantages(
            inputs["advantages"], teacher, student, mask, mixing, self.rlsd_clip_range
        )
        self._metrics[mode]["rlsd/scoring_seconds"].append(time.monotonic() - start)
        for label, selection in (
            ("all", mask),
            ("positive", mask * (inputs["advantages"] > 0).unsqueeze(1)),
            ("negative", mask * (inputs["advantages"] < 0).unsqueeze(1)),
        ):
            for name, values in stats.items():
                moments = torch.stack([(values * selection).sum(), selection.sum()])
                moments = self.accelerator.reduce(moments, reduction="sum")
                if moments[1] > 0:
                    self._metrics[mode][f"rlsd/{label}/{name}"].append(
                        (moments[0] / moments[1]).item()
                    )
        # A new dict prevents compounded weighting when a rollout buffer is reused.
        return super()._compute_loss(model, {**inputs, "advantages": advantages})

    @torch.no_grad()
    def _sync_teacher(self):
        self.teacher_model.load_state_dict(
            self.accelerator.unwrap_model(self.model).state_dict()
        )
        self.last_teacher_sync_step = self.state.global_step

    def _maybe_log_save_evaluate(self, *args, **kwargs):
        step = self.state.global_step
        if (
            step > self.last_teacher_sync_step
            and step % self.teacher_sync_steps == 0
            and self.mixing > 0
        ):
            self._sync_teacher()
        return super()._maybe_log_save_evaluate(*args, **kwargs)

    def _save_checkpoint(self, model, trial):
        super()._save_checkpoint(model, trial)
        if self.args.should_save:
            path = (
                Path(self._get_output_dir(trial))
                / f"checkpoint-{self.state.global_step}"
                / TEACHER_FILE
            )
            torch.save(
                {
                    "version": VERSION,
                    "global_step": self.state.global_step,
                    "last_sync_step": self.last_teacher_sync_step,
                    "state_dict": self.teacher_model.state_dict(),
                },
                path,
            )

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        path = Path(resume_from_checkpoint) / TEACHER_FILE
        if not path.is_file():
            raise FileNotFoundError(f"Missing RLSD teacher checkpoint: {path}")
        saved = torch.load(path, map_location="cpu", weights_only=True)
        state = json.loads(
            (Path(resume_from_checkpoint) / "trainer_state.json").read_text()
        )
        if saved["version"] != VERSION or saved["global_step"] != state["global_step"]:
            raise ValueError(
                "RLSD teacher and policy checkpoint versions/steps disagree"
            )
        super()._load_from_checkpoint(resume_from_checkpoint, model)
        self.teacher_model.load_state_dict(saved["state_dict"])
        self.last_teacher_sync_step = saved["last_sync_step"]


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen3-1.7B")
    p.add_argument(
        "--dataset", default="deepmath", choices=list(DATASET_REGISTRY_TRAIN)
    )
    p.add_argument(
        "--pi-mode", choices=["answer", "hint", "hint_trained"], default="answer"
    )
    p.add_argument("--hint-generator-model")
    p.add_argument("--hint-cache")
    p.add_argument("--output-root", default="/mnt/data/ujan/self-distill/outputs/rlsd")
    p.add_argument("--output-dir")
    p.add_argument("--max-samples", type=int)
    p.add_argument("--max-prompt-length", type=int, default=8192)
    p.add_argument("--max-completion-length", type=int, default=8192)
    p.add_argument("--num-generations", type=int, default=8)
    p.add_argument("--loss-type", choices=["dapo", "grpo", "dr_grpo"], default="dapo")
    p.add_argument("--rlsd-lambda", type=float, default=0.5)
    p.add_argument(
        "--rlsd-decay-steps", type=int, default=50, help="0 keeps lambda constant"
    )
    p.add_argument("--rlsd-clip-range", type=float, default=0.2)
    p.add_argument("--teacher-sync-steps", type=int, default=10)
    p.add_argument("--epsilon", type=float, default=0.2)
    p.add_argument("--epsilon-high", type=float, default=0.28)
    p.add_argument("--learning-rate", type=float, default=1e-6)
    p.add_argument("--optim", default="adamw_bnb_8bit")
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--per-device-train-batch-size", type=int, default=1)
    p.add_argument("--gradient-accumulation-steps", type=int, default=16)
    p.add_argument("--use-vllm", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.3)
    p.add_argument("--vllm-max-model-length", type=int)
    p.add_argument("--logging-steps", type=int, default=10)
    p.add_argument("--save-steps", type=int, default=20)
    p.add_argument("--report-to", default="tensorboard")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume-from-checkpoint")
    return p


def validate_args(args):
    if args.pi_mode == "hint_trained" and not args.hint_generator_model:
        raise ValueError(
            "hint_trained requires --hint-generator-model and its precomputed cache"
        )
    if args.pi_mode != "hint_trained" and args.hint_generator_model:
        raise ValueError("--hint-generator-model requires --pi-mode hint_trained")
    if args.pi_mode == "hint_trained" and args.hint_generator_model == args.model:
        raise ValueError(
            "hint_trained requires a different generator identity from the student"
        )
    if args.pi_mode == "answer" and args.hint_cache:
        raise ValueError("--hint-cache requires a hint PI mode")
    if not 0 <= args.rlsd_lambda <= 1 or not 0 <= args.rlsd_clip_range < 1:
        raise ValueError("Require lambda in [0, 1] and clip range in [0, 1)")
    if args.rlsd_decay_steps < 0 or args.teacher_sync_steps < 1:
        raise ValueError("Require decay_steps >= 0 and teacher_sync_steps >= 1")
    if min(args.max_prompt_length, args.max_completion_length) < 1:
        raise ValueError("Prompt and completion budgets must be positive")
    if (
        args.vllm_max_model_length is not None
        and args.vllm_max_model_length
        < args.max_prompt_length + args.max_completion_length
    ):
        raise ValueError(
            "vLLM context must accommodate the prompt and completion budgets"
        )


def main():
    p = build_parser()
    args = p.parse_args()
    try:
        validate_args(args)
    except ValueError as error:
        p.error(str(error))
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    dataset = build_dataset(args, tokenizer)
    generator, cache = (None, None)
    if args.pi_mode != "answer":
        generator, cache = resolve_hint_source(
            args.model, args.dataset, args.hint_generator_model, args.hint_cache
        )
    slug = args.pi_mode
    if generator and args.pi_mode == "hint_trained":
        slug += "_" + hashlib.sha256(generator.encode()).hexdigest()[:10]
    output = args.output_dir or str(
        Path(args.output_root)
        / args.model.rstrip("/").split("/")[-1]
        / f"{args.dataset}_{slug}"
    )
    excluded = {
        "output_root",
        "output_dir",
        "max_steps",
        "logging_steps",
        "save_steps",
        "report_to",
        "resume_from_checkpoint",
    }
    meta = {k: v for k, v in vars(args).items() if k not in excluded}
    meta.update(
        method=VERSION,
        teacher_model_kind="periodic",
        prompt_version="shared_pi_v1",
        hint_generator_model=generator,
        hint_cache=os.path.abspath(cache) if cache else None,
        num_train_examples=len(dataset),
        dataset_sha256=dataset_digest(dataset),
        world_size=int(os.environ.get("WORLD_SIZE", "1")),
        temperature=1.0,
        top_p=1.0,
        top_k=0,
        beta=0.0,
        scale_rewards="group",
        disable_dropout=True,
        lr_scheduler_type="constant",
    )
    if args.resume_from_checkpoint:
        if not (Path(args.resume_from_checkpoint).parent / "run_meta.json").is_file():
            raise ValueError("RLSD resume requires run_meta.json")
        validate_resume(args.resume_from_checkpoint, meta, strict_keys=tuple(meta))
    elif (Path(output) / "run_meta.json").exists():
        raise ValueError(
            "Output already contains a run; use --resume-from-checkpoint or a new directory"
        )
    config = GRPOConfig(
        output_dir=output,
        num_generations=args.num_generations,
        loss_type=args.loss_type,
        max_completion_length=args.max_completion_length,
        temperature=1.0,
        top_p=1.0,
        top_k=0,
        beta=0.0,
        scale_rewards="group",
        disable_dropout=True,
        epsilon=args.epsilon,
        epsilon_high=args.epsilon_high,
        learning_rate=args.learning_rate,
        optim=args.optim,
        lr_scheduler_type="constant",
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        use_vllm=args.use_vllm,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        vllm_max_model_length=args.vllm_max_model_length,
        gradient_checkpointing=True,
        bf16=True,
        model_init_kwargs={"dtype": "bfloat16", "trust_remote_code": True},
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        report_to=args.report_to,
        seed=args.seed,
    )
    trainer = RLSDTrainer(
        model=args.model,
        args=config,
        train_dataset=dataset,
        reward_funcs=accuracy_reward,
        processing_class=tokenizer,
        rlsd_lambda=args.rlsd_lambda,
        rlsd_decay_steps=args.rlsd_decay_steps,
        rlsd_clip_range=args.rlsd_clip_range,
        teacher_sync_steps=args.teacher_sync_steps,
        max_teacher_prompt_length=args.max_prompt_length,
    )
    if config.should_save:
        Path(output).mkdir(parents=True, exist_ok=True)
        (Path(output) / "run_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(str(Path(output) / "final"))


if __name__ == "__main__":
    main()
