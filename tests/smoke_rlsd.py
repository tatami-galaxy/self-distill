"""GPU integration smoke using real cached PI and synthetic alternating rewards.

The artificial verifier guarantees both positive and negative advantages even
with short, usually incorrect math responses. This is not a performance eval.
Run each PI in a fresh process; optionally resume checkpoint-1 into a new output.
"""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer
from trl import GRPOConfig

from train.opsd.train_sdft import build_hint_dataset
from train.rlsd.train_rlsd import TEACHER_FILE, RLSDTrainer
from utils import answer_context

GENERATOR = "/mnt/data/ujan/self-distill/outputs/hint_gen/Qwen3-1.7B/deepmath_t0.7_g6_lora_r16/checkpoint-80"
CACHE = "data/pi/hint/deepmath/Qwen3-1.7B-t0.7_g6_lora_r16-checkpoint-80"


def smoke_reward(completions, **kwargs):
    return [float(i % 2) for i in range(len(completions))]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pi-mode", choices=["answer", "hint", "hint_trained"], required=True
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume-from-checkpoint")
    parser.add_argument(
        "--use-vllm", action=argparse.BooleanOptionalAction, default=True
    )
    args = parser.parse_args()
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Choose an empty smoke output directory")
    model = "Qwen/Qwen3-1.7B"
    dataset = build_hint_dataset(
        model,
        "deepmath",
        8,
        hint_generator_model=GENERATOR if args.pi_mode == "hint_trained" else None,
        hint_cache=CACHE if args.pi_mode == "hint_trained" else None,
        include_reward_solution=True,
    )
    if args.pi_mode == "answer":
        # reward targets here are boxed; use the original cached gold string.
        from datasets import load_from_disk

        source = load_from_disk("data/pi/hint/deepmath/Qwen3-1.7B")
        dataset = dataset.map(
            lambda row, i: {
                "privileged_context": answer_context(
                    str(source[i]["final_answer"]), "deepmath"
                )
            },
            with_indices=True,
        )
    config = GRPOConfig(
        output_dir=str(output),
        max_steps=4,
        save_steps=1,
        logging_steps=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        num_generations=2,
        max_completion_length=64,
        temperature=1.0,
        top_p=1.0,
        top_k=0,
        beta=0.0,
        scale_rewards="group",
        disable_dropout=True,
        loss_type="dapo",
        epsilon=0.2,
        epsilon_high=0.28,
        learning_rate=1e-6,
        lr_scheduler_type="constant",
        optim="adamw_bnb_8bit",
        gradient_checkpointing=True,
        bf16=True,
        report_to="none",
        seed=42,
        use_vllm=args.use_vllm,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=0.15,
        vllm_max_model_length=4096,
        model_init_kwargs={"dtype": "bfloat16", "trust_remote_code": True},
    )
    trainer = RLSDTrainer(
        model=model,
        args=config,
        train_dataset=dataset,
        reward_funcs=smoke_reward,
        processing_class=AutoTokenizer.from_pretrained(model),
        teacher_sync_steps=2,
        rlsd_decay_steps=3,
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    assert trainer.state.global_step == 4
    assert trainer.last_teacher_sync_step == 2
    assert all(p.grad is None for p in trainer.teacher_model.parameters())
    logs = trainer.state.log_history
    active = [row for row in logs if row.get("rlsd/lambda", 0) > 0]
    assert active and any(row.get("grad_norm", 0) > 0 for row in logs)
    assert any(row.get("rlsd/lambda") == 0 for row in logs)
    for row in active:
        assert 0.899 <= row["rlsd/all/multiplier"] <= 1.101
    trainer.save_model(str(output / "final"))
    saved = torch.load(
        output / "checkpoint-4" / TEACHER_FILE, map_location="cpu", weights_only=True
    )
    assert saved["last_sync_step"] == 2 and saved["global_step"] == 4
    (output / "smoke_summary.json").write_text(
        json.dumps(
            {
                "pi_mode": args.pi_mode,
                "use_vllm": args.use_vllm,
                "reward": "synthetic_alternating_correctness",
                "steps": 4,
                "last_teacher_sync_step": 2,
                "resumed_from": args.resume_from_checkpoint,
                "active_weighting_checked": True,
                "zero_mixing_checked": True,
                "teacher_gradients_absent": True,
            },
            indent=2,
        )
        + "\n"
    )
    print("RLSD SMOKE PASSED", output, flush=True)


if __name__ == "__main__":
    main()
