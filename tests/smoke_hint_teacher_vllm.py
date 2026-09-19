"""Two-GPU integration smoke (not collected by pytest).

CUDA_VISIBLE_DEVICES=4 VLLM_USE_V2_MODEL_RUNNER=0 HF_HUB_OFFLINE=1 \
    .venv/bin/python -m tests.smoke_hint_teacher_vllm --teacher-gpu 5 --output-dir /tmp/hint-teacher-smoke
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoTokenizer
from trl import GRPOConfig
import trl.generation.vllm_generation as generation_module

from train.opsd.train_hint_gen.lib import (
    ConstrainedHintRewardConfig, make_constrained_reward_function,
)
from train.opsd.train_hint_gen.train_constrained_hint_gen import ConstrainedHintGRPOTrainer
from utils import grade, rollout_path
from utils.gen_hints import build_messages


def digest_trainable(model):
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            digest.update(name.encode())
            digest.update(parameter.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--teacher-gpu', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--model', default='Qwen/Qwen3-1.7B')
    parser.add_argument('--teacher-enforce-eager', action='store_true')
    args = parser.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    report = {'versions': {p: importlib.metadata.version(p) for p in ['trl', 'vllm', 'transformers', 'torch', 'peft']},
              'training_gpu': os.environ.get('CUDA_VISIBLE_DEVICES'), 'teacher_gpu': args.teacher_gpu,
              'model': args.model, 'policy_enforce_eager': True, 'teacher_enforce_eager': args.teacher_enforce_eager, 'runner_v2': os.getenv('VLLM_USE_V2_MODEL_RUNNER'),
              'verified': False}
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    questions = ['What is 2 + 2?', 'What is 3 multiplied by 5?']
    answers = ['4', '15']
    hints = ['Add the two integers.', 'Use repeated addition.']
    rows = []
    for i, (question, answer) in enumerate(zip(questions, answers, strict=True)):
        completion = r'\boxed{' + answer + '}'
        rows.append({'question': question, 'completion_ids': tokenizer.encode(completion, add_special_tokens=False),
                     'gen_model': args.model, 'dataset': 'deepmath', 'sample_idx': 0, 'rollout_id': str(i)})
    cache_root = str(out / 'rollouts')
    cache = rollout_path(args.model, 'deepmath', cache_root)
    Dataset.from_list(rows).save_to_disk(cache)
    config = ConstrainedHintRewardConfig(
        model=args.model, dataset='deepmath', rollout_root=cache_root,
        hint_budget=48, teacher_rollouts=2, transfer_rollouts=1,
        teacher_max_completion_length=1024, teacher_backend='vllm', teacher_gpu=args.teacher_gpu,
        teacher_max_model_length=2048, teacher_max_num_seqs=8,
        teacher_gpu_memory_utilization=0.2, teacher_enforce_eager=args.teacher_enforce_eager,
    )
    reward_func, reward = make_constrained_reward_function(config, tokenizer)
    backend = reward.backend
    llm_class = generation_module.LLM
    def eager_llm(**kwargs):
        kwargs['enforce_eager'] = True
        return llm_class(**kwargs)
    generation_module.LLM = eager_llm
    try:
        trainer = ConstrainedHintGRPOTrainer(
            model=args.model, processing_class=tokenizer,
            train_dataset=Dataset.from_list([
                {'prompt': build_messages(q, f'The worked solution gives {a}.'), 'question': q, 'final_answer': a}
                for q, a in zip(questions, answers, strict=True)
            ]), reward_funcs=reward_func, constrained_reward=reward,
            peft_config=LoraConfig(r=8, lora_alpha=16, target_modules='all-linear', task_type='CAUSAL_LM'),
            args=GRPOConfig(
                output_dir=str(out / 'training'), num_generations=2, generation_batch_size=2,
                per_device_train_batch_size=1, gradient_accumulation_steps=2, max_steps=2,
                max_completion_length=48, temperature=1.0, top_p=1.0,
                chat_template_kwargs={'enable_thinking': False}, loss_type='dapo', beta=0.0,
                use_vllm=True, vllm_mode='colocate', vllm_enable_sleep_mode=True,
                vllm_gpu_memory_utilization=0.2, vllm_max_model_length=2048,
                learning_rate=1e-4, optim='adamw_torch', bf16=True, gradient_checkpointing=True,
                model_init_kwargs={'dtype': 'bfloat16', 'local_files_only': True},
                logging_steps=1, save_steps=1, report_to='none', remove_unused_columns=False,
                seed=42, log_completions=False,
            ),
        )
        client = backend._teacher_client()
        report['worker'] = client.info
        teacher_weights_before = client.fingerprint()
        prompts = [backend._render_prompt(backend._hinted_messages(q, h)) for q, h in zip(questions, hints)]
        client.submit(prompts, seed=123)
        first = client.result()
        assert len(first['samples']) == 2 and all(len(s) == 2 for s in first['samples'])
        report['teacher_samples'] = first
        report['teacher_successes'] = [
            sum(bool(grade(s['text'], answer, 'deepmath')[1]) for s in samples) / len(samples)
            for samples, answer in zip(first['samples'], answers)
        ]
        assert any(score > 0 for score in report['teacher_successes']), 'No completed correct teacher answers'
        # Exercise batched generation + concurrent real HF transfer scoring.
        pairs = backend.score_hints(questions, answers, hints)
        assert all(0 <= success <= 1 and torch.isfinite(torch.tensor(transfer)) for success, transfer in pairs)
        report['preflight_scores'] = pairs
        report['preflight_metrics'] = backend.last_metrics.copy()
        before = digest_trainable(trainer.model)
        trainer.train()
        after = digest_trainable(trainer.model)
        assert trainer.state.global_step == 2
        assert before != after, 'The optimizer did not update the policy adapter'
        assert reward.dual_updates == 2
        report['trainable_parameters_changed'] = before != after
        report['dual_state'] = reward.state_dict()
        report['training_log'] = trainer.state.log_history
        for step in [1, 2]:
            state = json.loads((out / 'training' / f'checkpoint-{step}' / 'constrained_reward_state.json').read_text())
            assert state['dual_updates'] == step
        # Verify actual teacher weights; stochastic outputs need not be bitwise
        # identical across calls with different scheduling/prefix-cache state.
        client.submit(prompts, seed=123)
        last = client.result()
        teacher_weights_after = client.fingerprint()
        assert teacher_weights_before == teacher_weights_after, 'Frozen teacher weights changed'
        report['teacher_weight_fingerprint'] = teacher_weights_after
        report['teacher_repeat_samples_identical'] = first['samples'] == last['samples']
        report['teacher_repeat_samples'] = last
        report['teacher_unchanged_after_training'] = True
        report['verified'] = True
    finally:
        reward_func.close()
        report['worker_closed'] = backend._vllm_teacher is None
        (out / 'report.json').write_text(json.dumps(report, indent=2))
    print('SMOKE VERIFIED ' + str(out / 'report.json'), flush=True)


if __name__ == '__main__':
    main()
