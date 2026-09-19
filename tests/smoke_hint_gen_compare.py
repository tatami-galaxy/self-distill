"""Real two-GPU evaluator smoke; artifacts stay in the specified output directory.

Uses a synthetic two-question cohort and three existing LoRA checkpoints. Exercises
one shared generator engine, concurrent scoring, checkpoint extension, and a fully
cached rerun. This is not collected by pytest.
"""

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from datasets import Dataset, load_from_disk
from transformers import AutoTokenizer

from eval import hint_gen_compare as compare
from utils import rollout_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--steps', nargs=3, type=int, required=True)
    parser.add_argument('--gpus', nargs=2, required=True)
    parser.add_argument('--output-dir', required=True)
    options = parser.parse_args()
    root = Path(options.output_dir).resolve()
    root.mkdir(parents=True, exist_ok=False)
    output = root / 'evaluation'
    base_args = [
        '--run-dir', options.run_dir, '--output-dir', str(output), '--gpus', *options.gpus,
        '--rollout-root', str(root / 'rollouts'), '--hint-root', str(root / 'unused_hint_root'),
        '--num-problems', '2', '--hints-per-problem', '2', '--hint-max-tokens', '64',
        '--teacher-rollouts', '1', '--transfer-rollouts', '1', '--k', '1',
        '--teacher-max-tokens', '1024', '--teacher-condition-batch-size', '4',
        '--max-model-len', '2048', '--gpu-memory-utilization', '0.25',
        '--bootstrap-samples', '20', '--save-teacher-samples',
    ]
    args = compare.build_parser().parse_args(base_args + ['--steps', *map(str, options.steps[:2])])
    compare.validate_args(args, compare.build_parser())
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    cohort, rollouts = [], []
    for index, (question, answer) in enumerate([('What is 2 + 2?', '4'), ('What is 3 multiplied by 5?', '15')]):
        cohort.append(dict(
            question_id=compare.stable_question_id(index, question, answer),
            question_idx=index, question=question, final_answer=answer,
            solution=f'Compute the arithmetic to obtain {answer}.', difficulty='easy',
        ))
        rollouts.append(dict(
            question=question, sample_idx=0, rollout_id=str(index),
            completion_ids=tokenizer.encode(r'\boxed{' + answer + '}', add_special_tokens=False),
        ))
    cohort = Dataset.from_list(cohort)
    cohort_path, cohort_meta = compare.cohort_paths(args)
    compare.save_dataset_atomic(cohort, cohort_path)
    compare.write_json_atomic(cohort_meta, dict(
        status='complete', method=compare.METHOD, config=compare.prepare_config(args),
        cohort_fingerprint=compare.cohort_fingerprint(cohort),
    ))
    Dataset.from_list(rollouts).save_to_disk(rollout_path(args.model, args.dataset, args.rollout_root))
    report = dict(verified=False, gpus=options.gpus, model=args.model, runs=[])
    old_suff, old_transfer = {}, {}
    for name, steps in [('initial', options.steps[:2]), ('extend', options.steps), ('cached', options.steps)]:
        command = [sys.executable, '-u', '-m', 'eval.hint_gen_compare', *base_args, '--steps', *map(str, steps)]
        started = time.monotonic()
        with (root / f'{name}.log').open('w') as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        suff_meta = compare.read_json(output / 'sufficiency_meta.json')
        transfer_meta = compare.read_json(output / 'transfer_meta.json')
        suff = {row['condition_id']: row for row in load_from_disk(str(output / 'sufficiency'))}
        transfer = {(row['hint_id'], row['rollout_position']): row for row in load_from_disk(str(output / 'transfer'))}
        assert all(suff[key] == value for key, value in old_suff.items())
        assert all(transfer[key] == value for key, value in old_transfer.items())
        assert all(math.isfinite(row['raw_transfer']) for row in transfer.values())
        log = (root / f'{name}.log').read_text()
        if name == 'initial':
            assert log.count('Shared generator engine loaded') == 1
            assert suff_meta['condition_cache'] == {'hits': 0, 'misses': 14}
            assert transfer_meta['condition_cache'] == {'hits': 0, 'misses': 12}
        elif name == 'extend':
            assert suff_meta['condition_cache'] == {'hits': 14, 'misses': 4}
            assert transfer_meta['condition_cache'] == {'hits': 12, 'misses': 4}
            assert transfer_meta['student_cache'] == {'hits': 2, 'misses': 0}
        else:
            assert 'Shared generator engine loaded' not in log
            assert suff_meta['condition_cache'] == {'hits': 18, 'misses': 0}
            assert transfer_meta['condition_cache'] == {'hits': 16, 'misses': 0}
        if name != 'cached':
            for phase, gpu in [('sufficiency', options.gpus[0]), ('transfer', options.gpus[1])]:
                assert any(f'Starting {phase}:' in line and f'CUDA_VISIBLE_DEVICES={gpu}' in line for line in log.splitlines())
            assert log.index('Starting transfer:') < log.index('Finished sufficiency')
        report['runs'].append(dict(
            name=name, seconds=time.monotonic() - started,
            sufficiency_cache=suff_meta['condition_cache'], transfer_cache=transfer_meta['condition_cache'],
            student_cache=transfer_meta['student_cache'],
        ))
        old_suff, old_transfer = suff, transfer
        compare.write_json_atomic(root / 'report.json', report)
        print(json.dumps(report['runs'][-1]), flush=True)
    report['verified'] = True
    compare.write_json_atomic(root / 'report.json', report)


if __name__ == '__main__':
    main()
