"""Incremental evaluation, shared LoRA engines, and isolated worker execution."""

import json
import multiprocessing
import os
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from datasets import Dataset

from eval import hint_gen_compare as compare
from eval.hint_compare_cache import ConditionCache


@pytest.fixture
def scoring_run(tmp_path, monkeypatch):
    args = compare.build_parser().parse_args([
        '--model', 'base', '--output-dir', str(tmp_path), '--teacher-rollouts', '1',
        '--transfer-rollouts', '1', '--k', '1', '--teacher-condition-batch-size', '1',
    ])
    compare.resolve_run_configuration(args)
    cohort = Dataset.from_list([
        dict(question_id='q1', question_idx=0, question='one', final_answer='2', solution='worked'),
        dict(question_id='q2', question_idx=1, question='two', final_answer='3', solution='worked'),
    ])
    hints = [dict(
        hint_id='h1', generator_id='fresh_base', question_id='q1', hint_sample_idx=0,
        hint='Use addition.', hint_token_ids=[1, 2], invalid_reason='',
    )]
    rollouts = [dict(question='one', sample_idx=0, completion_ids=[4, 5]),
                dict(question='two', sample_idx=0, completion_ids=[6, 7])]
    monkeypatch.setattr(compare, 'load_cohort', lambda _: (cohort, {'cohort_fingerprint': 'cohort'}))
    monkeypatch.setattr(compare, 'load_all_hints', lambda _: (hints, {}))
    monkeypatch.setattr(compare, 'load_from_disk', lambda _: Dataset.from_list(rollouts))
    return SimpleNamespace(args=args, cohort=cohort, hints=hints, rollouts=rollouts, root=tmp_path)


class Tokenizer:
    def apply_chat_template(self, messages, *, tokenize=True, return_dict=False, **kwargs):
        if return_dict:
            return {'input_ids': [[1, 2]]}
        return [1, 2] if tokenize else json.dumps(messages)


class Teacher:
    instances = []
    fail_at = None

    def __init__(self, **kwargs):
        self.calls = []
        self.instances.append(self)

    def get_tokenizer(self):
        return Tokenizer()

    def generate(self, prompts, sampling):
        self.calls.append(prompts)
        if self.fail_at == len(self.calls):
            raise RuntimeError('interrupted teacher')
        return [SimpleNamespace(outputs=[SimpleNamespace(text='answer', finish_reason='stop')]) for _ in prompts]


def test_sufficiency_resumes_then_extends_without_rescoring(scoring_run, monkeypatch):
    import vllm
    run = scoring_run
    Teacher.instances = []
    Teacher.fail_at = 2
    monkeypatch.setattr(vllm, 'LLM', Teacher)
    monkeypatch.setattr(compare, 'grade', lambda *args: (None, True))
    with pytest.raises(RuntimeError, match='interrupted'):
        compare.sufficiency_phase(run.args)
    assert len(list((run.root / 'score_cache/sufficiency').glob('*.json'))) == 1
    Teacher.fail_at = None
    compare.sufficiency_phase(run.args)
    assert len(Teacher.instances[-1].calls) == 2  # second no-hint condition + h1
    before = len(Teacher.instances)
    compare.sufficiency_phase(run.args)
    assert len(Teacher.instances) == before  # all cache hits: no model startup
    run.hints.append(dict(run.hints[0], hint_id='h2', generator_id='checkpoint-10', hint='Use symmetry.'))
    compare.sufficiency_phase(run.args)
    assert len(Teacher.instances[-1].calls) == 1
    meta = compare.read_json(run.root / 'sufficiency_meta.json')
    assert meta['condition_cache'] == {'hits': 3, 'misses': 1}
    # Changed hint text under the SAME ID cannot reuse the old score.
    run.hints[-1]['hint'] = 'Use parity.'
    compare.sufficiency_phase(run.args)
    assert len(Teacher.instances[-1].calls) == 1
    # Grading target changes invalidate that question, including the no-hint control.
    cohort = run.cohort.to_list()
    cohort[0]['final_answer'] = '9'
    monkeypatch.setattr(compare, 'load_cohort', lambda _: (Dataset.from_list(cohort), {'cohort_fingerprint': 'new'}))
    compare.sufficiency_phase(run.args)
    assert len(Teacher.instances[-1].calls) == 3
    run.args.teacher_top_k = 10
    compare.sufficiency_phase(run.args)
    assert len(Teacher.instances[-1].calls) == 4


def test_transfer_reuses_student_scores_and_checks_rollout_tokens(scoring_run, monkeypatch):
    import transformers
    run = scoring_run
    model_loads, forwards = [], []

    class Model:
        def eval(self):
            return self

        def to(self, device):
            return self

    def load_model(*args, **kwargs):
        model_loads.append(1)
        return Model()

    def score(model, prompt, completion):
        forwards.append((prompt, completion))
        return torch.full((len(completion),), -float(len(prompt)))

    monkeypatch.setattr(transformers.AutoModelForCausalLM, 'from_pretrained', load_model)
    monkeypatch.setattr(transformers.AutoTokenizer, 'from_pretrained', lambda *a, **kw: Tokenizer())
    monkeypatch.setattr(compare, '_render_prompt_ids', lambda t, m: [1] * len(m))
    monkeypatch.setattr(compare, '_score_completion', score)
    compare.transfer_phase(run.args)
    assert len(forwards) == 2  # one base forward + one hinted forward
    compare.transfer_phase(run.args)
    assert len(forwards) == 2 and len(model_loads) == 1
    run.hints.append(dict(run.hints[0], hint_id='h2', generator_id='checkpoint-10', hint='Use symmetry.'))
    compare.transfer_phase(run.args)
    assert len(forwards) == 3  # reuse disk-cached student log probabilities
    meta = compare.read_json(run.root / 'transfer_meta.json')
    assert meta['condition_cache'] == {'hits': 1, 'misses': 1}
    assert meta['student_cache'] == {'hits': 1, 'misses': 0}
    run.rollouts[0]['completion_ids'] = [8, 9]
    compare.transfer_phase(run.args)
    assert len(forwards) == 6  # changed tokens require base + both hinted forwards
    run.args.force = True
    compare.transfer_phase(run.args)
    assert len(forwards) == 9


def test_condition_cache_detects_damage_and_force_repairs(tmp_path):
    cache = ConditionCache(tmp_path, 'scores', {'temperature': 1.0})
    key = cache.key({'prompt': 'question'})
    cache.save(key, {'score': 0.25})
    path = cache.root / f'{key}.json'
    record = json.loads(path.read_text())
    record['result']['score'] = 1.0
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match='Invalid condition cache'):
        cache.load(key)
    forced = ConditionCache(tmp_path, 'scores', cache.config, force=True)
    assert forced.load(key) is None
    forced.save(key, {'score': 0.5})
    assert cache.load(key) == {'score': 0.5}


def test_shared_generator_loads_once_and_selects_distinct_adapters(scoring_run, monkeypatch):
    import vllm
    run = scoring_run
    run.args.generator_group = [('fresh_base', 'base')]
    for step in (10, 20):
        checkpoint = run.root / f'checkpoint-{step}'
        checkpoint.mkdir()
        (checkpoint / 'adapter_config.json').write_text(json.dumps({
            'peft_type': 'LORA', 'base_model_name_or_path': 'base', 'r': step,
        }))
        run.args.generator_group.append((checkpoint.name, str(checkpoint)))
    calls, loads = [], []

    class Generator:
        def __init__(self, **kwargs):
            loads.append(kwargs)

        def get_tokenizer(self):
            return Tokenizer()

        def chat(self, conversations, sampling, lora_request, **kwargs):
            calls.append(lora_request)
            return [SimpleNamespace(outputs=[SimpleNamespace(
                text='Use symmetry.', token_ids=[1, 2], finish_reason='stop',
            ) for _ in range(sampling.n)]) for _ in conversations]

    monkeypatch.setattr(vllm, 'LLM', Generator)
    compare.generate_group_phase(run.args)
    assert len(loads) == 1 and loads[0]['max_lora_rank'] == 32
    assert calls[0] is None
    assert [request.lora_int_id for request in calls[1:]] == [2, 3]
    assert [request.lora_name for request in calls[1:]] == ['checkpoint-10', 'checkpoint-20']
    compare.generate_group_phase(run.args)
    assert len(loads) == 1  # entire generator group reused before allocating an engine


def test_gpu_selection_and_tp_conflicts(monkeypatch):
    parser = compare.build_parser()
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '4,5')
    args = parser.parse_args([])
    compare.validate_args(args, parser)
    assert args.gpus == ['4', '5']
    for extra in (['--gpus', '4', '4'], ['--gpus', '0', '1'],
                  ['--gpus', '4', '5', '--tensor-parallel-size', '2']):
        with pytest.raises(SystemExit):
            compare.validate_args(parser.parse_args(extra), parser)


def test_worker_visibility_is_set_before_spawn_and_parent_is_restored(monkeypatch):
    snapshots = []

    class Process:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def start(self):
            snapshots.append(dict(os.environ))

    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '4,5')
    monkeypatch.setenv('RANK', '7')
    monkeypatch.setattr(compare.multiprocessing, 'get_context', lambda _: SimpleNamespace(Process=Process))
    compare._start_phase(SimpleNamespace(), 'sufficiency', gpu='4')
    compare._start_phase(SimpleNamespace(), 'transfer', gpu='5')
    assert [env['CUDA_VISIBLE_DEVICES'] for env in snapshots] == ['4', '5']
    assert all('RANK' not in env for env in snapshots)
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '4,5'
    assert os.environ['RANK'] == '7'


def _exit_failure():
    raise SystemExit(3)


def test_failed_phase_stops_its_running_peer():
    context = multiprocessing.get_context('spawn')
    peer = context.Process(target=time.sleep, args=(60,), name='waiting-peer')
    failed = context.Process(target=_exit_failure, name='failed-phase')
    peer.start()
    failed.start()
    try:
        with pytest.raises(RuntimeError, match='exit code 3'):
            compare._wait_phases([peer, failed])
        assert not peer.is_alive()
    finally:
        compare._stop_phases([peer, failed])
