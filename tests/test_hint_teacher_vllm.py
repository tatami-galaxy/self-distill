import os
from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from train.opsd.train_hint_gen.lib import (
    ConstrainedHintReward, ConstrainedHintRewardConfig, FrozenHintTeacher,
    HintRewardConfig, validate_teacher_devices,
)
from train.opsd.train_hint_gen.teacher_vllm import worker


def config(**kwargs):
    return ConstrainedHintRewardConfig(
        model='mock', dataset='deepmath', teacher_backend='vllm', teacher_gpu='5',
        **kwargs,
    )


def test_transfer_overlaps_batched_teacher_and_scores_map_to_original_hints():
    events = []
    class Client:
        def submit(self, prompts, seed):
            events.append(('submit', prompts))
        def result(self):
            events.append(('result',))
            return {'samples': [
                [{'text': '4', 'token_ids': [4], 'finish_reason': 'stop'},
                 {'text': 'wrong', 'token_ids': [9], 'finish_reason': 'length'}],
                [{'text': '15', 'token_ids': [1, 5], 'finish_reason': 'stop'},
                 {'text': '15', 'token_ids': [1, 5], 'finish_reason': 'stop'}],
            ], 'generation_seconds': 0.2}
        def close(self):
            events.append(('close',))
    backend = object.__new__(FrozenHintTeacher)
    backend.config = config(teacher_rollouts=2)
    backend._vllm_teacher = Client()
    backend.last_metrics = {}
    backend._render_prompt = lambda messages: [len(messages[-1]['content'])]
    def transfer(question, hint):
        events.append(('transfer', question))
        return {'q1': 0.1, 'q2': 0.2}[question]
    backend.score_transfer = transfer
    reward = ConstrainedHintReward(backend.config, backend)
    extras = {}
    with patch('train.opsd.train_hint_gen.lib.grade', side_effect=lambda text, answer, dataset: (text, text == answer)):
        values = reward(
            prompts=[[]] * 3,
            completions=['Use addition.', 'The answer is 4.', 'Use multiplication.'],
            completion_ids=[[1, 2], [3], [4, 5]],
            question=['q1', 'q1', 'q2'], final_answer=['4', '4', '15'],
            log_extra=lambda name, value: extras.__setitem__(name, value),
        )
    assert [e[0] for e in events] == ['submit', 'transfer', 'transfer', 'result']
    assert len(events[0][1]) == 2  # Invalid hint was never sent to the teacher.
    assert extras['hint_sufficiency'] == [0.5, 0.0, 1.0]
    assert extras['hint_transfer'] == [0.1, 0.0, 0.2]
    assert values[1] < values[0]
    assert reward.dual_lambda == pytest.approx(1 + 0.05 * (0.7 - 0.5))
    assert backend.last_metrics['hint/teacher_generated_tokens'] == 6
    assert backend.last_metrics['hint/teacher_truncated_fraction'] == 0.25


def test_transfer_exception_closes_pending_teacher():
    backend = object.__new__(FrozenHintTeacher)
    backend.config = config()
    events = []
    backend._vllm_teacher = SimpleNamespace(close=lambda: events.append('closed'))
    backend._submit_sufficiency = lambda questions, hints: backend._vllm_teacher
    def fail(*args):
        raise RuntimeError('transfer failed')
    backend.score_transfer = fail
    with pytest.raises(RuntimeError, match='transfer failed'):
        backend.score_hints(['q'], ['4'], ['Use addition.'])
    assert events == ['closed']
    assert backend._vllm_teacher is None


def test_all_invalid_group_does_not_start_worker():
    backend = object.__new__(FrozenHintTeacher)
    backend.config = config()
    with patch.object(backend, '_teacher_client', side_effect=AssertionError('must not start')):
        reward = ConstrainedHintReward(backend.config, backend)
        assert reward(prompts=[[]], completions=[''], completion_ids=[[1]], question=['q'], final_answer=['4']) == [0.0]
    assert reward.dual_updates == 1


@pytest.mark.parametrize('config_type', [HintRewardConfig, ConstrainedHintRewardConfig])
def test_backend_configuration_requires_separate_teacher_gpu(config_type):
    with pytest.raises(ValueError, match='dedicated teacher_gpu'):
        config_type(model='m', dataset='d', teacher_backend='vllm').validate()
    with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '4', 'WORLD_SIZE': '1'}):
        validate_teacher_devices(config())
        with pytest.raises(ValueError, match='differ'):
            validate_teacher_devices(SimpleNamespace(teacher_backend='vllm', teacher_gpu='4'))
    with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '4,5', 'WORLD_SIZE': '1'}):
        with pytest.raises(ValueError, match='one training process'):
            validate_teacher_devices(config())


def test_worker_keeps_sampling_settings_and_reports_bad_order():
    received = {}
    class LLM:
        def __init__(self, **kwargs):
            received['llm'] = kwargs
        def generate(self, prompts, sampling, use_tqdm):
            received['sampling'] = sampling
            return [SimpleNamespace(prompt_token_ids=[999], outputs=[])]
    class Pipe:
        def __init__(self):
            self.requests = iter([asdict(config(teacher_rollouts=4)),
                                  {'operation': 'generate', 'prompts': [[1, 2]], 'seed': 123}])
            self.replies = []
        def recv(self):
            return next(self.requests)
        def send(self, reply):
            self.replies.append(reply)
        def close(self):
            self.closed = True
    pipe = Pipe()
    with (
        patch.dict('sys.modules', {'vllm': SimpleNamespace(LLM=LLM, SamplingParams=lambda **kw: kw)}),
        patch('torch.cuda.get_device_name', return_value='test GPU'),
        patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '5'}),
    ):
        worker(pipe)
    assert pipe.replies[0]['cuda_visible_devices'] == '5'
    assert 'ordering or rollout count' in pipe.replies[1]['error']
    assert received['llm']['model'] == 'mock'
    assert received['sampling']['n'] == 4
    assert received['sampling']['top_k'] == 20
    assert received['sampling']['temperature'] == 1.0
    assert received['sampling']['seed'] == 123
    assert pipe.closed
