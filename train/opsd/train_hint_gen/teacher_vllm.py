"""Persistent frozen-teacher process on a GPU separate from the trainer.

The control pipe is independent of stdout, so vLLM logs cannot corrupt replies.
CUDA visibility is set by subprocess before any torch/vLLM imports in the child.
There is deliberately no policy-weight synchronization API on this worker.
"""
from __future__ import annotations

import argparse
import hashlib
import atexit
import multiprocessing
import os
import subprocess
import sys
import time
import traceback
from dataclasses import asdict
from multiprocessing.connection import Connection


class FrozenVLLMClient:
    def __init__(self, config):
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(config.teacher_gpu)
        # Never join the trainer's distributed group or inherit its device rank.
        for key in list(env):
            if key in {'RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'LOCAL_WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT'} or key.startswith(('TORCHELASTIC_', 'ACCELERATE_')):
                env.pop(key)
        env['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'
        parent, child = multiprocessing.Pipe()
        self.connection = parent
        self.process = None
        self.pending = False
        self.timeout = config.teacher_timeout
        try:
            self.process = subprocess.Popen(
                [sys.executable, '-u', '-m', __name__, '--worker-fd', str(child.fileno())],
                env=env, pass_fds=(child.fileno(),),
            )
            child.close()
            self.connection.send(asdict(config))
            self.info = self._receive()
        except BaseException:
            child.close()
            self.close()
            raise
        atexit.register(self.close)

    def _receive(self):
        if not self.connection.poll(self.timeout):
            self.close()
            raise TimeoutError('Frozen vLLM teacher timed out')
        try:
            reply = self.connection.recv()
        except EOFError as error:
            code = self.process.poll() if self.process else None
            raise RuntimeError(f'Frozen vLLM teacher exited unexpectedly (exit code {code})') from error
        if 'error' in reply:
            raise RuntimeError('Frozen vLLM teacher failed:\n' + reply['error'])
        return reply

    def submit(self, prompts: list[list[int]], seed: int):
        if self.pending:
            raise RuntimeError('A teacher request is already pending')
        if self.process is None or self.process.poll() is not None:
            raise RuntimeError('Frozen vLLM teacher is not running')
        self.connection.send({'operation': 'generate', 'prompts': prompts, 'seed': seed})
        self.pending = True

    def result(self):
        if not self.pending:
            raise RuntimeError('No teacher request is pending')
        try:
            return self._receive()
        finally:
            self.pending = False

    def fingerprint(self):
        """Read-only full weight hash for integration checks, not the hot path."""
        if self.pending:
            raise RuntimeError('Cannot inspect teacher weights during generation')
        self.connection.send({'operation': 'fingerprint'})
        return self._receive()['fingerprint']

    def close(self):
        process = self.process
        if process is None:
            return
        self.process = None
        try:
            if process.poll() is None:
                if not self.pending:
                    try:
                        self.connection.send({'operation': 'close'})
                        process.wait(timeout=10)
                    except (BrokenPipeError, EOFError, OSError, subprocess.TimeoutExpired):
                        pass
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=5)
        finally:
            self.connection.close()
            atexit.unregister(self.close)



def _weight_fingerprint(vllm_worker):
    import torch
    digest = hashlib.sha256()
    for name, parameter in vllm_worker.model_runner.model.named_parameters():
        digest.update(name.encode())
        digest.update(parameter.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def worker(connection: Connection):
    try:
        config = connection.recv()
        import torch
        from vllm import LLM, SamplingParams

        llm = LLM(
            model=config['model'], dtype='bfloat16', trust_remote_code=True,
            tensor_parallel_size=1, distributed_executor_backend='uni',
            gpu_memory_utilization=config['teacher_gpu_memory_utilization'],
            max_model_len=config['teacher_max_model_length'],
            max_num_seqs=config['teacher_max_num_seqs'],
            max_num_batched_tokens=4096,
            enable_prefix_caching=True,
            enforce_eager=config['teacher_enforce_eager'],
            seed=0,
        )
        connection.send({
            'model': config['model'], 'pid': os.getpid(),
            'cuda_visible_devices': os.environ['CUDA_VISIBLE_DEVICES'],
            'gpu_name': torch.cuda.get_device_name(0),
        })
        while True:
            request = connection.recv()
            if request['operation'] == 'close':
                break
            if request['operation'] == 'fingerprint':
                connection.send({'fingerprint': llm.collective_rpc(_weight_fingerprint)})
                continue
            if request['operation'] != 'generate':
                raise ValueError('Unsupported teacher operation')
            prompts = request['prompts']
            if any(len(p) + config['teacher_max_completion_length'] > config['teacher_max_model_length'] for p in prompts):
                raise ValueError('Teacher prompt plus completion budget exceeds teacher_max_model_length')
            sampling = SamplingParams(
                n=config['teacher_rollouts'],
                max_tokens=config['teacher_max_completion_length'],
                temperature=config['teacher_temperature'], top_p=config['teacher_top_p'],
                top_k=config['teacher_top_k'] or -1,
                seed=request['seed'],
            )
            started = time.perf_counter()
            outputs = llm.generate(
                [{'prompt_token_ids': ids} for ids in prompts], sampling, use_tqdm=False,
            )
            elapsed = time.perf_counter() - started
            if len(outputs) != len(prompts):
                raise RuntimeError('Teacher returned the wrong number of prompts')
            samples = []
            for prompt, output in zip(prompts, outputs, strict=True):
                if list(output.prompt_token_ids) != prompt or len(output.outputs) != config['teacher_rollouts']:
                    raise RuntimeError('Teacher prompt ordering or rollout count does not match request')
                samples.append([
                    {'text': candidate.text, 'token_ids': list(candidate.token_ids),
                     'finish_reason': candidate.finish_reason}
                    for candidate in output.outputs
                ])
            connection.send({'samples': samples, 'generation_seconds': elapsed})
        del llm
    except EOFError:
        pass  # The trainer exited and closed its pipe.
    except BaseException:
        try:
            connection.send({'error': traceback.format_exc()})
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        connection.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker-fd', type=int, required=True)
    worker(Connection(parser.parse_args().worker_fd))
