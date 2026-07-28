"""Shared fixtures for the trainer tests.

Lives here rather than in one test module because every arm that gives the critic its own
prompt (train_ppo_val.py today, train_ppo_pi.py whenever it grows coverage) needs the same
two things: a chat tokenizer whose generation header is identifiable, and a stub carrying
exactly the attributes TRL's prompt rendering reads.
"""

import types
from collections import defaultdict

import torch

from train.ppo.train_ppo import PPOTrainer

TOKENIZER_ID = "Qwen/Qwen3-1.7B"


class FakeChatTokenizer:
    """Minimal chat tokenizer for always-on prompt-boundary tests."""

    pad_token_id = 0
    generation_header = [901, 902]

    def __init__(self):
        self.calls = []

    def apply_chat_template(
        self,
        conversation,
        tools=None,
        chat_template=None,
        add_generation_prompt=False,
        tokenize=True,
        return_dict=False,
        **kwargs,
    ):
        self.calls.append(
            {
                "conversation": conversation,
                "tools": tools,
                "chat_template": chat_template,
                "add_generation_prompt": add_generation_prompt,
                "tokenize": tokenize,
                "return_dict": return_dict,
                "kwargs": kwargs,
            }
        )
        role_tokens = {"system": 101, "user": 102, "assistant": 103}
        input_ids = []
        for messages in conversation:
            ids = []
            for message in messages:
                ids.append(role_tokens[message["role"]])
                ids.extend(200 + len(word) for word in message["content"].split())
            if add_generation_prompt:
                ids.extend(self.generation_header)
            input_ids.append(ids)
        return {"input_ids": input_ids}


def make_prompt_stub(tokenizer, tools=None, chat_template=None, chat_template_kwargs=None):
    """Attributes read by our prompt builder and TRL's `_tokenize_prompts`.

    Listed EXPLICITLY rather than defaulted via a catch-all `__getattr__`: an
    AttributeError here means TRL's prompt rendering grew a branch, which is exactly the
    drift these tests exist to catch, so it should stop the suite and be read. A stub that
    answered None to anything would swallow it and let the alignment assertions pass
    against a path TRL no longer takes.

    `_render_value_prompts` is bound from the real `PPOTrainer` rather than faked, because it
    is the method under test in the alignment cases -- the arms only compose the messages.
    """
    stub = types.SimpleNamespace(
        processing_class=tokenizer,
        _tokenizer=tokenizer,
        tools=tools,
        chat_template=chat_template,
        chat_template_kwargs={} if chat_template_kwargs is None else chat_template_kwargs,
        _is_vlm=False,
        # TRL 1.9.0: `_tokenize_prompts` renders each prompt with its own tool schema when
        # environments are configured. None selects the single batched apply_chat_template
        # call -- the path `_render_value_prompts` mirrors, and the one under test.
        environment_factories=None,
        _metrics={"train": defaultdict(list)},
        model=types.SimpleNamespace(training=True),
    )
    stub._render_value_prompts = types.MethodType(PPOTrainer._render_value_prompts, stub)
    return stub


__all__ = ["TOKENIZER_ID", "FakeChatTokenizer", "make_prompt_stub", "torch"]
