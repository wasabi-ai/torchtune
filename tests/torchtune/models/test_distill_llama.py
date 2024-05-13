# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from tests.test_utils import fixed_init_model
from torchtune.models import distill_llama
from torchtune.utils.seed import set_seed

EMBED_DIM = 128
NUM_LAYERS = 4
NUM_HEADS = 16
NUM_KV_HEADS = 8
VOCAB_SIZE = 32000
MAX_SEQ_LEN = 2048
BSZ = 2
SEQ_LEN = 100


def model_size(model):
    model = model.to(dtype=torch.bfloat16)
    param_size = 0
    buffer_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    return {
        "param_size": param_size / 1e9,
        "buffer_size": buffer_size / 1e9,
    }


class TestDistllLlama:
    @pytest.fixture
    def inputs(self):
        return torch.randint(0, VOCAB_SIZE, (BSZ, SEQ_LEN))

    def test_init_2b(self):
        model = distill_llama.distill_llama_2b()
        print(model_size(model))
