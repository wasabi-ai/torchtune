# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from typing import List, Optional
from functools import partial

from torch import nn

from torchtune.models.distill_llama._component_builders import distill_llama
from torchtune.models.distill_llama._model_utils import scale_hidden_dim_for_mlp

from torchtune.modules import TransformerDecoder
from torchtune.modules.tokenizers import TikTokenTokenizer


"""
Model builders build specific instantiations using component builders. For example
the distill_llama_8b model builder uses the distill_llama component builder to create the
distill_llama 8B model.
"""


def distill_llama_2b() -> TransformerDecoder:
    """
    Builder for creating a distill_llama model initialized w/ the default 8b parameter values.

    Returns:
        TransformerDecoder: Instantiation of distill_llama 8B model
    """
    return distill_llama(
        vocab_size=128_256,
        num_layers=32,
        num_heads=32,
        num_kv_heads=8,
        inner_dim=256,
        embed_dim=4096,
        max_seq_len=8192,
        intermediate_dim=14336,
        attn_dropout=0.0,
        norm_eps=1e-5,
        rope_base=500000.0,
    )


def distill_llama_8b() -> TransformerDecoder:
    """
    Builder for creating a distill_llama model initialized w/ the default 8b parameter values.

    Returns:
        TransformerDecoder: Instantiation of distill_llama 8B model
    """
    return distill_llama(
        vocab_size=128_256,
        num_layers=80,
        num_heads=64,
        num_kv_heads=8,
        inner_dim=1024,
        embed_dim=8192,
        max_seq_len=8192,
        intermediate_dim=28672,
        attn_dropout=0.0,
        norm_eps=1e-5,
        rope_base=500000.0,
    )


def distill_llama_70b() -> TransformerDecoder:
    """
    Builder for creating a distill_llama model initialized w/ the default 70B parameter values.

    Returns:
        TransformerDecoder: Instantiation of distill_llama 70 model
    """
    return distill_llama(
        vocab_size=128_256,
        num_layers=80,
        num_heads=64,
        num_kv_heads=8,
        embed_dim=8192,
        max_seq_len=8192,
        intermediate_dim=28672,
        attn_dropout=0.0,
        norm_eps=1e-5,
        rope_base=500000.0,
    )


def distill_llama_tokenizer(path: str) -> TikTokenTokenizer:
    tiktoken = TikTokenTokenizer(path)
    tiktoken.pad_id = 0
    return tiktoken
