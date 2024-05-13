# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial
from typing import List, Literal, Optional
import torch
from torch import nn

from torchtune.models.distill_llama._model_utils import scale_hidden_dim_for_mlp
from torchtune.modules import (
    CausalSelfAttention,
    RMSNorm,
    FeedForward,
    RotaryPositionalEmbeddings,
    TransformerDecoder,
    TransformerDecoderLayer,
)

"""
Component builders for the distill_llama model and popular variants such as LoRA.

TorchTune provides composable building blocks. Builder functions help
stitch these building blocks into higher-level components. This design has
two benefits:
- The building blocks themselves are very flexible. For example, ``CausalSelfAttention``
can take either nn.Linear or nn.LoRALinear for ``q_proj``.
- Builder functions expose a set of configurable params which keep the constructors of
the building blocks simple.
"""


class FactorizedLinear(nn.Module):
    """
    Factorized linear layer for LoRA. This is a wrapper around two linear layers.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        bias: bool = True,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.bias = bias
        print(in_dim, hidden_dim, out_dim, bias, device)
        self.proj1 = nn.Linear(in_dim, hidden_dim, bias=bias, device=device)
        self.proj2 = nn.Linear(hidden_dim, out_dim, bias=bias, device=device)
        for param in self.parameters():
            print("FactorizedLinear: ", param.nelement())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj2(self.proj1(x))


class FactorizedEmbeding(nn.Module):
    """
    Factorized embedding layer for LLama.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        inner_dim: int,
        bias: bool = True,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.inner_dim = inner_dim
        self.bias = bias
        self.device = device
        self.proj1 = nn.Embedding(vocab_size, inner_dim, device=device)
        self.proj2 = nn.Linear(inner_dim, embed_dim, bias=bias, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embed = self.proj1(x)
        output = self.proj2(embed)
        return output


def distill_llama(
    vocab_size: int,
    num_layers: int,
    num_heads: int,
    num_kv_heads: int,
    inner_dim: int,
    embed_dim: int,
    max_seq_len: int,
    attn_dropout: float = 0.0,
    rope_base: int = 500000.0,
    intermediate_dim: Optional[int] = None,
    norm_eps: float = 1e-5,
    device: Optional[torch.device] = None,
):
    """
    Build the decoder associated with the distill_llama model. This includes:
    - Token embeddings
    - num_layers number of TransformerDecoderLayer blocks
    - RMS Norm layer applied to the output of the transformer
    - Final projection into token space

    Args:
        vocab_size (int): number of tokens in vocabulary.
        num_layers (int): number of layers in the transformer decoder.
        num_heads (int): number of query heads. For MHA this is also the
            number of heads for key and value
        num_kv_heads (int): number of key and value heads. If specified,
            user should ensure `num_heads` % `num_kv_heads` == 0. Default value is
            `None`, in which case this is the same as MHA
        embed_dim (int): embedding dimension for self-attention
        max_seq_len (int): maximum sequence length the model will be run with, as used
            by :func:`~torchtune.modules.KVCache`
        attn_dropout (float): dropout value passed onto scaled_dot_product_attention.
            Default: 0.0
        intermediate_dim (Optional[int]): intermediate dimension for MLP. If not specified,
            this is computed using :func:`~torchtune.modules.scale_hidden_dim_for_mlp`
        norm_eps (float): epsilon in RMS norms.

    Returns:
        TransformerDecoder: Instantiation of distill_llama model.
    """
    head_dim = embed_dim // num_heads
    num_kv_heads = num_kv_heads if num_kv_heads else num_heads
    rope = RotaryPositionalEmbeddings(
        dim=head_dim, max_seq_len=max_seq_len, base=rope_base
    )
    self_attn = CausalSelfAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        q_proj=FactorizedLinear(
            in_dim=embed_dim,
            hidden_dim=inner_dim,
            out_dim=num_heads * head_dim,
            bias=False,
            device=device,
        ),
        k_proj=FactorizedLinear(
            in_dim=embed_dim,
            hidden_dim=inner_dim,
            out_dim=num_kv_heads * head_dim,
            bias=False,
            device=device,
        ),
        v_proj=FactorizedLinear(
            in_dim=embed_dim,
            hidden_dim=inner_dim,
            out_dim=num_kv_heads * head_dim,
            bias=False,
            device=device,
        ),
        output_proj=FactorizedLinear(
            in_dim=embed_dim,
            hidden_dim=inner_dim,
            out_dim=embed_dim,
            bias=False,
            device=device,
        ),
        pos_embeddings=rope,
        max_seq_len=max_seq_len,
        attn_dropout=attn_dropout,
    )
    hidden_dim = (
        intermediate_dim if intermediate_dim else scale_hidden_dim_for_mlp(embed_dim)
    )
    mlp = distill_llama_mlp(
        embed_dim=embed_dim, inner_dim=inner_dim, hidden_dim=hidden_dim
    )
    layer = TransformerDecoderLayer(
        attn=self_attn,
        mlp=mlp,
        sa_norm=RMSNorm(dim=embed_dim, eps=norm_eps),
        mlp_norm=RMSNorm(dim=embed_dim, eps=norm_eps),
    )

    tok_embeddings = FactorizedEmbeding(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        inner_dim=inner_dim,
        device=device,
    )
    for param in tok_embeddings.parameters():
        print("Embed: ", param.nelement())
    output_proj = FactorizedLinear(
        embed_dim,
        inner_dim,
        vocab_size,
        bias=False,
        device=device,
    )
    return TransformerDecoder(
        tok_embeddings=tok_embeddings,
        layer=layer,
        num_layers=num_layers,
        max_seq_len=max_seq_len,
        num_heads=num_heads,
        head_dim=head_dim,
        norm=RMSNorm(embed_dim, eps=norm_eps),
        output=output_proj,
    )


def distill_llama_mlp(embed_dim: int, inner_dim: int, hidden_dim: int) -> FeedForward:
    """
    Build the MLP layer associated with the Llama model.
    """
    gate_proj = FactorizedLinear(
        in_dim=embed_dim, hidden_dim=inner_dim, out_dim=hidden_dim, bias=False
    )
    up_proj = FactorizedLinear(
        in_dim=embed_dim, hidden_dim=inner_dim, out_dim=hidden_dim, bias=False
    )
    down_proj = FactorizedLinear(
        in_dim=hidden_dim, hidden_dim=inner_dim, out_dim=embed_dim, bias=False
    )
    return FeedForward(gate_proj=gate_proj, down_proj=down_proj, up_proj=up_proj)
