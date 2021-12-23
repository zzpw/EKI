from typing import Optional

import torch
import random
import numpy as np
from torch import nn

def set_seed(seed):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
  """
  Shift input ids one token to the right.
  """
  shifted_input_ids = input_ids.new_zeros(input_ids.shape)
  shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
  shifted_input_ids[:, 0] = decoder_start_token_id

  if pad_token_id is None:
    raise ValueError("self.model.config.pad_token_id has to be defined.")
  # replace possible -100 values in labels by `pad_token_id`
  shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

  return shifted_input_ids


def _make_causal_mask(input_ids_shape: torch.Size, dtype: torch.dtype, past_key_values_length: int = 0):
  """
  Make causal mask used for bi-directional self-attention.
  """
  bsz, tgt_len = input_ids_shape
  mask = torch.full((tgt_len, tgt_len), float("-inf"))
  mask_cond = torch.arange(mask.size(-1))
  mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
  mask = mask.to(dtype)

  if past_key_values_length > 0:
    mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype), mask], dim=-1)
  return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
  """
  Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
  """
  bsz, src_len = mask.size()
  tgt_len = tgt_len if tgt_len is not None else src_len

  expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

  inverted_mask = 1.0 - expanded_mask

  return inverted_mask.masked_fill(inverted_mask.bool(), torch.finfo(dtype).min)


class BartLearnedPositionalEmbedding(nn.Embedding):
  """
  This module learns positional embeddings up to a fixed maximum size.
  """

  def __init__(self, num_embeddings: int, embedding_dim: int):
    # Bart is set up so that if padding_idx is specified then offset the embedding ids by 2
    # and adjust num_embeddings appropriately. Other models don't have this hack
    self.offset = 2
    super().__init__(num_embeddings + self.offset, embedding_dim)

  def forward(self, input_ids_shape: torch.Size, past_key_values_length: int = 0):
    """`input_ids_shape` is expected to be [bsz x seqlen]."""
    bsz, seq_len = input_ids_shape[:2]
    positions = torch.arange(
      past_key_values_length, past_key_values_length + seq_len, dtype=torch.long, device=self.weight.device
    )
    return super().forward(positions + self.offset)
