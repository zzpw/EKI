import torch
from torch import nn
from transformers import BartConfig
from transformers.activations import ACT2FN

from attention import BartAttention


class BartEncoderLayer(nn.Module):
  def __init__(self, config: BartConfig):
    super().__init__()
    self.embed_dim = config.d_model
    self.self_attn = BartAttention(
      embed_dim=self.embed_dim,
      num_heads=config.encoder_attention_heads,
      dropout=config.attention_dropout,
    )
    self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
    self.dropout = config.dropout
    self.activation_fn = ACT2FN[config.activation_function]
    self.activation_dropout = config.activation_dropout
    self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
    self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
    self.final_layer_norm = nn.LayerNorm(self.embed_dim)

  def forward(
      self,
      hidden_states: torch.Tensor,
      attention_mask: torch.Tensor,
      layer_head_mask: torch.Tensor,
      output_attentions: bool = False,
  ):
    """
    Args:
        hidden_states (:obj:`torch.FloatTensor`): input to the layer of shape `(seq_len, batch, embed_dim)`
        attention_mask (:obj:`torch.FloatTensor`): attention mask of size
            `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
        layer_head_mask (:obj:`torch.FloatTensor`): mask for attention heads in a given layer of size
            `(encoder_attention_heads,)`.
        output_attentions (:obj:`bool`, `optional`):
            Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under
            returned tensors for more detail.
    """
    residual = hidden_states
    hidden_states, attn_weights, _ = self.self_attn(
      hidden_states=hidden_states,
      attention_mask=attention_mask,
      layer_head_mask=layer_head_mask,
      output_attentions=output_attentions,
    )
    hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states = residual + hidden_states
    hidden_states = self.self_attn_layer_norm(hidden_states)

    residual = hidden_states
    hidden_states = self.activation_fn(self.fc1(hidden_states))
    hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    hidden_states = self.fc2(hidden_states)
    hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states = residual + hidden_states
    hidden_states = self.final_layer_norm(hidden_states)

    if hidden_states.dtype == torch.float16 and (
        torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any()
    ):
      clamp_value = torch.finfo(hidden_states.dtype).max - 1000
      hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

    outputs = (hidden_states,)

    if output_attentions:
      outputs += (attn_weights,)

    return outputs
