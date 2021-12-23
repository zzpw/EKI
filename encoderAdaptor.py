import torch
from torch import nn
from torch.nn import functional as F
from attention import BartAttention

class EncoderAdaptor(nn.Module):
  def __init__(
    self,
    input_dim,
    inner_dim,
    activation_fn,
    adaptor_dropout,
    adaptor_init_scale=1e-3,
  ):
    super(EncoderAdaptor, self).__init__()
    self.dense = nn.Linear(input_dim, inner_dim)
    # nn.init.xavier_normal(self.dense.weight)
    # nn.init.xavier_normal(self.dense.bias)
    nn.init.normal_(self.dense.weight, std=adaptor_init_scale)
    nn.init.zeros_(self.dense.bias)
    self.activation_fn = activation_fn
    self.dropout = nn.Dropout(adaptor_dropout)
    self.out_proj = nn.Linear(inner_dim, input_dim)
    # nn.init.xavier_normal(self.out_proj.weight)
    # nn.init.xavier_normal(self.out_proj.bias)
    nn.init.normal_(self.out_proj.weight, std=adaptor_init_scale)
    nn.init.zeros_(self.out_proj.bias)
    self.out_scale = 2.0

  def forward(self, features):
    x = features
    x = self.dropout(x)
    x = self.dense(x)
    x = self.activation_fn(x)
    x = self.dropout(x)
    x = self.out_proj(x)
    scaled = x.sigmoid()
    gated = scaled * self.out_scale
    x = features * gated
    return x, gated

class AttentionEncoderAdaptor(nn.Module):
  def __init__(self, embed_dim, num_heads):
    super(AttentionEncoderAdaptor, self).__init__()
    self.att = BartAttention(embed_dim=embed_dim, num_heads=num_heads)

  def forward(self, features, attention_mask=None):
    x = features
    gated = self.att(hidden_states=x, attention_mask=attention_mask)
    out = features * gated
    return out, gated