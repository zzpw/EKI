import torch
import json
from torch import nn
from models import BartCommonGen
from transformers import BartTokenizer, BartConfig

# model = BartCommonGen.from_pretrained('facebook/bart-large')
# tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
# config = BartConfig()
#
# text = 'i am ok'
# input_ids = torch.tensor([tokenizer.encode(text)])
# out = model(input_ids=input_ids, labels=input_ids)
# loss_fct = nn.CrossEntropyLoss(label_smoothing=0.1)
# loss_fct(torch.Tensor([[0.1, 2.3]]), torch.tensor([1,])).item()

with open('re_ext_train_data.json', 'r') as f:
  re_data = json.load(f)

with open('valid_data.json', 'r') as f:
  valid_data = json.load(f)

with open('train_data.json', 'r') as f:
  train_data = json.load(f)

re_concepts_set = set()
for concepts in re_data['concepts']:
  re_concepts_set = re_concepts_set.union(set(concepts))

val_concepts_set = set()
for concepts in valid_data['concepts']:
  val_concepts_set = val_concepts_set.union(set(concepts))

train_concepts_set = set()
for concepts in train_data['concepts']:
  train_concepts_set = train_concepts_set.union(set(concepts))

val_re_intersection = re_concepts_set.intersection(val_concepts_set)
val_train_intersection = train_concepts_set.intersection(val_concepts_set)
re_train_intersection = re_concepts_set.intersection(train_concepts_set)
