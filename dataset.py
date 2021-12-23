# -*- coding: utf-8 -*-
# @Author : zengpengwu
# @Time : 2021/12/20 15:36
# @File : dataset.py

import argparse
import json

from torch.utils.data import Dataset
from transformers import (
  BartTokenizer,
  PreTrainedTokenizer,
)


class CombinedConcepts(Dataset):
  def __init__(self, json_data_path, tokenizer, concepts_max_length=None, target_max_length=None):
    """

    Args:
      json_data_path (str):
      tokenizer (PreTrainedTokenizer):
      concepts_max_length (int):
      target_max_length (int):
    """
    super(CombinedConcepts, self).__init__()
    self.json_data_path = json_data_path
    self.tokenizer = tokenizer
    self.max_concepts_length = concepts_max_length
    self.max_target_length = target_max_length
    self.data = None
    self.input_ids = None
    self.attention_mask = None
    self.labels = None
    self.setup_dataset()


  def setup_dataset(self):
    with open(self.json_data_path, 'r') as f:
      self.data = json.load(f)
    self.data['combined_concepts'] = []
    for idx, concept in enumerate(self.data['concepts']):
      comb = ' '.join(concept) + self.tokenizer.sep_token
      comb += self.tokenizer.sep_token.join(self.data['retrieved'][idx])
      self.data['combined_concepts'].append(comb)
    if self.max_concepts_length is None:
      inter_data = self.tokenizer(self.data['combined_concepts'], padding=True, return_tensors='pt')
    else:
      inter_data = self.tokenizer(self.data['combined_concepts'], padding='max_length', truncation=True,
                                  max_length=self.max_concepts_length, return_tensors='pt')
    self.input_ids = inter_data['input_ids']
    self.attention_mask = inter_data['attention_mask']
    if 'target' in self.data:
      if self.max_target_length is None:
        self.labels = self.tokenizer(self.data['target'], padding=True, return_tensors='pt')['input_ids']
      else:
        self.labels = self.tokenizer(self.data['target'], padding='max_length', truncation=True,
                                     max_length=self.max_target_length, return_tensors='pt')['input_ids']
  
  def __getitem__(self, index):
    if self.labels is not None:
      return (
        self.input_ids[index],
        self.attention_mask[index],
        self.labels[index],
      )
    else:
      return (
        self.input_ids[index],
        self.attention_mask[index],
      )

  def __len__(self):
    return len(self.data['concepts'])


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--concepts-max-length', type=int, default=128)
  parser.add_argument('--target-max-length', type=int, default=32)
  parser.add_argument('--train-data-dir', type=str, default='./train_data.json')
  parser.add_argument('--valid-data-dir', type=str, default='./valid_data.json')
  args = parser.parse_args()
  tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
  train_dataset = CombinedConcepts(args.train_data_dir, tokenizer, args.concepts_max_length, args.target_max_length)
  valid_dataset = CombinedConcepts(args.valid_data_dir, tokenizer, args.concepts_max_length, args.target_max_length)
  train_data = train_dataset.data
  valid_data = valid_dataset.data
