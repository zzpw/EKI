import json

import torch
from datasets import load_dataset
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from transformers import T5Tokenizer


GEN_MODEL_SAVED_PATH = './checkpoints/model.pt'
CACHED_TRAIN_PATH = './train_data_combined.json'
CACHED_VAL_PATH = './val_data_combined.json'
CACHED_TRAIN_WITH_PREFIX_PATH = './train_data_combined_with_prefix.json'
CACHED_VAL_WITH_PREFIX_PATH = './val_data_combined_with_prefix.json'


def packet_batch_data_into_ids(batch_text_data, tokenizer=None, max_length=None):
  """

  :param batch_text_data:
  :param tokenizer:
  :param max_length:
  :return: tokenized ids data, includes 'input_ids', 'attention_mask', etc.
  """
  if tokenizer is None:
    tokenizer = T5Tokenizer.from_pretrained('t5-base')
  if max_length is None:
    ids_data = tokenizer(batch_text_data, padding=True, return_tensors='pt', truncation=True)
  else:
    ids_data = tokenizer(batch_text_data, padding='max_length', max_length=max_length,
                         return_tensors='pt', truncation=True)
  return ids_data


def get_concepts_target_ids_data(batch_data, tokenizer=None, max_concepts_length=None, max_target_length=None):
  """

  :param batch_data:
  :param tokenizer:
  :param max_concepts_length:
  :param max_target_length:
  :return: concepts_ids, target_ids
  """
  batch_concepts_data = batch_data['concepts']
  batch_target_data = batch_data['target']
  concepts_ids_data = packet_batch_data_into_ids(batch_concepts_data, tokenizer, max_concepts_length)
  target_ids_data = packet_batch_data_into_ids(batch_target_data, tokenizer, max_target_length)
  return concepts_ids_data, target_ids_data


def packet_ids_into_dataloader(input_ids_data, label_ids_data, batch_size):
  """

  :param input_ids_data: batch ids data
  :param label_ids_data:
  :param batch_size:
  :return: dataloader
  """
  inputs = torch.tensor(input_ids_data['input_ids'])
  attention_mask = torch.tensor(input_ids_data['attention_mask'])
  labels = torch.tensor(label_ids_data['input_ids'])
  data = TensorDataset(inputs, attention_mask, labels)
  sampler = RandomSampler(data)
  dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)
  return dataloader

def replace_special_token(data, tokenizer):
  """

  :param data:
  :param tokenizer:
  :param suffix:
  :return:
  """
  for idx, concepts in enumerate(data['concepts']):
    concepts = concepts.replace('[CLS]', tokenizer.bos_token)
    concepts = concepts.replace('[SEP]', tokenizer.sep_token)
    data['concepts'][idx] = concepts


def get_data_from_cache(batch_size, tokenizer, max_concepts_length, max_target_length,
                        train_cache_path=CACHED_TRAIN_PATH, val_cache_path=CACHED_VAL_PATH, add_prefix=False):
  """

  :param batch_size:
  :param tokenizer:
  :param max_concepts_length:
  :param max_target_length:
  :param train_cache_path:
  :param val_cache_path:
  :param add_prefix:
  :return:
  """
  train_data, val_data = get_text_data_from_cache(train_cache_path, val_cache_path, add_prefix=add_prefix)
  replace_special_token(train_data, tokenizer)
  replace_special_token(val_data, tokenizer)
  train_concepts_ids, train_target_ids = get_concepts_target_ids_data(train_data, tokenizer, max_concepts_length,
                                                                      max_target_length)
  val_concepts_ids, val_target_ids = get_concepts_target_ids_data(val_data, tokenizer, max_concepts_length,
                                                                  max_target_length)
  train_dataloader = packet_ids_into_dataloader(train_concepts_ids, train_target_ids, batch_size)
  val_dataloader = packet_ids_into_dataloader(val_concepts_ids, val_target_ids, batch_size)
  return train_dataloader, val_dataloader, train_data, val_data


def get_text_data_from_cache(train_cache_path=CACHED_TRAIN_PATH, val_cache_path=CACHED_VAL_PATH, add_prefix=False):
  """

  :param train_cache_path:
  :param val_cache_path:
  :param add_prefix:
  :return:
  """
  if add_prefix:
    train_cache_path = CACHED_TRAIN_WITH_PREFIX_PATH
    val_cache_path = CACHED_VAL_WITH_PREFIX_PATH
  with open(train_cache_path, 'r') as f:
    train_data = json.load(f)
  with open(val_cache_path, 'r') as f:
    val_data = json.load(f)
  return train_data, val_data
