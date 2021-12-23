import json
import os

import torch
from transformers import (
  AutoModelForSeq2SeqLM,
  AutoTokenizer,
  AutoConfig
)
from dataset import CombinedConcepts
from data_preprocess import GEN_MODEL_SAVED_PATH
from generate_utils import generate_sentences
from train_utils import model_train_batch

metrics_str = ['Bleu_3: ', 'Bleu_4: ', 'ROUGE_L: ', 'CIDEr: ', 'METEOR: ', 'SPICE: ']


@torch.no_grad()
def model_evaluation_with_loss(val_dataloader, model=None, MODEL_SAVE_PATH=None):
  if MODEL_SAVE_PATH is None:
    MODEL_SAVE_PATH = GEN_MODEL_SAVED_PATH
  if model is None:
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_SAVE_PATH)

  origin_device = model.device
  device = torch.device(torch.cuda.device_count() - 1)
  model.to(device)
  val_loss = 0
  for step, batch in enumerate(val_dataloader):
    loss = model_train_batch(model, batch)
    val_loss += loss.item()
  torch.cuda.empty_cache()
  model.to(origin_device)
  return val_loss / len(val_dataloader)


@torch.no_grad()
def model_evaluation_with_metric(concept_set, base_model=None, model=None, tokenizer=None,
                                 gen_data_type='dev', batch_size=16, model_save_path=None, available_device=(0, 1)):
  if model_save_path is None:
    model_save_path = GEN_MODEL_SAVED_PATH
  if model is None:
    model = AutoModelForSeq2SeqLM(AutoConfig.from_pretrained(base_model)).from_pretrained(model_save_path)
  if tokenizer is None:
    tokenizer = AutoTokenizer.from_pretrained(base_model)
  model.eval()
  batch_num = len(concept_set) // batch_size
  device = torch.device(available_device[-1])
  origin_device = model.device
  model.to(device)

  output_filename = './{}_generation.txt'.format(gen_data_type)
  evaluation_path = '~/JupyterPath/CommonGen/dataset/final_data/commongen/{}_generation.txt'.format(gen_data_type)

  with open(output_filename, mode='w') as f:
    f.truncate()

  with open(output_filename, mode='a') as f:
    for i in range(batch_num):
      doc = concept_set[i * batch_size: (i + 1) * batch_size]
      ans = generate_sentences(doc, model, tokenizer)
      for sen in ans:
        f.write(sen + '\n')

  doc = concept_set[batch_size * batch_num:]
  ans = generate_sentences(doc, model, tokenizer)
  with open(output_filename.format(gen_data_type), mode='a') as f:
    for sen in ans:
      f.write(sen + '\n')

  model.to(origin_device)
  print('evaluating model on {} dataset...'.format(gen_data_type))
  os.system("cp {} {}".format(output_filename, evaluation_path))
  cwd = os.popen('bash evaluation.sh')
  results, results_str = get_evaluation_result(cwd.read())
  return results, results_str


def get_evaluation_result(eval_str):
  eval_str = eval_str.split('\n')
  eval_results = dict()
  result_str = ''
  for metric in metrics_str:
    eval_results[metric] = 0.
    result_str += '{:^15s}'.format(metric)
  for line in eval_str:
    for metric in metrics_str:
      if line.startswith(metric):
        eval_results[metric] = float(line.replace(metric, ''))
  result_str += '\n'
  for metric in metrics_str:
    result_str += '{:^15f}'.format(eval_results[metric])
  return eval_results, result_str


if __name__ == '__main__':
  model = AutoModelForSeq2SeqLM.from_pretrained('facebook/bart-large')
  tokenizer = AutoTokenizer.from_pretrained('facebook/bart-large')
  model.load_state_dict(torch.load('./checkpoints/integration_2021-12-21/model.pt'))
  model.to(torch.device(1))
  valid_dataset = CombinedConcepts('./valid_data.json', tokenizer)
  val_combined_concepts = valid_dataset.data['combined_concepts']
  result, result_str = model_evaluation_with_metric(concept_set=val_combined_concepts, model=model, tokenizer=tokenizer)
  print(result_str)

  t = 2021
  print(val_combined_concepts[t].split(tokenizer.sep_token)[0])
  print(generate_sentences([val_combined_concepts[t]], model, tokenizer))
  # ans = []
  # for concept in val_combined_concepts:
  #   ans.append(generate_sentences([concept], model, tokenizer))