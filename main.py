import argparse

import os
import datetime
import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import BartTokenizer
from transformers import get_linear_schedule_with_warmup

from dataset import CombinedConcepts
from evaluation import model_evaluation_with_metric, metrics_str
from models import BartCommonGen
from utils import set_seed
from train_utils import model_train_batch, set_weight_decay

parser = argparse.ArgumentParser()

today =str(datetime.date.today())
parser.add_argument('--base-model', type=str, default='facebook/bart-large')
parser.add_argument('--model-save-path', type=str, default='./checkpoints/integration_' + today + '/model.pt')
parser.add_argument('--log-dir', type=str, default='./run/integration_' + today + '/')
parser.add_argument('--concepts-max-length', type=int, default=128)
parser.add_argument('--target-max-length', type=int, default=24)
parser.add_argument('--available-device', type=int, default=[0, 1], nargs='+',
                    help='available cuda device list, like: [0, 1]')
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--batch-size', type=int, default=16)
parser.add_argument('--max-grad-norm', type=float, default=1.)
parser.add_argument('--gradient-accumulation', type=int, default=4)
parser.add_argument('--warmup-ratio', type=float, default=.05)
parser.add_argument('--learning-rate', type=float, default=2e-5)
parser.add_argument('--weight-decay', type=float, default=0.01)
parser.add_argument('--evaluation-data-type', type=str, default='dev')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--train-data-dir', type=str, default='./integration_train_data.json')
parser.add_argument('--valid-data-dir', type=str, default='./valid_data.json')
args = parser.parse_args()

set_seed(args.seed)

model_save_path_dir = os.path.join(*os.path.split(args.model_save_path)[:-1])
if not os.path.exists(model_save_path_dir):
  os.makedirs(model_save_path_dir)

if not os.path.exists(args.log_dir):
  os.makedirs(args.log_dir)

print('-' * 80)
for k, v in vars(args).items():
  print(k, '=', v)
print('-' * 80)

tokenizer = BartTokenizer.from_pretrained(args.base_model)
model = BartCommonGen.from_pretrained(args.base_model)
model.to(torch.device(args.available_device[0]))

print('Creating dataloader...')
# train_dataloader, val_dataloader, train_data, val_data = get_data_from_cache(
#   args.batch_size, tokenizer, args.max_concepts_length, args.max_target_length, add_prefix=False)
train_dataset = CombinedConcepts(args.train_data_dir, tokenizer, args.concepts_max_length, args.target_max_length)
valid_dataset= CombinedConcepts(args.valid_data_dir, tokenizer, args.concepts_max_length, args.target_max_length)
train_data = train_dataset.data
valid_data = valid_dataset.data
train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True)
print('Dataloader Created.')

epoch_step_num = len(train_dataloader)
total_steps = epoch_step_num * args.epochs
warmup_steps = int(total_steps * args.warmup_ratio)
eval_steps = int(0.3 * epoch_step_num)
pre_result = 0.
train_epoch_loss = 0.
train_total_steps = 0

optimizer = optim.AdamW(params=set_weight_decay(model), lr=args.learning_rate, weight_decay=args.weight_decay)
optimizer_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps // args.gradient_accumulation,
                                                      num_training_steps=total_steps // args.gradient_accumulation)
# optimizer_scheduler = get_polynomial_decay_schedule_with_warmup(optimizer=optimizer,
#                                                                 num_warmup_steps=warmup_steps // args.gradient_accumulation,
#                                                                 num_training_steps=total_steps // args.gradient_accumulation + 1,
#                                                                 power=-0.5)
writer = SummaryWriter(args.log_dir)

for epoch in range(1, args.epochs + 1):
  model.train()
  torch.cuda.empty_cache()
  # with trange(len(train_dataloader)) as t:
  # t.set_description('Epoch: {}/{}'.format(epoch, args.epochs))
  for step, batch in enumerate(train_dataloader):
    loss = model_train_batch(model, batch)
    train_epoch_loss += loss.item()
    train_total_steps += 1
    # t.set_postfix(loss=train_epoch_loss / train_total_steps)
    # training_loss_cache.append(train_epoch_loss / train_total_steps)

    if train_total_steps % 20 == 0:
      writer.add_scalar('Train/Loss', loss.item(), train_total_steps)
      writer.add_scalar('Train/Learning rate', optimizer_scheduler.get_last_lr()[0], train_total_steps)

    loss = loss / args.gradient_accumulation
    loss.backward()

    if (step + 1) % args.gradient_accumulation == 0:
      optimizer.step()
      optimizer_scheduler.step()
      optimizer.zero_grad()

    if train_total_steps % eval_steps == 0:
      model.eval()
      torch.cuda.empty_cache()
      eval_result, result_str = model_evaluation_with_metric(concept_set=valid_data['combined_concepts'], model=model,
                                                             tokenizer=tokenizer,
                                                             gen_data_type=args.evaluation_data_type,
                                                             model_save_path=args.model_save_path)
      if eval_result['Bleu_4: '] >= pre_result:
        pre_result = eval_result['Bleu_4: ']
        torch.save(model.state_dict(), args.model_save_path)
      print('=' * 80)
      print('Epoch: {},  evaluating model...'.format(train_total_steps / epoch_step_num))
      print(result_str)
      print('=' * 80)
      for metric in metrics_str:
        if metric != 'CIDEr: ':
          writer.add_scalar('Validation/' + metric, eval_result[metric] * 100, train_total_steps)
        else:
          writer.add_scalar('Validation/' + metric, eval_result[metric] * 10, train_total_steps)
      torch.cuda.empty_cache()
      model.train()
