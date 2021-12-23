import torch

def move_batch_to_device(batch, device):
  """

  :param batch:
  :param device:
  :return:
  """
  batch = tuple(x.to(device) for x in batch)
  return batch

def model_train_batch(model, batch, max_grad_norm=1.):
  """

  :param model:
  :param batch:
  :param max_grad_norm:
  :return:
  """
  batch = move_batch_to_device(batch, model.device)
  b_input_ids, b_attention_mask, b_labels = batch
  output = model(input_ids=b_input_ids, attention_mask=b_attention_mask, labels=b_labels)
  loss = output.loss
  torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_grad_norm)
  # optimizer.step()
  # model.zero_grad()
  return loss

def set_weight_decay(model, full_fine_tune=True):
  no_decay = ['bias', 'gamma', 'beta']

  param_optimizer = list(model.named_parameters())
  if full_fine_tune:
    # split params into two grouped: decay and no_decay.
    grouped_params = [
      {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
       'weight_decay_rate': 0.00},
      {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
       'weight_decay_rate': 0.01}
    ]
  else:
    grouped_params = [{'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                       'weight_decay_rate': 0.00}]

  return grouped_params