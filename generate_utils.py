import torch

@torch.no_grad()
def generate_sentences(concepts, model, tokenizer, max_sentences_length=None):
  model.eval()
  inputs = tokenizer(concepts, padding=True, truncation=True, return_tensors='pt')
  inputs.to(model.device)
  beam_output = model.generate(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'],
                               max_length=max_sentences_length, num_beams=5, early_stopping=True,
                               no_repeat_ngram_size=2)
  gen_text = []
  for ids in beam_output:
    sentence = tokenizer.decode(ids, skip_special_tokens=True)
    gen_text.append(sentence)
  return gen_text
