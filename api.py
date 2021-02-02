import os
import argparse
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, top_k_top_p_filtering
import numpy as np
import torch.nn.functional as F
from fastapi import FastAPI

app = FastAPI()

@app.get("/generate/{tag}")
async def preditc(tag):
    return generate(tag)
	  




def load_models(model_name):
  print ('Loading Trained GPT-2 Model')
  tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')
  model = GPT2LMHeadModel.from_pretrained('distilgpt2')
  model_path = model_name
  SPECIAL_TOKENS_DICT = {
    'pad_token': '<pad>',
    'additional_special_tokens': ['<tag>', '<quote>'],
	}
  tokenizer.add_special_tokens(SPECIAL_TOKENS_DICT)
  model.resize_token_embeddings(len(tokenizer))
  model.load_state_dict(torch.load(model_path))
  return tokenizer, model

# From HuggingFace, adapted to work with the tag/quote separation:
def sample_sequence(model, length, context, segments_tokens=None, num_samples=1, temperature=1, top_k=20, top_p=0.8, repetition_penalty=5,
                    device='cpu'):
    context = torch.tensor(context, dtype=torch.long, device=device)
    context = context.unsqueeze(0).repeat(num_samples, 1)
    generated = context

    with torch.no_grad():
        for _ in trange(length):

            inputs = {'input_ids': generated}
            if segments_tokens != None:
              inputs['token_type_ids'] = torch.tensor(segments_tokens[:generated.shape[1]]).unsqueeze(0).repeat(num_samples, 1)


            outputs = model(**inputs)  # Note: we could also use 'past' with GPT-2/Transfo-XL/XLNet/CTRL (cached hidden-states)
            next_token_logits = outputs[0][:, -1, :] / (temperature if temperature > 0 else 1.)

            # repetition penalty from CTRL (https://arxiv.org/abs/1909.05858)
            for i in range(num_samples):
                for _ in set(generated[i].tolist()):
                    next_token_logits[i, _] /= repetition_penalty
                
            filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
            if temperature == 0: # greedy sampling:
                next_token = torch.argmax(filtered_logits, dim=-1).unsqueeze(-1)
            else:
                next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
            generated = torch.cat((generated, next_token), dim=1)
    return generated

def generate(tag, length=30, num_samples=1):
    model,tokenizer=load_models('bot.pt')
    tag_tkn = tokenizer.additional_special_tokens_ids[0]
    quote_tkn = tokenizer.additional_special_tokens_ids[1]
    input_ids = [tag_tkn] + tokenizer.encode(tag)
    segments = [quote_tkn] * 64
    segments[:len(input_ids)] = [tag_tkn] * len(input_ids)
    input_ids += [quote_tkn]
    model.to(torch.device('cpu'))
    generated = sample_sequence(model, length,context=input_ids, num_samples = num_samples, segments_tokens=segments)
    quote = tokenizer.decode(generated.squeeze().tolist())
    quote = quote.split('<|endoftext|>')[0].split('<quote>')[1]
    return quote