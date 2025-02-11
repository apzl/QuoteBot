import os
import argparse
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import torch.nn.functional as F
from tqdm import trange



def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (batch size x vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(dim=1, index=sorted_indices, src=sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits


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


def generate(model, tokenizer, tag, length, num_samples):
	tag_tkn = tokenizer.additional_special_tokens_ids[0]
	quote_tkn = tokenizer.additional_special_tokens_ids[1]

	input_ids = [tag_tkn] + tokenizer.encode(tag)

	segments = [quote_tkn] * 64

	segments[:len(input_ids)] = [tag_tkn] * len(input_ids)

	input_ids += [quote_tkn]

	# Move the model back to the CPU for inference:
	model.to(torch.device('cpu'))

	# Generate 10 samples of max length 50
	generated = sample_sequence(model, length,context=input_ids, num_samples = num_samples, segments_tokens=segments)

	print('\n\n--- Generated Slogans ---\n')
	for g in generated:
	  quote = tokenizer.decode(g.squeeze().tolist())
	  quote = quote.split('<|endoftext|>')[0].split('<quote>')[1]
	  print(quote)



if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Arguments for inferencing Text Augmentation model')

	parser.add_argument('--model_name', default='bot.pt', type=str, action='store', help='Name of the model file')
	parser.add_argument('--length', type=int, default=30, action='store', help='Length of generated outputs')
	parser.add_argument('--sentences', type=int, default=1, action='store', help='Number of sentences in outputs')
	parser.add_argument('--tag', type=str, action='store', help='Label for which to produce text')
	args = parser.parse_args()

	SENTENCES = args.sentences
	LENGTH = args.length
	MODEL_NAME = args.model_name
	TAG = args.tag

	TOKENIZER, MODEL = load_models(MODEL_NAME)

	generate(MODEL, TOKENIZER, TAG, LENGTH, SENTENCES)