import csv
import os
import argparse
import torch
from tqdm import tqdm
import math, random
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW
import warnings
warnings.filterwarnings('ignore')

class MyDataset(Dataset):
  def __init__(self, data_file_name, tokenizer, seq_length=64):
    super().__init__()
    data_path = os.path.join(data_file_name)
    self.examples = []
    
    
    tag_tkn = tokenizer.additional_special_tokens_ids[0]
    quote_tkn = tokenizer.additional_special_tokens_ids[1]
    pad_tkn = tokenizer.pad_token_id
    eos_tkn = tokenizer.eos_token_id

    with open(data_path) as csv_file:
      csv_reader = csv.reader(csv_file, delimiter=',')
      
      for row in csv_reader:
        tag = [tag_tkn] + tokenizer.encode(row[1], max_length=seq_length//2-1)
        quote = [quote_tkn] + tokenizer.encode(row[2],max_length=seq_length//2-2) + [eos_tkn]
        tokens = tag + quote + [pad_tkn] * ( seq_length - len(tag) - len(quote) )
        segments = [tag_tkn] * len(tag) + [quote_tkn] * ( seq_length - len(tag) )
        labels = [-100] * (len(tag)+1) + quote[1:] + [-100] * ( seq_length - len(tag) - len(quote) )
        self.examples.append((tokens, segments, labels))

        
  def __len__(self):
    return len(self.examples)
      
  def __getitem__(self, item):
    
    return torch.tensor(self.examples[item])



def get_data_loader(data_file_name,tokenizer):
	dataset = MyDataset(data_file_name,tokenizer)
	# Create data indices for training and validation splits:

	indices = list(range(len(dataset)))

	random.seed(42)
	random.shuffle(indices)

	split = math.floor(0.1 * len(dataset))
	train_indices, val_indices = indices[split:], indices[:split]

	# Build the PyTorch data loaders:

	train_sampler = SubsetRandomSampler(train_indices)
	val_sampler = SubsetRandomSampler(val_indices)

	train_loader = DataLoader(dataset, batch_size=32, sampler=train_sampler)
	val_loader = DataLoader(dataset, batch_size=64, sampler=val_sampler)
	return train_loader, val_loader



def fit(model, optimizer, train_dl, val_dl, epochs, device):

  for i in range(epochs):

    print('\n--- Starting epoch #{} ---'.format(i))

    model.train()

    # These 2 lists will keep track of the batch losses and batch sizes over one epoch:
    losses = []
    nums = []

    for xb in tqdm(train_dl, desc="Training"):
      # Move the batch to the training device:
      inputs = xb.to(device)

      # Call the model with the token ids, segment ids, and the ground truth (labels)
      outputs = model(inputs[:,0,:], token_type_ids=inputs[:,1,:], labels=inputs[:,2,:])
      
      # Add the loss and batch size to the list:
      loss = outputs[0]
      losses.append(loss.item())
      nums.append(len(xb))

      loss.backward()

      optimizer.step()
      model.zero_grad()

    # Compute the average cost over one epoch:
    train_cost = np.sum(np.multiply(losses, nums)) / sum(nums)


    # Now do the same thing for validation:

    model.eval()
    
    with torch.no_grad():
      losses = []
      nums = []

      for xb in tqdm(val_dl, desc="Validation"):
        inputs = xb.to(device)
        outputs = model(inputs[:,0,:], token_type_ids=inputs[:,1,:], labels=inputs[:,2,:])
        losses.append(outputs[0].item())
        nums.append(len(xb))

    val_cost = np.sum(np.multiply(losses, nums)) / sum(nums)

    print('\n--- Epoch #{} finished --- Training cost: {} / Validation cost: {}'.format(i, train_cost, val_cost))
  return model

def save_model(model, name):
	"""
	Summary:
		Saving model to the Disk
	Parameters:
		model: Trained model object
		name: Name of the model to be saved
	"""
	print ("Saving model to Disk")
	torch.save(model.state_dict(), f"{name}.pt")
	return

def load_models():
	"""
	Summary:
		Loading Pre-trained model
	"""
	print ('Loading/Downloading GPT-2 Model')
	tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
	model = GPT2LMHeadModel.from_pretrained('gpt2-medium')
	return tokenizer, model



if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Arguments for training Text Augmentation model')

	parser.add_argument('--epoch', default= 2,type=int, action='store', help='Number of epochs to run')
	parser.add_argument('--model_name', default='bot.pt', type=str, action='store', help='Name of the model file')
	parser.add_argument('--data_file', default='bot.csv', type=str, action='store', help='Name of the data file')
	parser.add_argument('--batch', type=int, default=32, action='store', help='Batch size')
	parser.add_argument('--learning_rate', default=3e-5, type=float, action='store', help='Learning rate for the model')
	parser.add_argument('--max_len', default=200, type=int, action='store', help='Maximum length of sequence')
	args = parser.parse_args()

	BATCH_SIZE = args.batch
	EPOCHS = args.epoch
	LEARNING_RATE = args.learning_rate
	MAX_SEQ_LEN = args.max_len
	MODEL_NAME = args.model_name
	DATA_FILE = args.data_file

	TOKENIZER, MODEL = load_models()
	SPECIAL_TOKENS_DICT = {
    'pad_token': '<pad>',
    'additional_special_tokens': ['<tag>', '<quote>'],
	}
	TOKENIZER.add_special_tokens(SPECIAL_TOKENS_DICT)
	MODEL.resize_token_embeddings(len(TOKENIZER))
	TRAIN_LOADER, VAL_LOADER = get_data_loader(DATA_FILE, TOKENIZER)

	DEVICE = 'cpu'
	if torch.cuda.is_available():
		DEVICE = 'cuda'

	model = MODEL.to(DEVICE)
	optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
	model = fit(model, optimizer, TRAIN_LOADER, VAL_LOADER, EPOCHS, DEVICE):
	save_model(model, MODEL_NAME)


