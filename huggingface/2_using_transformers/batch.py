from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

#Initialize the checkpointed tokenizer and model
checkpoint = 'distilbert-base-uncased-finetuned-sst-2-english'
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model     = AutoModelForSequenceClassification.from_pretrained(checkpoint)

sequence  = 'The cat found a large bucket and immediately climbed inside it.'

#Breaking open the tokenizer and trying to manually do each step
tokens    = tokenizer.tokenize(sequence)
ids       = tokenizer.convert_tokens_to_ids(tokens)
input_ids = torch.tensor(ids)

#This should fail because by manually tokenizing input we only have 1-dim,
#   but the model is expecting 2
try:
    output = model(input_ids)
except Exception as e:
    print('Tried inference but failed with following exception:\n', e)

#Passing the sequence through the tokenizer gives it another dimension for number of sequences in batch
#   it might also add special tokens to each of the sequences like the BOS & EOS
tokenized_input = tokenizer(sequence, return_tensors = 'pt')
print('Sequence passed through the tokenizer: ', tokenized_input['input_ids'])

#We can recreate this effect pretty simply
input_ids = torch.tensor([ids])
output    = model(input_ids)
print('Example from adding the dimension manually: ', output.logits)

#Examine the padding token the tokenizer uses
pad_id = tokenizer.pad_token_id
print('Tokenizer uses padding ID: ', pad_id)

sequences = [sequence, 'This is another sentence about cats and buckets to show how the model & tokenizer work with sequences of different length.']

#Example of padding a batch of input into model, options for padding are longst, max_length, or do_not_pad
input_ids = tokenizer(sequences, 
                      padding = 'longest',
                      return_tensors = 'pt',
                      max_length = 10,
                      truncation = True)

print(input_ids)