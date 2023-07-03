from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding

#Download and cache the dataset locally
raw_datasets = load_dataset('glue', 'mrpc')
print('Example dataset info: ', raw_datasets)

#Access a split and record using dictionary & list indexing
print('Example record: ', raw_datasets['train'][0])

#Examing information about the dataset
print('Dataset split features: ', raw_datasets['train'].features)

#Process the sequences individually and as a batch
checkpoint = 'bert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
tokenized_sen_1 = tokenizer(raw_datasets['train'][42]['sentence1'])
tokenized_sen_2 = tokenizer(raw_datasets['train'][42]['sentence2'])
input_ids = tokenizer(raw_datasets['train'][42]['sentence1'], raw_datasets['train'][42]['sentence2'])

print('Sentence 1: ', raw_datasets['train'][42]['sentence1'])
print('Sentence 2: ', raw_datasets['train'][42]['sentence2'])

#The output tensors are different because additional values are needed for padding and to separate input
print('Tokenized sentence 1: ', tokenized_sen_1)
print('Tokenized sentence 2: ', tokenized_sen_2)
print('Batch tokenized: ', input_ids)

#We can preprocess the whole dataset at once like the following, assuming enough RAM to store it
tokenized_dataset = tokenizer(raw_datasets['train']['sentence1'], 
                              raw_datasets['train']['sentence2'],
                              padding = True,
                              truncation = True)
print('Dataset tokenization w/out tricks is finished.')

#We can also keep the data as a dataset using map, which applies a function to each element of the dataset
def tokenize_function(example):
    return tokenizer(example['sentence1'], example['sentence2'], truncation = True)

#using the batched flag allows for much faster preprocessing, also has num_proc to multiprocess if not using a fast tokenizer
#   the keys returned by the function used are added to the original dataset, however these could be changed or removed
tokenized_datasets = raw_datasets.map(tokenize_function, batched = True)
print('Dataset tokenization w/mapping finished: ', tokenized_datasets)

#Prepare datacollator using the tokenizer for collage function, only useful if not on TPU
data_collator = DataCollatorWithPadding(tokenizer = tokenizer)
samples = {k: v for k, v in tokenized_datasets['train'][:5].items() if k not in ['idx', 'sentence1', 'sentence2']}
print('Length of sequences for dynamic padding: ', [len(x) for x in samples['input_ids']])
