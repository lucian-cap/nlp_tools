from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification, get_scheduler
import evaluate
from torch.utils.data import DataLoader
from torch.optim import AdamW
import torch
from tqdm.auto import tqdm

raw_datasets = load_dataset('glue', 'mrpc')
checkpoint = 'bert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

def tokenize_function(example):
    return tokenizer(example['sentence1'], example['sentence2'], truncation = True)

tokenized_datasets = raw_datasets.map(tokenize_function, batched = True)
data_collator = DataCollatorWithPadding(tokenizer = tokenizer)



#Remove columns the model isn't expecting, rename label to labels to match model expectations, and set format
tokenized_datasets = tokenized_datasets.remove_columns(['sentence1', 'sentence2', 'idx'])
tokenized_datasets = tokenized_datasets.rename_column('label', 'labels')
tokenized_datasets.set_format('torch')
print(tokenized_datasets.column_names)

#Prepare the two dataloaders to be used when iterating over batched input
train_dataloader = DataLoader(tokenized_datasets['train'], 
                              shuffle = True,
                              batch_size = 8,
                              collate_fn = data_collator)

eval_dataloader = DataLoader(tokenized_datasets['validation'], 
                              shuffle = True,
                              batch_size = 8,
                              collate_fn = data_collator)

#Generate each batch and examine their shape
for batch in train_dataloader:
    break
print('Example shapes of dynamically padded batched inputs: ', {k: v.shape for k, v in batch.items()})

#Load model and do forward pass to confirm it works
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels = 2)
outputs = model(**batch)
print(outputs.loss, outputs.logits.shape)

#Define optimizer and learning rate scheduler for training
optimizer = AdamW(model.parameters(), lr=5e-5)
num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)
lr_schedular = get_scheduler('linear',
                             optimizer = optimizer,
                             num_warmup_steps = 0,
                             num_training_steps = num_training_steps)
print('Examine the trainer/scheduler info: ', num_training_steps)

#First check out the GPU if its available, otherwise CPU
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)
print(device)

#Create a progress bar while training and put the model in training model
progress_bar = tqdm(range(num_training_steps))
model.train()

#For each batch in a epoch, for each epoch, process data through the model, & compute gradients over weights
for epoch in range(num_epochs):
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        #Do backward pass over model parameters, step once through learning rate, and zero out gradients
        optimizer.step()
        lr_schedular.step()
        optimizer.zero_grad()
        progress_bar.update(1)

#Metrics can accumulate over batches using the add_batch function, then finalized using compute()
metric = evaluate.load('glue', 'mrpc')
model.eval()
for batch in eval_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    
    #Since we're not updating weights we don't need to accumulate gradients
    with torch.no_grad():
        outputs = model(**batch)

    #get the scores from the model and use them to prepare metric computation
    logits = outputs.logits
    preds = torch.argmax(logits, dim = -1)

    metric.add_batch(predictions = preds, references = batch['labels'])

#Collate scores across batches
scores = metric.compute()
print('Example scores from evaluating: ', scores)

