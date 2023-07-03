from transformers import AutoModelForSequenceClassification, get_scheduler, AutoTokenizer, DataCollatorWithPadding
from accelerate import Accelerator
from torch.optim import AdamW
from torch.utils.data import DataLoader
from datasets import load_dataset
import torch
from tqdm.auto import tqdm

checkpoint = 'bert-base-uncased'

model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
optimizer = AdamW(model.parameters(), lr=3e-5)
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
data_collator = DataCollatorWithPadding(tokenizer = tokenizer)

accelerator = Accelerator()

# device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# model.to(device)

raw_datasets = load_dataset('glue', 'mrpc')

def tokenize_function(example):
    return tokenizer(example['sentence1'], example['sentence2'], truncation = True)

tokenized_datasets = raw_datasets.map(tokenize_function, batched = True)


train_dataloader = DataLoader(tokenized_datasets['train'], 
                              shuffle = True,
                              batch_size = 8,
                              collate_fn = data_collator)

eval_dataloader = DataLoader(tokenized_datasets['validation'], 
                              shuffle = True,
                              batch_size = 8,
                              collate_fn = data_collator)

train_dataloader, eval_dataloader, model, optimizer = accelerator.prepare(train_dataloader, eval_dataloader, model, optimizer)

num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)

progress_bar = tqdm(range(num_training_steps))

model.train()
for epoch in range(num_epochs):
    for batch in train_dataloader:
        # batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        # loss.backward()
        accelerator.backward(loss)

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)