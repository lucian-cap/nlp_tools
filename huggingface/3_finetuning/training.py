from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding, TrainingArguments, AutoModelForSequenceClassification, Trainer
import evaluate
import os
import numpy as np

PATH = os.getcwd() + '/3_finetuning/'

raw_datasets = load_dataset('glue', 'mrpc')
checkpoint = 'bert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

def tokenize_function(example):
    return tokenizer(example['sentence1'], example['sentence2'], truncation = True)

tokenized_datasets = raw_datasets.map(tokenize_function, batched = True)
data_collator = DataCollatorWithPadding(tokenizer = tokenizer)

#First we have to define all the hyperparameters used for training & evaluation, 
#   including directory where model is saved
training_args = TrainingArguments(PATH + 'test_trainer')
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels = 2) #warning is expected b/c new head is needed

#Using everything defined above we can now instantiate our trainer
trainer = Trainer(model,
                  training_args,
                  train_dataset = tokenized_datasets['train'],
                  eval_dataset = tokenized_datasets['validation'],
                  data_collator = data_collator,
                  tokenizer = tokenizer)

#And then run it
# trainer.train()

#Model can be evaluated rather simply, first get the predictions for the validation
predictions = trainer.predict(tokenized_datasets['validation'])
print('Confirming prediction shapes match expected: ', predictions.predictions.shape, predictions.label_ids.shape)

#Since the outputs of transformers are logits, they need a little extra to identify labels
preds = np.argmax(predictions.predictions, axis=-1)

#Load the metrics with the dataset as follows
metric = evaluate.load('glue', 'mrpc')
score = metric.compute(predictions = preds, references = predictions.label_ids)
print('Example scores: ', score)

#Should take a EvalPrediction object, named tuple with predictions & label_ids fields
def compute_metrics(eval_preds):
    metric = evaluate.load('glue', 'mrpc')
    logits, labels = eval_preds
    preds = np.argmax(logits, axis = -1)
    return metric.compute(predictions = preds, references = labels)

#Can now wrap up this metric computation with the trainer
training_args = TrainingArguments(PATH + 'test_trainer', 
                                  evaluation_strategy = 'epoch')
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels = 2)

#Loss & metrics will be reported at the end of each epoch using this trainer
trainer = Trainer(model,
                  training_args, 
                  train_dataset = tokenized_datasets['train'],
                  eval_dataset = tokenized_datasets['validation'],
                  data_collator = data_collator,
                  tokenizer = tokenizer,
                  compute_metrics = compute_metrics)

trainer.train()