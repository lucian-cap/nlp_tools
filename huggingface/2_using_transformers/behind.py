from transformers import pipeline
from transformers import AutoTokenizer
from transformers import AutoModel, AutoModelForSequenceClassification
import torch

torch.set_printoptions(sci_mode = False)

text_input = ['I love using machine learning models to solve real-world problems.',
              'Cats do not enjoy wet buckets, only dry ones filled with kibble.']
checkpoint = 'distilbert-base-uncased-finetuned-sst-2-english'

#Example of full pipeline using sentiment analysis
classifier = pipeline('sentiment-analysis')
output     = classifier(text_input)
print('Text run through sentiment analysis: ', output)


#Breaking down the pipeline
#we can specify the model checkpoint to use for loading the tokenizer 
tokenizer  = AutoTokenizer.from_pretrained(checkpoint)
model      = AutoModel.from_pretrained(checkpoint)

inputs = tokenizer(text_input, padding = True, truncation = True, return_tensors = 'pt')
print('Tokenization example: \n\tText Input: ', text_input, '\n\tTokenizer Output: ', inputs)

#model outputs last hidden state, usually in the shape of (Batch size, sequence length, hidden size)
outputs = model(**inputs)
print('Shape of models final hidden state: ', outputs.last_hidden_state.shape) #or outputs['last_hidden_state'] or even using index outputs[0]

#this output hidden state is sent to the model head to compute the final output
#   choosing which head can be as simple as appending the task to the AutoModel
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
outputs = model(**inputs)
print('Shape of AutoModel output for sequence classification: ', outputs.logits.shape)

#transformers models outputs the logits, so the last activation function still needs to be applied if normalized scores are needed
print('Raw, unnormalized scores: ', outputs.logits)
predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
print('Normalized scores: ', predictions)

#we can identify the what labels are which ids using the model config
print('ID -> Label link: ', model.config.id2label)