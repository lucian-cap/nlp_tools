from transformers import AutoTokenizer

sequence = 'The cat found a large bucket and immediately climbed inside it.'

#Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')

#Example output, creates token ids, token types, and attention mask tensors
output = tokenizer(sequence)
print('Example of tokenizer output: ', output)

#Example of how to save, same as saving a model
tokenizer.save_pretrained('./2_using_transformers/saving_model_test/')

#Tokenization is done in 2 steps, first is actually tokenizing the input
tokens = tokenizer.tokenize(sequence)
print('Example of tokenization (w/out encoding): ', tokens)

#Encoding the tokens is the second step
ids = tokenizer.convert_tokens_to_ids(tokens)
print('Example of ID conversion: ', ids)

#Token ids can then be turned back into text
decoded_text = tokenizer.decode(ids)
print('Example of decoding IDs into text: ', decoded_text)