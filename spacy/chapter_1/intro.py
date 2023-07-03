import spacy

#create a blank English pipeline
nlp = spacy.blank('en')

#Doc object created by processing a string of text with the nlp object
doc = nlp('It costs $5.')

#Documents can be iterated upon to view their tokens, indexed, or sliced (which returns a Span, only a view of the Doc)
print('Iteration:')
for token in doc:
    print(token.text)

print(f'Indexing: {doc[1].text}')
print(f'Slicing: {doc[1:3].text}')

#Lexical attributes, refer to the entry in the vocab & don't depend on the token's context
print('Index: ', [token.i for token in doc])
print('Text: ', [token.text for token in doc])
print('is_alpha: ', [token.is_alpha for token in doc])
print('is_punct: ', [token.is_punct for token in doc])
print('like_num: ', [token.like_num for token in doc])