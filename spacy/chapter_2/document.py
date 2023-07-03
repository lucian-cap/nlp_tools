import spacy
from spacy.tokens import Doc

#Documents can be made directly using the nlp variable or manually
nlp = spacy.blank('en')

#to make the Document manually we need a vocab, words, spaces
words = ['spaCy', 'is', 'cool', '!']
spaces = [True, True, False, False]

doc = Doc(nlp.vocab, words=words, spaces=spaces)
print(doc.text)