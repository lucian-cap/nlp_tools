#spaCy supports updating existing models & training new models
import spacy
from spacy.tokens import Span, DocBin
import random

#spaCy can be updated from data stored as Doc objs

nlp = spacy.blank('en')

#example with entity span we want to predict
doc1 = nlp('iPhone X is coming')
doc1.ents = [Span(doc1, 0, 2, label = 'GADGET')]

#negative example to get a "full" distribution
doc2 = nlp('I need a new phone! Any tips?')

docs = [doc1 if random.randint(0, 1) == 0 else doc2 for _ in range(100)]

#as usual we want to shuffle and split our dataset into a training & validation set, this does a 50/50 split which isn't what we'll usually want
random.shuffle(docs)
train_docs = docs[:len(docs) // 2]
val_docs = docs[len(docs) // 2: ]

#DocBin obj is a container for efficiently storing & serializing Doc objs, typically use .spacy for saved files
#   using the .to_disk() from DocBin is faster & more efficient than using binary serialization protocols like pickle
train_docbin = DocBin(docs = train_docs)
val_docbin = DocBin(docs = val_docs)

train_docbin.to_disk('./train.spacy')
val_docbin.to_disk('./val.spacy')

#spaCy's convert command can be used to automatically convert files in other formats (e.g. .conll, .conllu, .iob) to spaCy's binary format