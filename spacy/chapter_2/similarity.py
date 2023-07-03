import spacy

#Doc, Span, & Token objs all support similarity comparison
#   score is between 0 to 1, requires a pipeline with word vectors
nlp = spacy.load('en_core_web_md')

#by default the similarity is a cosine similarity measurement
doc1 = nlp('I like fast food')
doc2 = nlp('I like pizza')
print(doc1.similarity(doc2))

doc = nlp('I like pizza and pasta')
token1 = doc[2]
token2 = doc[4]
print(token1.similarity(token2))