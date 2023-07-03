#nlp.pipe process text as a stream and yields Doc objs, so it speeds up processing a lot of text
import spacy
import random

nlp = spacy.load('en_core_web_sm')

summary = 'The Lord of the Rings is a saga set in the fictional world of Middle-earth. The story follows the hobbit Frodo Baggins as he and the Fellowship embark on a quest to destroy the One Ring, to ensure the destruction of its maker, the Dark Lord Sauron. The Fellowship eventually splits up and Frodo continues the quest with his loyal companion Sam and the treacherous Gollum. The film begins with a summary of the prehistory of the ring of power.'
LOTS_OF_TEXT = [summary for _ in range(100)]


# bad: docs = [nlp(text) for text in LOTS_OF_TEXT]
docs = list(nlp.pipe(LOTS_OF_TEXT))

#supports passing in tuples of (text, context) if as_tuples=True, useful for extra metadata
LOTS_OF_TEXT = [(text, {'id': i, 'page_number': random.randint(0, len(LOTS_OF_TEXT))}) for i, text in enumerate(LOTS_OF_TEXT)]
for doc, context in nlp.pipe(LOTS_OF_TEXT, as_tuples=True):
    print(doc.text, context['page_number'])