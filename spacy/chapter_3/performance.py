import spacy

nlp = spacy.load('en_core_web_sm')
text = 'The Lord of the Rings is a saga set in the fictional world of Middle-earth. The story follows the hobbit Frodo Baggins as he and the Fellowship embark on a quest to destroy the One Ring, to ensure the destruction of its maker, the Dark Lord Sauron. The Fellowship eventually splits up and Frodo continues the quest with his loyal companion Sam and the treacherous Gollum. The film begins with a summary of the prehistory of the ring of power.'


#if you just need a tokenized Doc obj, use nlp.make_doc to just invoke the tokenizer and not the entire pipeline
#bad: doc = nlp(text)
doc = nlp.make_doc(text)

#specific components can also be enabled/disabled using select_pipes
#   kwargs of enable/disable take a list of component names as strings
#   after the with block the disabled components are autmatically restored
with nlp.select_pipes(disable=['tagger', 'parser']):
    doc = nlp(text)
    print(doc.ents)