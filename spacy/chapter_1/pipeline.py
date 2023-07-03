import spacy

#loads a predefined pipeline after downloading it
nlp = spacy.load('en_core_web_lg')

#process some text
doc = nlp('She ate the pizza')

#iterate over the tokens
for token in doc:
    
    #attributes that return strings usually have a trailing underscore, attributes w/out return a integer ID value (hash of the str)
    #   .pos_ : part-of-speech
    #   .dep_ : dependency label
    #   .head : returns syntactic head token, kind of the parent token this word is attached to
    print(token.text, token.pos_, token.dep_, token.head)


print('Demo named entity recognition:')

#Named-entity recognition demo
ner_doc = nlp('Apple is looking at buying U.K. startup for $1 billion')

#ents attribute of a Document returns a iterator of Span objects accessing the named entities predicted
for ent in ner_doc.ents:
    print(ent.text, ent.label_)

#helper function to get definitions of most common tags & labels
print(spacy.explain('GPE'))
print(spacy.explain('NNP'))
print(spacy.explain('dobj'))