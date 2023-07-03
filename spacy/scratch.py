import spacy

spacy.prefer_gpu()
nlp = spacy.load('en_core_web_lg')
doc = nlp('Apple is looking at buying U.K. startup for $1 billion')
for token in doc:
    print(token, token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
          token.shape_, token.is_alpha, token.is_stop, token.has_vector)


for ent in doc.ents:
    print(ent.text, ent.start_char, ent.end_char, ent.label_)

doc1 = nlp('I like salty fries and hamburgers.')
doc2 = nlp('Fast food tastes very good.')

print(doc1, '<->', doc2, doc1.similarity(doc2))
french_fries = doc1[2:4]
burgers = doc1[5]
print(french_fries, '<->', burgers, french_fries.similarity(burgers))