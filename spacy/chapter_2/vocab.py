import spacy

nlp = spacy.blank('en')

#shared strings are stored in a bidirectional lookup table
nlp.vocab.strings.add('coffee')
coffee_hash = nlp.vocab.strings['coffee']
coffee_str = nlp.vocab.strings[coffee_hash]

#Document obj also exposes the vocab lookup table
nlp = spacy.load('en_core_web_sm')
doc = nlp('I love coffee')
print(f'Hash value: {doc.vocab.strings["coffee"]}')

#Lexeme obj is a entry in the vocab, contains context independent info
lexeme = nlp.vocab['coffee']
print(lexeme.text, lexeme.orth, lexeme.is_alpha)