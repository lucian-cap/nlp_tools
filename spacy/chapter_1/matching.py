import spacy

#the matcher works with Doc & Token objects instead of only strings like regex, can use lexical attributes along with text
from spacy.matcher import Matcher

nlp = spacy.load('en_core_web_lg')

#init the matcher with the shared vocab
matcher = Matcher(nlp.vocab)

#match patterns are lists of dictionaries, each dictionary describes one token
exact_pattern = [{'TEXT': 'iPhone'}, {'TEXT': 'X'}]
lexical_pattern = [{'LOWER': 'iphone'}, {'LOWER': 'x'}]
attribute_pattern = [{'LEMMA': 'buy'}, {'POS': 'NOUN'}]

#add the patterns to the matcher
matcher.add('EXACT_IPHONE_PATTERN', [exact_pattern])
matcher.add('LEXICAL_IPHONE_PATTERN', [lexical_pattern])
matcher.add('ATTRIBUTE_PATTERN', [attribute_pattern])

doc = nlp('Upcoming iPhone X release date leaked')

#returns a list of tuples, each consisting of 3 values
#   (match ID, start index, end index)
matches = matcher(doc)

for match_id, start, end in matches:
    print(nlp.vocab[match_id].text, doc[start:end].text)

#operators & quantifiers define how often a token should be matched, added using OP key
#   !: negates the token, matched 0 times
#   ?: token is optional, matches it 0 or 1 times
#   +: matches a token 1 or more times
#   *: matches 0 or more times
pattern = [
    {'LEMMA': 'buy'},
    {'POS': 'DET', 'OP': '?'}, #optional: match 0 or 1 times
    {'POS': 'NOUN'}
]

doc = nlp('I bought a smartphone. Now I\'m buying apps.')

matcher.add('PATTERN', [pattern])
matches = matcher(doc)

for match_id, start, end in matches:
    print(nlp.vocab[match_id].text, doc[start:end].text)