#PhraseMatcher like regex or keyword search but with access to the tokens
#   more efficient & faster than Matcher, better for large word lists
#   takes Document objs as patterns
import spacy
from spacy.matcher import PhraseMatcher

nlp = spacy.blank('en')

matcher = PhraseMatcher(nlp.vocab)

pattern = nlp('Golden Retriever')
matcher.add('DOG', [pattern])
doc = nlp('I have a Golden Retriever')

for match_id, start, end in matcher(doc):
    span = doc[start:end]
    print('Matched span:', span.text)