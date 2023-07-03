#custom attributes are available via ._ property, need to be registered on the global Doc, Token, & Span classes from spacy.tokens
#   3 types of extensions: attribute, property, & method
import spacy
from spacy.tokens import Doc, Token, Span

#to register a custom attribute use the set_extension method
Doc.set_extension('title', default = None)
Token.set_extension('is_blue', default = False)
Span.set_extension('has_color', default = False)

nlp = spacy.load('en_core_web_sm')

#attribute extensions: set a default value that can be overwritten
doc = nlp('The sky is blue')
doc[3]._.is_blue = True

#property extensions: define a getter & optional setter functions
#   getting is only called when you retrieve the attribute, computing the value dynamically
def get_is_color(token: Token):
    colors = ['red', 'yellow', 'blue']
    return token.text in colors

Token.set_extension('is_color', getter = get_is_color)

doc = nlp('The sky is blue.')
print(doc[3]._.is_color, '-', doc[3].text)

#method extensions: make the extension attribute a callable method, computing attribute values dynamically
#   can take arguments to change behavior, but fist arg is also the object itself
def has_token(doc: Doc, token_text: str):
    in_doc = token_text in [token.text for token in doc]
    return in_doc

Doc.set_extension('has_token', method = has_token)

doc = nlp('The sky is blue.')
print(doc._.has_token('blue'), '- blue')
print(doc._.has_token('cloud'), '- cloud')