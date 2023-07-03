#pipeline components are run in order, custom components can be used to add custom metadata or update built-in attributes
#a component is a function that takes a Document obj, modifies it, & returns it
import spacy
from spacy.language import Language
from spacy.tokens import Doc

#custom components should be decorated with Language.component to register it with spaCy and tell it how it should be called
@Language.component('custom_component')
def custom_component_function(doc: Doc):
    print(f'Doc length: {len(doc)}')
    return doc

nlp = spacy.load('en_core_web_sm')

#specify where to add the component using first, last, before, or after kwargs
#   first & last take a bool value, before & after take the name of a existing component
nlp.add_pipe('custom_component', first = True)

print(f'Pipeline: {nlp.pipe_names}')

doc = nlp('Hello world!')