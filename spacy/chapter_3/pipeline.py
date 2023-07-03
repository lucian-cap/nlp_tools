#all pipelines contain a config.cfg that defines the language and components in the pipeline
import spacy

nlp = spacy.load('en_core_web_lg')
print(f'Pipeline component names: {nlp.pipe_names}')
print(f'Component name, type: {nlp.pipeline}')