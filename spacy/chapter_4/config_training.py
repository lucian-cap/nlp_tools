#spaCy uses the config.cfg config file as the single source of truth for all settings
#   defines how to initialize the nlp obj, what pipeline components to use, all settings for training, and hyperparameters

#can reference Python function using the '@' notation

#quickstart widget in the spaCy docs helps generate a config, or spaCy's "spacy init config" command

#"spacy train" command will train a pipeline using the config.cfg, training, & validation data
#   also allows you to overrides settings defined in the config.cfg file

#within each epoch of training spaCy outputs accuracy scores every 200 examples by default (can change this)
#training will run until the model stops improving and exits automatically, the last trained pipeline and the one with the best score
#   will be saved as model-last and model-best respectively

#"spacy package" command can take a custom pipeline and generate a Python package that can be exported for use elsewhere