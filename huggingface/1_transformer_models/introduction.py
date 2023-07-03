#Pipelines are the most utility in the transformers library
#   connects model with pre/post-processing so text can be input directly
from transformers import pipeline
import fire

NUM_RETURNS = 3
MAX_LENGTH  = 50
MIN_LENGTH  = 20

#the pipeline selects a particular model based on the task selected
def sentiment_analysis():
    sentiment_analysis = pipeline('sentiment-analysis')
    output             = sentiment_analysis(input('Input sentence to run sentiment analysis on (positive/negative classification): '))
    return output

#Zero-shot classification demo, allows us to specify labels outside of what a model has been pretrained on
def zero_shot_classification():
    zero_shot = pipeline('zero-shot-classification')
    labels    = input('Enter a list of classes, separated by spaces, to use for zero-shot classification (i.e. education, politics, business): ')
    labels    = labels.split(' ')

    if len(labels) < 2:
        print('Labels not formatted correctly. Using default classes of education, politics, and business.')
        labels = ['education', 'politics', 'business']

    output    = zero_shot(input('Input sentence to run zero-shot classification on: '),
                        candidate_labels = labels)
    
    return output

#Text generation
def text_generation(num_returns = NUM_RETURNS, max_len = MAX_LENGTH):
    #generator = pipeline('text-generation')

    #or this if we want to choose a particular model
    generator = pipeline('text-generation',
                         model = 'distilgpt2')

    output    = generator(input('Enter text to use for text generation: '),
                        num_return_sequences = num_returns,
                        max_length           = max_len)
    return output

#Using a pipeline to fill a mask, the mask token used is defined per model so we'll need to check on the Hub
def fill_mask(model_name = 'distilroberta-base', 
              num_returns = NUM_RETURNS):
    
    unmasker = pipeline('fill-mask', 
                        model = model_name)
    
    output   = unmasker(input('Enter a sentence to use for mask filling (be sure to correctly capitalize the mask token): '), 
                        top_k = num_returns)
    return output

#Named entity recognition pipeline to identify persons, locations, or orgs
#   the grouped_entities parameter groups parts of the sentence the model thinks is part of the same entity
def ner(agg_strat = 'simple'):
    ner = pipeline('ner', 
                   aggregation_strategy = agg_strat)
    
    output = ner(input('Enter sentence to run named-entity recognition on: '))

    return output

#Question answering, using context to extract the answer from
def question_answering():
    question_answerer = pipeline('question-answering')

    output = question_answerer(question = input('Enter your question: '),
                               context = input('Enter the context to extract the answer from: '))
    return output

#Summarization demo
def summarization(input_text = None):
    summarizer = pipeline('summarization')

    if not input_text:
        input_text = """Star Trek: The Next Generation (TNG) is an American science fiction television series created by Gene Roddenberry. It originally aired from September 28, 1987, to May 23, 1994, in syndication, spanning 178 episodes over seven seasons. The third series in the Star Trek franchise, it was inspired by Star Trek: The Original Series. Set in the latter third of the 24th century, when Earth is part of the United Federation of Planets, it follows the adventures of a Starfleet starship, the USS Enterprise (NCC-1701-D), in its exploration of the Alpha quadrant in the Milky Way galaxy.
        In the 1980s, Roddenberry—who was responsible for the original Star Trek, Star Trek: The Animated Series (1973-1974), and the first of a series of films—was tasked by Paramount Pictures with creating a new series in the franchise. He decided to set it a century after the events of his original series. The Next Generation featured a new crew: Patrick Stewart as Captain Jean-Luc Picard, Jonathan Frakes as William Riker, Brent Spiner as Data, Michael Dorn as Worf, LeVar Burton as Geordi La Forge, Marina Sirtis as Deanna Troi, Gates McFadden as Dr. Beverly Crusher, Denise Crosby as Tasha Yar, Wil Wheaton as Wesley Crusher, and a new Enterprise.
        Roddenberry, Maurice Hurley, Rick Berman, Michael Piller, and Jeri Taylor served as executive producers at various times throughout its production. The series was broadcast in first-run syndication with dates and times varying among individual television stations. Stewart's voice-over introduction during each episode's opening credits stated the starship's purpose:
        Space: the final frontier. These are the voyages of the starship Enterprise. Its continuing mission: to explore strange new worlds, to seek out new life and new civilizations, to boldly go where no one has gone before.
        The show was very popular, reaching almost 12 million viewers in its 5th season, with the series finale in 1994 watched by over 30 million viewers. Due to its success, Paramount commissioned Rick Berman and Michael Piller to create a fourth series in the franchise, Star Trek: Deep Space Nine, which launched in 1993. The characters from The Next Generation returned in four films: Star Trek Generations (1994), Star Trek: First Contact (1996), Star Trek: Insurrection (1998), and Star Trek: Nemesis (2002), and in the television series Star Trek: Picard (2020-2023). The series is also the setting of numerous novels, comic books, and video games. It received many accolades, including 19 Emmy Awards, two Hugo Awards, five Saturn Awards, and a Peabody Award."""
    
    output = summarizer(input_text,
                        max_length = MAX_LENGTH,
                        min_length = MIN_LENGTH)

    return output

#Translation, can choose a default model by providing a language pair in the task name but easier to pick the specific model
def translation(task = 'translation', 
                model_name = 'Helsinki-NLP/opus-mt-en-fr'):
    
    if task != 'translation':
        translator = pipeline(task,
                              min_length = MIN_LENGTH, 
                              max_length = MAX_LENGTH)
    else:
        translator = pipeline('translation', 
                              model = model_name,
                              min_length = MIN_LENGTH,
                              max_length = MAX_LENGTH) 

    output = translator(input('Enter text to be translated: '))

    return output

#Demo of how these models can propogate biases from their training data
def bias_demo():
    unmasker = pipeline('fill-mask', 
                        model = 'bert-base-uncased')

    output = unmasker('This man works as a [MASK].')
    print('Output for men\'s jobs: ', [o['token_str'] for o in output])

    output = unmasker('This woman works as a [MASK].')
    print('Output for women\'s jobs: ', [o['token_str'] for o in output])



if __name__ == '__main__':
    fire.Fire({
        'sentiment-analysis': sentiment_analysis,
        'zero-shot-classification': zero_shot_classification,
        'text-generation': text_generation,
        'fill-mask': fill_mask,
        'ner': ner,
        'question-answering': question_answering,
        'summarization': summarization,
        'translation': translation,
        'bias-demo': bias_demo
    })