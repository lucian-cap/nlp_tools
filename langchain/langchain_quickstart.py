from langchain.llms import OpenAI
#import os

# Can set envrionment variable in the terminal with: "export OPENAI_API_KEY='...'" or run the below lines
# os.environ['OPEN_API_KEY'] = '...'
# os.environ['SERPAPI_API_KEY] = '...'

#another alternative is to set the API key dynamically with the openai_api_key arg in the LLM wrapper
#llm = OpenAI(openai_api_key='...')

#Get predictions from a LLM
llm = OpenAI(temperature = 0.9)
input_text = 'What would be a good company name for a company that makes colorful socks?'
# output = llm(input_text)

#Creating a Prompt Template
from langchain.prompts import PromptTemplate
prompt = PromptTemplate(input_variables = ['product'],
                        template = 'What is a good name for a company that makes {product}?')
print(prompt.format(product = 'colorful socks'))

#Creating a chain, LLMChain consists of a PromptTemplate and a LLM
from langchain.chains import LLMChain
chain = LLMChain(llm = llm, prompt = prompt)
#chain.run('colorful socks')

#Defining and using a agent to determine action selection
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
tools = load_tools(['serpapi', 'llm-math'], llm = llm)
agent = initialize_agent(tools, llm, 
                         agent   = AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
                         verbose = True)
#agent.run('What was the high termperature in Warner Robins, GA yesterday in Fahrenheit? What is that number raised to the .023 power?')

#Giving the agent memory using ConversationChain w/2 types of memory
from langchain import OpenAI, ConversationChain
llm = OpenAI(temperature = 0)
conversation = ConversationChain(llm = llm, verbose = True)
#output = conversation.predict(input = 'Hi there!')
#print(output)
# output = conversation.predict(input = "I'm doing well! Just having a conversation with an AI.")
# print(output)

#Using a Chat model, they use LLMs under the hood but expose a different interface using chat messages as I/O instead of text in/out
from langchain.chat_models import ChatOpenAI
from langchain.schema import (AIMessage, HumanMessage, SystemMessage)
chat = ChatOpenAI(temperature = 0)

# chat([HumanMessage(content = 'Translate this sentence from English to French. I love programming.')])

messages = [SystemMessage(content = 'You are a helpful assistant that translates English to French.'),
            HumanMessage(content = 'I love programming.')]
# chat(messages)

batch_messages = [[SystemMessage(content = 'You are a helpful assistant that translates English to French.'),
                   HumanMessage(content = 'I love programming.')],
                  [SystemMessage(content = 'You are a helpful assistant that translates English to French.'),
                   HumanMessage(content = 'I love artificial intelligence.')]]
# result = chat.generate(batch_messages)
# print(result)
# print('Token usage: ', result.llm_output['token_usage'])

#Using Prompt Templates in Chats
from langchain.prompts.chat import (ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate)
system_template = 'You are a helpful assistant that translates {input_language} to {output_language}.'
system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
human_template = '{text}'
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

# chat(chat_prompt.format_prompt(input_language = 'English',
#                                output_language = 'French',
#                                text = 'I love programming.').to_messages())

#LLMChains can be used with chat models as well, using the templates & prompts above:
chain = LLMChain(llm = chat,
                 prompt = chat_prompt)
# chain.run(input_language = 'English', 
#           output_language = 'French',
#           text = 'I love programming.')

#Running a agent with a chat model
chat = ChatOpenAI(temperature = 0)
llm  = OpenAI(temperature     = 0)
tools = load_tools(['serpapi', 'llm-math'], llm=llm)
agent = initialize_agent(tools, chat, 
                         agent   = AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
                         verbose = True)
#agent.run('Who is Sir Patrick Stewart\'s wife? What is her current age raised to the 0.23 power?')

#Using memory with chains & agents made with chat models, different from using LLMs w/memory
from langchain.prompts import (ChatPromptTemplate, 
                               MessagesPlaceholder,
                               SystemMessagePromptTemplate,
                               HumanMessagePromptTemplate)
from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory

prompt = ChatPromptTemplate.from_messages([SystemMessagePromptTemplate.from_template('The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.'),
                                           MessagesPlaceholder(variable_name = 'history'),
                                           HumanMessagePromptTemplate.from_template('{input}')])
llm = ChatOpenAI(temperature = 0)
memory = ConversationBufferMemory(return_messages = True)
conversation = ConversationChain(memory = memory, 
                                 prompt = prompt, 
                                 llm = llm)
# conversation.predict(input = 'Hi there!')
# conversation.predict(input = 'I\'m doing well! Just having a conversation with an AI.')