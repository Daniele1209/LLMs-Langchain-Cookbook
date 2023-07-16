import os
from api_key import api_key_openai
from response_handler import ResponseHandler

import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper

# declare an environment variable mapped to the according key
os.environ['OPENAI_API_KEY'] = api_key_openai

# --- Streamlit website part below ---

st.title('LLM Showcase ☄️') 
# input field for prompts
prompt = st.text_input('State-of-the-art on the topic of:')

# Prompt templates
research_template = PromptTemplate(
    input_variables= ['topic'],
    template = 'What is the state-of-the-art on the topic of {topic}. Write it as a list'
)

topic_template = PromptTemplate(
    input_variables= ['research', 'wikipedia_research'],
    template = 'Choose from the following research topics what is the newest and most relevent at the moment, and expand upon it: {research}. Validate it using the following information: {wikipedia_research}'
)

# Memory
research_memory = ConversationBufferMemory(input_key='topic', memory_key='chat_memory')
topics_memory = ConversationBufferMemory(input_key='research', memory_key='chat_memory')

# LLMs
creativity_indicator = 0.8 # indicates how creative the LLMs responses are 
llm = OpenAI(temperature = creativity_indicator)
research_chain = LLMChain(llm=llm, prompt=research_template, verbose=True, output_key='research', memory=research_memory)
topics_chain = LLMChain(llm=llm, prompt=topic_template, verbose=True, output_key='topics', memory=topics_memory)

# seq_chain = SequentialChain(chains=[research_chain, topics_chain], verbose=True,
#                             input_variables=['topic'], output_variables=['research', 'topics'])

wiki = WikipediaAPIWrapper()
response_handler = ResponseHandler()

# write response if given prompt using the defined prompt format
if prompt:
    # response = seq_chain({'topic': prompt})
    # response_handler.set_response(response['research'])
    # response_handler.post_process_list_response()
    research = research_chain.run(prompt)
    wiki_research = wiki.run(prompt)
    topics = topics_chain.run(research=research, wikipedia_research=wiki_research)

    st.write(research)
    st.write(topics)

    with st.expander('Research History'):
        st.info(research_memory.buffer)

    with st.expander('Topic History'):
        st.info(topics_memory.buffer)

    with st.expander('Wiki History'):
        st.info(wiki_research)