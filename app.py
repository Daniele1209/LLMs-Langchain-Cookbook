import os
from response_handler import ResponseHandler
from config import first_request, second_request, title_string, description_string

import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper
from dotenv import load_dotenv

load_dotenv()

# --- Streamlit website part below ---

st.title(title_string) 
# input field for prompts
# prompt = st.text_input('State-of-the-art on the topic of:')
prompt = st.text_input(description_string)

# Prompt templates
research_template = PromptTemplate(
    input_variables= ['topic'],
    # template = 'What is the state-of-the-art on the topic of {topic}. Write it as a list'
    template = first_request
)

topic_template = PromptTemplate(
    input_variables= ['research', 'wikipedia_research'],
    template = second_request
)

hashtags_template = PromptTemplate(
    input_variables= ['post'],
    template = second_request
)

# Memory
# research_memory = ConversationBufferMemory(input_key='topic', memory_key='chat_memory')
topics_memory = ConversationBufferMemory(input_key='research', memory_key='chat_memory')
hashtag_memory = ConversationBufferMemory(input_key='topics', memory_key='chat_memory')

# LLMs
creativity_indicator = 0.8 # indicates how creative the LLMs responses are 
llm = OpenAI(temperature = creativity_indicator)
research_chain = LLMChain(llm=llm, prompt=research_template, verbose=True, output_key='research')
topics_chain = LLMChain(llm=llm, prompt=topic_template, verbose=True, output_key='topics', memory=topics_memory)
hashtags_chain = LLMChain(llm=llm, prompt=hashtags_template, verbose=True, output_key='hashtags', memory=hashtag_memory)

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
    post = topics_chain.run(research=research, wikipedia_research=wiki_research)
    hashtags = hashtags_chain.run(post=post)

    st.write(research)
    st.write(post)

    # with st.expander('Research History'):
    #     st.info(research_memory.buffer)

    with st.expander('Topic History'):
        st.info(topics_memory.buffer)

    with st.expander('Hashtags History'):
        st.info(hashtag_memory.buffer)

    with st.expander('Wiki History'):
        st.info(wiki_research)