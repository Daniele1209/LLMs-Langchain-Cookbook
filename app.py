import os
from api_key import api_key_openai
from response_handler import ResponseHandler

import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain

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
    input_variables= ['research'],
    template = 'Choose from these research topics {research} what is the newest and most relevent at the moment. Expand upon it'
)

# LLMs
creativity_indicator = 0.8 # indicates how creative the LLMs responses are 
llm = OpenAI(temperature = creativity_indicator)
research_chain = LLMChain(llm=llm, prompt=research_template, verbose=True, output_key='research')
title_chain = LLMChain(llm=llm, prompt=topic_template, verbose=True, output_key='topics')
seq_chain = SequentialChain(chains=[research_chain, title_chain], verbose=True,
                            input_variables=['topic'], output_variables=['research', 'topics'])

response_handler = ResponseHandler()

# write response if given prompt using the defined prompt format
if prompt:
    response = seq_chain({'topic': prompt})
    # response_handler.set_response(response['research'])
    # response_handler.post_process_list_response()
    st.write(response['research'])
    st.write(response['topics'])