import os
from api_key import api_key_openai
from response_handler import ResponseHandler

import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain 

# declare an environment variable mapped to the according key
os.environ['OPENAI_API_KEY'] = api_key_openai

# --- Streamlit website part below ---

st.title('LLM Showcase ☄️') 
# input field for prompts
prompt = st.text_input('State-of-the-art on the topic of:')

# Prompt template
title_template = PromptTemplate(
    input_variables= ['topic'],
    template = 'What is the state-of-the-art on the topic of {topic}. Write it as a list'
)

# LLMs
creativity_indicator = 0.8 # indicates how creative the LLMs responses are 
llm = OpenAI(temperature = creativity_indicator)
title_chain = LLMChain(llm=llm, prompt=title_template, verbose=True)

response_handler = ResponseHandler()

# write response if given prompt using the defined prompt format
if prompt:
    response = title_chain.run(topic=prompt)
    response_handler.set_response(response)
    response_handler.post_process_response()
    st.write(response_handler.get_reponse())