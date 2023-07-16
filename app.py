import os
from api_key import api_key_openai

import streamlit as st
from langchain.llms import OpenAI

# declare an environment variable mapped to the according key
os.environ['OPENAI_API_KEY'] = api_key_openai

# --- Streamlit website part below ---

st.title('LLM Showcase ☄️') 
# input field for prompts
prompt = st.text_input('Enter prompt')

# LLMs
creativity_indicator = 0.8 # indicates how creative the LLMs responses are 
llm = OpenAI(temperature = creativity_indicator)

# write response if given prompt 
if prompt:
    response = llm(prompt)
    st.write(response)