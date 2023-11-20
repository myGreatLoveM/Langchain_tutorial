# Q&A ChatBot
import streamlit as st
from dotenv import load_dotenv
import os
from langchain.llms import OpenAI

load_dotenv()

def get_openai_response(question):
    llm = OpenAI(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        model_name="text-davinci-003",
        temperature=0.6,
    )
    response = llm.predict(question)
    return response


st.set_page_config(
    page_title='Q&A Demo'
)

st.header('Langchain Applictaion')

input = st.text_input("Input : ", key="input")

response = get_openai_response(input)

submit = st.button('Ask the Question')

if submit:
    st.subheader('The Response is ')
    st.write(response)