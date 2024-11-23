from langchain_groq import ChatGroq
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import streamlit as st

llm = ChatGroq(
    model="llama-3.1-70b-versatile",
    temperature=0,
    groq_api_key = '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'
)

def my_chatbot(language, user_text):
    prompt = PromptTemplate(
        input_variables = ['language', 'user_text'],
        template = "You are a chatbot. You are in {language}.\n\n{user_text}"
    )

    llm_chain = LLMChain(llm = llm, prompt = prompt)
    response = llm_chain({'language':language, 'user_text':user_text})
    
    return response

st.title('Retrieval Augmented Generation Chatbot')

language = st.sidebar.selectbox('Language', ['english','spanish','hindi'])

if language:
    user_text = st.sidebar.text_area(label = "What is your question?",
                                     max_chars = 100)
    
if user_text:
    response = my_chatbot(language, user_text)
    st.write(response['text'])    

# response = llm.invoke('The first person to land on moon was...')
# print(response.content)