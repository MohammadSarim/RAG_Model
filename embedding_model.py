from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
import streamlit as st

model_cache_dir = "/Users/mohammadsarim/RAG_Model/Embedding_Model"

embed_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",
    cache_folder=model_cache_dir
)

prompt_template = """

Human: Use the following pieces of context to provide a 
concise answer to the question at the end but usse atleast summarize with 
250 words with detailed explaantions. If you don't know the answer, 
just say that you don't know, don't try to make up an answer.
<context>
{context}
</context

Question: {question}

Assistant:"""


def get_documents():
    loader=PyPDFDirectoryLoader('data')
    documents = loader.load()
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,
                                                 chunk_overlap=500)
    
    docs = text_splitter.split_documents(documents)

    return docs

def get_vector_store(docs):
    vectorstore_faiss = FAISS.from_documents(
        docs,
        embed_model
    )
    vectorstore_faiss.save_local('faiss_index')

def get_llm():
    llm = ChatGroq(
    model="llama-3.1-70b-versatile",
    max_tokens=512,
    temperature=0,
    groq_api_key = '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'
)

    return llm

PROMPT = PromptTemplate(
    template=prompt_template,input_variable=['context','question']
)

def get_response_llm(llm,vectorstore_faiss,query):
    qa = RetrievalQA.from_chain_type(
        llm = llm,
        chain_type = 'stuff',
        retriever = vectorstore_faiss.as_retriever(
            search_type = 'similarity', search_kwargs = {'k': 3 }
        ),
        return_source_documents = True,
        chain_type_kwargs = {'prompt':PROMPT}
    )
    answer = qa({'query':query})

    return answer['result']

def main():
    st.set_page_config("RAG Demo")
    st.header("End to end RAG Application")
    user_question = st.text_input("Ask a Question from the PDF Files")


    with st.sidebar:
        st.title("Update Or Create Vector Store:")
        
        if st.button("Store Vector"):
            with st.spinner("Processing..."):
                docs = get_documents()
                get_vector_store(docs)
                st.success("Done")
    
    if st.button("Send"):
        with st.spinner("Processing..."):
            faiss_index = FAISS.load_local("faiss_index", embed_model, allow_dangerous_deserialization=True)
            llm = get_llm()
            st.write(get_response_llm(llm,faiss_index,user_question))



if __name__ == "__main__":
    main()
    