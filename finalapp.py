import streamlit as st
import os
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
import bs4



from dotenv import load_dotenv
load_dotenv()

## Load the NVIDIA API KEY

os.environ['NVIDIA_API_KEY']=os.getenv('NVIDIA_API_KEY')

llm=ChatNVIDIA(model="meta/llama3-70b-instruct")   ## NVIDIA NIM Inferencing

def vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = NVIDIAEmbeddings()

# List of URLs to load
        urls = [
            "https://wellbeingtrust.org/mental-health/mental-health-resources-helpful-web-site/",
            "https://digitalwellnesslab.org/articles/where-to-start-mental-health-resources-from-mental-health-america/",
            "https://www.nimh.nih.gov/",
            "https://www.nimh.nih.gov/news/science-updates/2025/study-illuminates-the-genetic-architecture-of-bipolar-disorder/"
        ]
        st.session_state.loader=WebBaseLoader(urls)
        st.session_state.docs=st.session_state.loader.load()
        st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=700,chunk_overlap=50) ## Chunk Creation
        st.session_state.final_documents=st.session_state.text_splitter.split_documents(st.session_state.docs[:30])
        st.session_state.vectors=FAISS.from_documents(st.session_state.final_documents,st.session_state.embeddings)



st.title("Mental Health & Wellness Chatbot")

prompt=ChatPromptTemplate.from_template(
"""
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question
<context>
{context}
<context>
Questions:{input}


"""        
)

prompt1=st.text_input("Enter Your Question From Documents")

if st.button("Document Embedding"):
    vector_embedding()
    st.write("FAISS Vector Store DB Is Ready Using NvidiaEmbedding")

import time
if prompt1:
    document_chain=create_stuff_documents_chain(llm,prompt)
    retriever=st.session_state.vectors.as_retriever()
    retrieval_chain=create_retrieval_chain(retriever,document_chain)
    start=time.process_time()
    response=retrieval_chain.invoke({'input':prompt1})
    print("Response time :",time.process_time()-start)
    st.write(response['answer'])


    # With a streamlit expander
    with st.expander("Document Similarity Serach"):
        # Find the relevant chunks
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("---------------------------------")





