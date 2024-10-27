# import for LLM
from langchain_groq import ChatGroq
# import for RAG
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
# import for streamlit prompt template
from langchain_core.prompts import ChatPromptTemplate
# import for streamlit app
import streamlit as st
# import for env 
from dotenv import load_dotenv
import os

load_dotenv()


GROQ_API_KEY = os.getenv("GROQ_API_KEY")

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")



#RAG

loader = PyPDFLoader("research.pdf")
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
documents = text_splitter.split_documents(docs)

embaddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

vector = FAISS.from_documents(documents, embedding=embaddings)
retriever = vector.as_retriever()

# if "vactor" not in st.session_state:
#     st.session_state.loader = WebBaseLoader("https://docs.smith.langchain.com/")
#     st.session_state.docs = st.session_state.loader.load()
    
#     st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#     st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)
    
#     st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#     # st.session_state.embeddings = OllamaEmbeddings()
    
#     st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)
# retriever = st.session_state.vectors.as_retriever()



# creating LLM
llm = ChatGroq(
    model="gemma-7b-it",
    temperature=0,
    stop_sequences=""
    
)

# creating prompt
prompt = ChatPromptTemplate.from_template(
    """
    Answer the question based on the provided context only.
    Please provide the most accurate response based on the question
    <context>
    {context}
    <context>
    
    Question:{input}
    
    """
)


# creating chaining 
document_chain = create_stuff_documents_chain(llm, prompt)


chain = create_retrieval_chain(retriever, document_chain)


# invoking to get response
def get_response_from_groq(user_input):
    response = chain.invoke({"input":user_input})
    return response["answer"]


# streamlit app

st.title("Gemma Model Document Q&A ")
user_input = st.text_input("Ask That You Want:")

submit = st.button("Ask the Question")

if submit:
    response = get_response_from_groq(user_input)
    st.subheader("The Response is:")
    st.write(response)
    



