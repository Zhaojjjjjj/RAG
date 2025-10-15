import streamlit as st
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA

VECTOR_PATH = "vectorstore"

st.set_page_config(page_title="🧠 RAG(Mistral)", layout="centered")
st.title("RAG")
st.markdown("**LangChain + Ollama + Chroma + Mistral** ")

@st.cache_resource
def load_qa_chain():
    embeddings = OllamaEmbeddings(model="mistral")
    db = Chroma(persist_directory=VECTOR_PATH, embedding_function=embeddings)
    retriever = db.as_retriever(search_kwargs={"k": 3})
    llm = OllamaLLM(model="mistral")
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

qa = load_qa_chain()

query = st.text_input("Question: ", placeholder="e.g. What is this document about?")
if query:
    with st.spinner("Thinking..."):
        answer = qa.run(query)
        st.markdown("**Answer:** " + answer)
