import streamlit as st
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.callbacks.base import BaseCallbackHandler

VECTOR_PATH = "vectorstore"

st.set_page_config(page_title="RAG(Mistral)", layout="centered")
st.title("RAG")
st.markdown("**LangChain + Ollama + Chroma + Mistral** ")

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)

@st.cache_resource
def load_qa_chain():
    embeddings = OllamaEmbeddings(model="mistral")
    db = Chroma(persist_directory=VECTOR_PATH, embedding_function=embeddings)
    retriever = db.as_retriever(search_kwargs={"k": 3})
    llm = OllamaLLM(model="mistral", streaming=True)
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

qa = load_qa_chain()

query = st.text_input("Question: ", placeholder="e.g. What is this document about?")
if query:
    with st.spinner("Thinking..."):
        answer_container = st.empty()
        stream_handler = StreamHandler(answer_container, initial_text="**Answer:** ")
        qa.run(query, callbacks=[stream_handler])
