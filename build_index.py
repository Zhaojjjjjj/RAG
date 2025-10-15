from langchain_ollama import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.vectorstores import Chroma
import os

DATA_PATH = "data"
VECTOR_PATH = "vectorstore"

def load_all_documents():
    docs = []
    for filename in os.listdir(DATA_PATH):
        file_path = os.path.join(DATA_PATH, filename)
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
            docs.extend(loader.load())
        elif filename.endswith(".txt"):
            loader = TextLoader(file_path)
            docs.extend(loader.load())
        else:
            print(f"❌ 暂不支持的文件类型：{filename}")
    return docs

def main():
    print("📄 正在加载文档...")
    docs = load_all_documents()

    print("✂️ 正在分割文本...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    split_docs = splitter.split_documents(docs)

    print("🔍 正在生成向量...")
    embeddings = OllamaEmbeddings(model="mistral")
    db = Chroma.from_documents(split_docs, embeddings, persist_directory=VECTOR_PATH)
    db.persist()

    print("✅ 向量数据库构建完成！存储路径:", VECTOR_PATH)

if __name__ == "__main__":
    main()
