from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def get_vectorstore():
    loader = DirectoryLoader("./rag/documents", glob="**/*.txt")
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings()
    vectordb = Chroma.from_documents(docs, embedding=embeddings, persist_directory="./rag/chroma_db")
    return vectordb