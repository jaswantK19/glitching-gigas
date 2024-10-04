from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_document(file_path):
    loader = TextLoader(file_path)
    documents = loader.load()
    logging.info(f"Loaded {len(documents)} documents from {file_path}")

    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)
    logging.info(f"Split documents into {len(docs)} chunks")

    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(docs, embedding_model)
    logging.info("FAISS vector store created with embeddings")

    return vector_store
