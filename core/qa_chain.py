from langchain.chains.question_answering import load_qa_chain
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_ollama import OllamaLLM
import logging


def create_qa_chain(vectorstore):
    model = OllamaLLM(model='llama3.2:1b')
    logging.info(f"Using model: llama3.2:1b")

    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    logging.info("Retriever created with similarity search")

    qa_chain = load_qa_chain(llm=model, chain_type="stuff")
    logging.info("QA chain loaded")

    return RetrievalQA(combine_documents_chain=qa_chain, retriever=retriever)
