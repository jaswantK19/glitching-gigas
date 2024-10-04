from langchain.chains.question_answering import load_qa_chain
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_ollama import OllamaLLM


def create_qa_chain(vectorstore):
    model = OllamaLLM(model='llama3.2:1b')
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    qa_chain = load_qa_chain(llm=model, chain_type="stuff")

    return RetrievalQA(combine_documents_chain=qa_chain, retriever=retriever)
