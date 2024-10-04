from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains.question_answering import load_qa_chain

# Define the template for answering questions using the document
template = """
You are a helpful assistant. Use the following context from the document to answer the question accurately.

Here is the context: {context}

Question: {question}
"""
model = OllamaLLM(model='llama3.2:1b')
prompt = ChatPromptTemplate.from_template(template)


# Load the document and split into chunks
def load_document(file_path):
    # Load the document into memory
    loader = TextLoader(file_path)
    documents = loader.load()

    # Split the document into manageable chunks for retrieval
    text_splitter = CharacterTextSplitter(chunk_size=500,
                                          chunk_overlap=50)  # Reduce chunk size to improve context retrieval
    docs = text_splitter.split_documents(documents)

    # Create a HuggingFace embedding model to generate local embeddings
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")  # Efficient model for local embeddings

    # Create a FAISS vector store using these embeddings and document chunks
    vectorstore = FAISS.from_documents(docs, embedding_model)

    return vectorstore


# Initialize the retrieval-based QA chain
def create_qa_chain(vectorstore):
    # Create a retriever
    retriever = vectorstore.as_retriever(search_type="similarity",
                                         search_kwargs={"k": 5})  # Increase k to retrieve more context

    # Load a QA chain with the LLM
    qa_chain = load_qa_chain(llm=model, chain_type="stuff")

    # Create the RetrievalQA chain by combining the retriever and the QA chain
    retrieval_qa = RetrievalQA(combine_documents_chain=qa_chain, retriever=retriever)

    return retrieval_qa


def handle_simple_questions(user_input):
    # Predefined answers for simple or common questions
    simple_questions = {
        "who are you": "I am an AI chatbot here to assist you with information about debt collection and related topics.",
        "what do you do": "I help answer questions and provide assistance, especially related to debt collection and recovery processes.",
        "what is your name": "I am your helpful assistant, designed to assist with debt collection inquiries."
    }

    # Normalize user input to handle common questions
    normalized_input = user_input.lower().strip()
    return simple_questions.get(normalized_input, None)


def handle_conversations(qa_chain):
    print("Welcome to the AI ChatBot, Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break

        # First check if it's a simple question
        predefined_answer = handle_simple_questions(user_input)
        if predefined_answer:
            print("Bot: ", predefined_answer)
            continue

        # Otherwise, get the result by querying the retrieval chain with 'query' and include retrieved context
        result = qa_chain.invoke({"query": user_input})

        # Check if a response was generated based on retrievals
        if 'result' in result:
            print("Bot: ", result['result'])
        else:
            print("Bot: I'm sorry, I couldn't find relevant information from the document.")


if __name__ == "__main__":
    # Load your document (e.g., 'loan-recovery.txt')
    vectorstore = load_document("loan-recovery.txt")

    # Create a QA chain
    qa_chain = create_qa_chain(vectorstore)

    # Start the chatbot
    handle_conversations(qa_chain)
