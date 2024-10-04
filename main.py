import os
import nltk
from dotenv import load_dotenv
import azure.cognitiveservices.speech as speechsdk
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains.question_answering import load_qa_chain

# Load environment variables for Azure services
load_dotenv()

# NLTK setup
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Azure TTS and STT settings
settings = {
    'speech_config': speechsdk.SpeechConfig(
        subscription=os.getenv('AZURE_SUBSCRIPTION_ID'),
        region=os.getenv("AZURE_REGION")
    ),
    'recognize_audio_config': speechsdk.audio.AudioConfig(use_default_microphone=True),
    'synthesize_audio_config': speechsdk.audio.AudioOutputConfig(use_default_speaker=True)
}

# Speech synthesis setup
speech_config_for_synthesis = settings['speech_config']
speech_config_for_synthesis.speech_synthesis_voice_name = 'en-US-JennyNeural'
speech_synthesizer = speechsdk.SpeechSynthesizer(
    speech_config=speech_config_for_synthesis,
    audio_config=settings['synthesize_audio_config']
)

# Speech recognition setup
speech_config = settings['speech_config']
speech_config.speech_recognition_language = "en-US"
speech_recognizer = speechsdk.SpeechRecognizer(
    speech_config=speech_config,
    audio_config=settings['recognize_audio_config']
)

# Initialize the model for document retrieval-based QA
model = OllamaLLM(model='llama3.2:1b')

# Define the prompt template for answering questions from a document
template = """
You are a helpful assistant. Use the following context from the document to answer the question accurately.

Here is the context: {context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)


# Load and process the document for retrieval
def load_document(file_path):
    loader = TextLoader(file_path)
    documents = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)

    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(docs, embedding_model)

    return vector_store


# Create a QA chain for answering document-based questions
def create_qa_chain(vectorstore):
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    qa_chain = load_qa_chain(llm=model, chain_type="stuff")
    retrieval_qa = RetrievalQA(combine_documents_chain=qa_chain, retriever=retriever)
    return retrieval_qa


# Handle simple predefined questions
def handle_simple_questions(user_input):
    simple_questions = {
        "who are you": "I am an AI chatbot here to assist you with information about debt collection and related topics.",
        "what do you do": "I help answer questions and provide assistance, especially related to debt collection and recovery processes.",
        "what is your name": "I am your helpful assistant, designed to assist with debt collection inquiries."
    }
    return simple_questions.get(user_input.lower().strip(), None)


# Recognize speech input using Azure STT
def recognize_from_microphone():
    try:
        print("Speak into your microphone.")
        speech_recognition_result = speech_recognizer.recognize_once_async().get()
        if speech_recognition_result.reason == speechsdk.ResultReason.RecognizedSpeech:
            return speech_recognition_result.text
        else:
            return None
    except Exception as e:
        print(f"Error during speech recognition: {str(e)}")
        return None


# Synthesize speech using Azure TTS
def synthesize_speech(input_text):
    try:
        speech_synthesis_result = speech_synthesizer.speak_text_async(input_text).get()
        if speech_synthesis_result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            print(f"Speech synthesized for: {input_text}")
    except Exception as e:
        print(f"Error during speech synthesis: {str(e)}")


# Start a voice conversation
def handle_conversations(qa_chain):
    synthesize_speech("Welcome to the AI ChatBot. You can ask me anything about debt collection. Say 'exit' to quit.")
    while True:
        user_input = recognize_from_microphone()
        if user_input and user_input.lower() == "exit":
            synthesize_speech("Goodbye!")
            break

        predefined_answer = handle_simple_questions(user_input)
        if predefined_answer:
            synthesize_speech(predefined_answer)
            continue

        result = qa_chain.invoke({"query": user_input})
        if 'result' in result:
            synthesize_speech(result['result'])
        else:
            synthesize_speech("I'm sorry, I couldn't find relevant information from the document.")


# Main function to start the voice bot
if __name__ == "__main__":
    vectorstore = load_document("loan-recovery.txt")
    qa_chain = create_qa_chain(vectorstore)
    handle_conversations(qa_chain)
