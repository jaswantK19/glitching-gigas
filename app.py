import streamlit as st
from core.document_loader import load_document
from core.qa_chain import create_qa_chain
from speech.recognition import recognize_from_microphone
from speech.synthesis import synthesize_speech, stop_speech
from utils.helpers import handle_simple_questions
import threading
import time


def handle_conversations(qa_chain, user_input):
    predefined_answer = handle_simple_questions(user_input)
    if predefined_answer:
        return predefined_answer

    result = qa_chain.invoke({"query": user_input})
    if 'result' in result:
        return result['result']
    else:
        return "I'm sorry, I couldn't find relevant information from the document."


def speech_synthesis_thread(response):
    try:
        synthesize_speech(response)
    except Exception as e:
        st.write(f"Error during speech synthesis: {e}")


def main():
    st.title("Debt Collection AI ChatBot")
    st.write("Ask me anything about debt collection.")

    # Initialize session state for QA chain and response
    if "qa_chain" not in st.session_state:
        vectorstore = load_document("loan-recovery.txt")
        st.session_state.qa_chain = create_qa_chain(vectorstore)
    if "response" not in st.session_state:
        st.session_state.response = None
    if "previous_input" not in st.session_state:
        st.session_state.previous_input = ""

    # Chat Interface
    user_input = st.text_input("Type your question here:", value=st.session_state.previous_input)

    # Process the input and get a response only if new input is provided
    if user_input and user_input != st.session_state.previous_input:
        st.session_state.previous_input = user_input
        response = handle_conversations(st.session_state.qa_chain, user_input)
        st.session_state.response = response
        st.write(f"ChatBot: {response}")
    elif st.session_state.response:
        st.write(f"ChatBot: {st.session_state.response}")

    # Speech synthesis for responses
    if st.session_state.response:
        if st.button("Read Response Aloud"):
            # Start speech synthesis in a separate thread
            synthesis_thread = threading.Thread(target=speech_synthesis_thread, args=(st.session_state.response,))
            synthesis_thread.start()

    # Button to interrupt speech synthesis
    if st.button("Interrupt"):
        # Stop the speech synthesis immediately
        stop_speech()
        st.write("Speech synthesis interrupted.")


if __name__ == "__main__":
    main()
