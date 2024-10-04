
from core.document_loader import load_document
from core.qa_chain import create_qa_chain
from speech.recognition import recognize_from_microphone
from speech.synthesis import synthesize_speech
from utils.helpers import handle_simple_questions

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


if __name__ == "__main__":
    vectorstore = load_document("loan-recovery.txt")
    qa_chain = create_qa_chain(vectorstore)
    handle_conversations(qa_chain)
