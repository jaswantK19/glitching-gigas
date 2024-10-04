def handle_simple_questions(user_input):
    simple_questions = {
        "who are you": "I am an AI chatbot here to assist you with information about debt collection and related topics.",
        "what do you do": "I help answer questions and provide assistance, especially related to debt collection and recovery processes.",
        "what is your name": "I am your helpful assistant, designed to assist with debt collection inquiries."
    }
    return simple_questions.get(user_input.lower().strip(), None)
