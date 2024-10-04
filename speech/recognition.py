from core.azure_services import AzureSpeechService
import azure.cognitiveservices.speech as speechsdk


def recognize_from_microphone():
    azure_service = AzureSpeechService()
    speech_recognizer = azure_service.get_recognizer()

    try:
        print("Speak into your microphone.")
        result = speech_recognizer.recognize_once_async().get()
        if result.reason == speechsdk.ResultReason.RecognizedSpeech:
            return result.text
        else:
            return None
    except Exception as e:
        print(f"Error during speech recognition: {str(e)}")
        return None
