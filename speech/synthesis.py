from core.azure_services import AzureSpeechService
import azure.cognitiveservices.speech as speechsdk


def synthesize_speech(input_text):
    azure_service = AzureSpeechService()
    speech_synthesizer = azure_service.get_synthesizer()

    try:
        result = speech_synthesizer.speak_text_async(input_text).get()
        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            print(f"Speech synthesized for: {input_text}")
    except Exception as e:
        print(f"Error during speech synthesis: {str(e)}")
