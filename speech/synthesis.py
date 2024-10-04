from core.azure_services import AzureSpeechService
import azure.cognitiveservices.speech as speechsdk

# Create a global Azure speech service instance
azure_service = AzureSpeechService()

def synthesize_speech(input_text):
    speech_synthesizer = azure_service.get_synthesizer()
    try:
        result = speech_synthesizer.speak_text_async(input_text).get()
        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            print(f"Speech synthesized for: {input_text}")
    except Exception as e:
        print(f"Error during speech synthesis: {str(e)}")

def stop_speech():
    """Stop any ongoing speech synthesis."""
    azure_service.stop_speech()
