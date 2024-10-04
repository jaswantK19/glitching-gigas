import os
from dotenv import load_dotenv
import azure.cognitiveservices.speech as speechsdk

load_dotenv()

class AzureSpeechService:
    def __init__(self):
        self.speech_config = speechsdk.SpeechConfig(
            subscription=os.getenv('AZURE_SUBSCRIPTION_ID'),
            region=os.getenv("AZURE_REGION")
        )
        self.speech_config.speech_synthesis_voice_name = 'en-US-JennyNeural'
        self.recognize_audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)
        self.synthesize_audio_config = speechsdk.audio.AudioOutputConfig(use_default_speaker=True)

    def get_recognizer(self):
        return speechsdk.SpeechRecognizer(
            speech_config=self.speech_config,
            audio_config=self.recognize_audio_config
        )

    def get_synthesizer(self):
        return speechsdk.SpeechSynthesizer(
            speech_config=self.speech_config,
            audio_config=self.synthesize_audio_config
        )
