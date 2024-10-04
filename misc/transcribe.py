import azure.cognitiveservices.speech as speechsdk


def transcribe_local_audio(speech_key, service_region, audio_filename):
    # Create an instance of a speech config with specified subscription key and service region.
    speech_config = speechsdk.SpeechConfig(
        subscription=speech_key, region=service_region)

    # Enable speaker diarization
    speech_config.set_property(
        speechsdk.PropertyId.SpeechServiceConnection_EnableAudioDiarization, "true")
    # Optionally, set the expected number of speakers if known
    speech_config.set_property(
        speechsdk.PropertyId.SpeechServiceConnection_InitialSilenceTimeoutMs, "60000")  # longer initial silence
    speech_config.set_property(
        speechsdk.PropertyId.SpeechServiceConnection_EndSilenceTimeoutMs, "3000")       # longer end silence

    # Create an audio configuration that points to an audio file.
    audio_config = speechsdk.audio.AudioConfig(filename=audio_filename)

    # Create a recognizer with the given settings
    speech_recognizer = speechsdk.SpeechRecognizer(
        speech_config=speech_config, audio_config=audio_config)

    print("Recognizing first result...")
    # Starts speech recognition, and returns after a single utterance is recognized. The end of a
    # single utterance is determined by listening for silence at the end or until a maximum of 15
    # seconds of audio is processed.  The task returns the recognition text as result.
    result = speech_recognizer.recognize_once()

    # Checks result.
    if result.reason == speechsdk.ResultReason.RecognizedSpeech:
        print("Recognized: {}".format(result.text))
        print("Speaker diarization:")
        for word in result.best().words:
            print(f"Word: {word.text}, Speaker: {word.speaker_id}")
    elif result.reason == speechsdk.ResultReason.NoMatch:
        print("No speech could be recognized: {}".format(result.no_match_details))
    elif result.reason == speechsdk.CancellationReason.Error:
        print("Error: {}".format(result.cancellation_details.reason))
        print("Error details: {}".format(
            result.cancellation_details.error_details))
    else:
        print("Canceled: {}".format(result.cancellation_details.reason))


# Replace with your subscription key and service region from Azure
transcribe_local_audio("YourAzureSubscriptionKey",
                       "YourAzureRegion", "path_to_your_audio_file.wav")
