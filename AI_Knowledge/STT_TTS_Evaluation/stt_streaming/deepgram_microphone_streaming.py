import os
from dotenv import load_dotenv

from deepgram import (
    DeepgramClient,
    LiveTranscriptionEvents,
    LiveOptions,
    DeepgramClientOptions,
    Microphone
)

import argparse


def microphone_streaming(microphone_streaming_options):
    try:
        deepgram: DeepgramClient = DeepgramClient(DEEPGRAM_API_KEY)

        dg_connection = deepgram.listen.live.v("1")
        is_finals = []

        # Define event handlers
        def on_open(self, open, **kwargs):
            print(f"Deepgram Connection Open")

        def on_message(self, result, **kwargs):
            global is_finals
            sentence = result.channel.alternatives[0].transcript
            print(f"is_final: {result.is_final}, speech_final: {result.speech_final}: \t{sentence}")

        def on_metadata(self, metadata, **kwargs):
            print(f"Deepgram Metadata: {metadata}")

        def on_speech_started(self, speech_started, **kwargs):
            print(f"Deepgram Speech Started")

        def on_utterance_end(self, utterance_end, **kwargs):
            global is_finals
            if len(is_finals) > 0:
                utterance = ' '.join(is_finals)
                print(f"Deepgram Utterance End: {utterance}")
                is_finals = []
            else: 
                print(f"Deepgram Utterance End - Empty is_finals")

        def on_close(self, close, **kwargs):
            print(f"Deepgram Connection Closed")

        def on_error(self, error, **kwargs):
            print(f"Deepgram Handled Error: {error}")

        def on_unhandled(self, unhandled, **kwargs):
            print(f"Deepgram Unhandled Websocket Message: {unhandled}")

        dg_connection.on(LiveTranscriptionEvents.Open, on_open)
        dg_connection.on(LiveTranscriptionEvents.Transcript, on_message)
        dg_connection.on(LiveTranscriptionEvents.Metadata, on_metadata)
        dg_connection.on(LiveTranscriptionEvents.SpeechStarted, on_speech_started)
        dg_connection.on(LiveTranscriptionEvents.UtteranceEnd, on_utterance_end)
        dg_connection.on(LiveTranscriptionEvents.Close, on_close)
        dg_connection.on(LiveTranscriptionEvents.Error, on_error)
        dg_connection.on(LiveTranscriptionEvents.Unhandled, on_unhandled)

        options: LiveOptions = microphone_streaming_options

        addons = {
            # Prevent waiting for additional numbers
            "no_delay": "true"
        }

        print("\n\nPress Enter to stop recording...\n\n")
        if dg_connection.start(options, addons=addons) is False:
            print("Failed to connect to Deepgram")
            return

        # Open a microphone stream on the default input device
        microphone = Microphone(dg_connection.send)

        # start microphone
        microphone.start()

        # wait until finished
        input("")

        # Wait for the microphone to close
        microphone.finish()

        # Indicate that we've finished
        dg_connection.finish()

        print("Finished")
        # sleep(30)  # wait 30 seconds to see if there is any additional socket activity
        # print("Really done!")

    except Exception as e:
        print(f"Could not open socket: {e}")
        return

def parse_arguments():
    parser = argparse.ArgumentParser(description="Microphone Streaming Options")
    
    parser.add_argument("--model", type=str, default="nova-2", help="Model to use for speech recognition")
    parser.add_argument("--language", type=str, default="es-ES", help="Language for speech recognition")
    parser.add_argument("--smart_format", type=bool, default=True, help="Apply smart formatting to the output")
    parser.add_argument("--encoding", type=str, default="linear16", help="Raw audio format encoding")
    parser.add_argument("--channels", type=int, default=1, help="Number of audio channels")
    parser.add_argument("--sample_rate", type=int, default=16000, help="Audio sample rate")
    parser.add_argument("--interim_results", type=bool, default=True, help="Get interim results")
    parser.add_argument("--vad_events", type=bool, default=True, help="Voice Activity Detection events")
    parser.add_argument("--endpointing", type=int, default=500, help="Milliseconds of silence before finalizing speech")

    return parser.parse_args()


if __name__ == "__main__":
    load_dotenv()

    DEEPGRAM_API_KEY = "4f81177f9f82a5fe564d866bac4195b0c7666b25"
    #os.getenv("DEEPGRAM_API_KEY")
    if not DEEPGRAM_API_KEY:
        raise Exception("Please check that you've indicated DEEPGRAM_API_KEY in your .env file")
    else:
        print("Deepgram API Key validated")

    args = parse_arguments()

    microphone_streaming_options = LiveOptions(
        model=args.model,
        language=args.language,
        smart_format=args.smart_format,
        encoding=args.encoding,
        channels=args.channels,
        sample_rate=args.sample_rate,
        interim_results=args.interim_results,
        vad_events=args.vad_events,
        endpointing=args.endpointing,
    )

    microphone_streaming(microphone_streaming_options)
