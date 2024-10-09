import asyncio
import websockets
import json
import base64
import pyaudio
import os
from dotenv import load_dotenv

load_dotenv()

GLADIA_API_KEY = os.getenv("GLADIA_API_KEY")

ERROR_KEY = 'error'
TYPE_KEY = 'type'
TRANSCRIPTION_KEY = 'transcription'
LANGUAGE_KEY = 'language'
CHUNK_SIZE = 1024  # Number of frames per buffer (affects latency)

SAMPLE_RATE = 16000  # Gladia's required sample rate
ENCODING = "WAV"
LANGUAGE_BEHAVIOUR = 'manual'
ENDPOINTING = 300
LANGUAGE = "spanish"
MODEL_TYPE = "fast" #"fast" or "accurate"
AUDIO_ENHANCER = False

if not GLADIA_API_KEY:
    print('You must provide a gladia key. Go to app.gladia.io')
    exit(1)
else:
    print('Using the Gladia key: ' + GLADIA_API_KEY)

# Gladia websocket URL
gladia_url = "wss://api.gladia.io/audio/text/audio-transcription"

async def send_audio(socket):
    # Configure stream with a configuration message
    configuration = {
        "x_gladia_key": GLADIA_API_KEY,
        "encoding":ENCODING,
        "sample_rate": SAMPLE_RATE,
        "language_behaviour": LANGUAGE_BEHAVIOUR,
        "language": LANGUAGE,
        "endpointing": ENDPOINTING,
        "model_type": MODEL_TYPE,
        "audio_enhancer": AUDIO_ENHANCER
    }
    await socket.send(json.dumps(configuration))

    # Initialize PyAudio for streaming from the microphone
    audio = pyaudio.PyAudio()
    stream = audio.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=SAMPLE_RATE,
        input=True,
        frames_per_buffer=CHUNK_SIZE,
    )

    print("Streaming audio from the microphone...")

    try:
        while True:
            audio_chunk = stream.read(CHUNK_SIZE, exception_on_overflow=False)
            base64_audio = base64.b64encode(audio_chunk).decode('utf-8')
            message = {"frames": base64_audio}
            await socket.send(json.dumps(message))
            await asyncio.sleep(0.1)  # Small delay to manage flow
    except asyncio.CancelledError:
        # Close the stream gracefully when the task is canceled
        stream.stop_stream()
        stream.close()
        audio.terminate()
        print("Audio streaming stopped.")

async def receive_transcription(socket):
    while True:
        response = await socket.recv()
        utterance = json.loads(response)
        if utterance:
            if ERROR_KEY in utterance:
                print(f"{utterance[ERROR_KEY]}")
                break
            else:
                if TYPE_KEY in utterance.keys():
                    print(f"{utterance[TYPE_KEY]}: ({utterance[LANGUAGE_KEY]}) {utterance[TRANSCRIPTION_KEY]}")
        else:
            print('Empty response, waiting for next utterance...')

async def main():
    async with websockets.connect(gladia_url) as socket:
        send_task = asyncio.create_task(send_audio(socket))
        receive_task = asyncio.create_task(receive_transcription(socket))
        await asyncio.gather(send_task, receive_task)

asyncio.run(main())
