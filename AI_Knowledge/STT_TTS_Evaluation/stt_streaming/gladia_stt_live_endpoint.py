import asyncio
import base64
import json
import signal
import sys
from datetime import time
from typing import Literal, TypedDict

import pyaudio
import requests
from websockets.asyncio.client import ClientConnection, connect
from websockets.exceptions import ConnectionClosedOK

import functools  # Add this import

import tracemalloc
import os
from dotenv import load_dotenv
load_dotenv()

## Constants
GLADIA_API_URL = "https://api.gladia.io"
GLADIA_API_KEY = os.getenv("GLADIA_API_KEY")

## Type definitions
class InitiateResponse(TypedDict):
    id: str
    url: str


class LanguageConfiguration(TypedDict):
    languages: list[str] | None
    code_switching: bool | None


class StreamingConfiguration(TypedDict):
    # This is a reduced set of options. For a full list, see the API documentation.
    # https://docs.gladia.io/api-reference/v2/live/init
    encoding: Literal["wav/pcm", "wav/alaw", "wav/ulaw"]
    bit_depth: Literal[8, 16, 24, 32]
    sample_rate: Literal[8_000, 16_000, 32_000, 44_100, 48_000]
    channels: int
    language_config: LanguageConfiguration | None




def init_live_session(config: StreamingConfiguration) -> InitiateResponse:
    response = requests.post(
        f"{GLADIA_API_URL}/v2/live",
        headers={"X-Gladia-Key": GLADIA_API_KEY},
        json=config,
        timeout=3,
    )
    if not response.ok:
        print(f"{response.status_code}: {response.text or response.reason}")
        exit(response.status_code)
    return response.json()


def format_duration(seconds: float) -> str:
    milliseconds = int(seconds * 1_000)
    return time(
        hour=milliseconds // 3_600_000,
        minute=(milliseconds // 60_000) % 60,
        second=(milliseconds // 1_000) % 60,
        microsecond=milliseconds % 1_000 * 1_000,
    ).isoformat(timespec="milliseconds")


async def print_messages_from_socket(socket: ClientConnection) -> None:
    async for message in socket:
        content = json.loads(message)
        if content.get("error",None):
            print(content['error'])
        if content["type"] == "transcript" and content["data"]["is_final"]:
            start = format_duration(content["data"]["utterance"]["start"])
            end = format_duration(content["data"]["utterance"]["end"])
            text = content["data"]["utterance"]["text"].strip()
            print(f"{start} --> {end} | {text}")
        if content["type"] == "post_final_transcript":
            print("\n################ End of session ################\n")
            print(json.dumps(content, indent=2, ensure_ascii=False))


async def stop_recording(websocket: ClientConnection) -> None:
    print(">>>>> Ending the recording…")
    await websocket.send(json.dumps({"type": "stop_recording"}))
    await asyncio.sleep(0)


## Sample code
P = pyaudio.PyAudio()

CHANNELS = 1
FORMAT = pyaudio.paInt16
FRAMES_PER_BUFFER = 3200
SAMPLE_RATE = 16_000
MODEL_TYPE = "fast" #"fast" or "accurate"
ENDPOINTING = 0.3
AUDIO_ENHANCER = False
NAMED_ENTITY_RECOGNITION = True
SENTIMENT_ANALYSIS = True


STREAMING_CONFIGURATION: StreamingConfiguration = {
    "encoding": "wav/pcm",
    "sample_rate": SAMPLE_RATE,
    "bit_depth": 16,  # It should match the FORMAT value
    "channels": CHANNELS,
    "language_config": {
        "languages": [],
        "code_switching": True,
    },
    "endpointing": ENDPOINTING,
    "pre_processing": {
        "audio_enhancer": AUDIO_ENHANCER,
    },
    "realtime_processing": {
        "words_accurate_timestamps":False,
        "custom_vocabulary":False,
        "custom_vocabulary_config": [""],
        "named_entity_recognition": NAMED_ENTITY_RECOGNITION,
        "sentiment_analysis":SENTIMENT_ANALYSIS,
    }
}


async def send_audio(socket: ClientConnection) -> None:
    stream = P.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=SAMPLE_RATE,
        input=True,
        frames_per_buffer=FRAMES_PER_BUFFER,
    )

    while True:
        data = stream.read(FRAMES_PER_BUFFER)
        data = base64.b64encode(data).decode("utf-8")
        json_data = json.dumps({"type": "audio_chunk", "data": {"chunk": str(data)}})
        try:
            await socket.send(json_data)
            await asyncio.sleep(0.1)  # Send audio every 100ms
        except ConnectionClosedOK:
            return


async def main():
    response = init_live_session(STREAMING_CONFIGURATION)
    async with connect(response["url"]) as websocket:
        print("\n################ Begin session ################\n")
        loop = asyncio.get_running_loop()
        # loop.add_signal_handler(
        #     signal.SIGINT,
        #     loop.create_task,
        #     stop_recording(websocket),
        # )

        # Correct signal handling with lambda function
        loop.add_signal_handler(
            signal.SIGINT,
            lambda: asyncio.create_task(stop_recording(websocket))
        )

        send_audio_task = asyncio.create_task(send_audio(websocket))
        print_messages_task = asyncio.create_task(print_messages_from_socket(websocket))
        await asyncio.wait(
            [send_audio_task, print_messages_task],
        )

def print_streaming_configuration(streaming_configuration: StreamingConfiguration):
    print("\n################ Streaming Configuration ################\n")
    for param in streaming_configuration:
        print(f"{param}: {streaming_configuration[param]}")

if __name__ == "__main__":
    tracemalloc.start()
    print_streaming_configuration(streaming_configuration=STREAMING_CONFIGURATION)
    asyncio.run(main())
