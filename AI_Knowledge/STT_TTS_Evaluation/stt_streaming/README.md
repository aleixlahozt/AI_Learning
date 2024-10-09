# Testing GLADIA STT: [link](https://www.gladia.io)

* Developer [guide](https://docs.gladia.io/chapters/introduction/pages/introduction)
* Live Speech Recognition [guide](https://docs.gladia.io/chapters/speech-to-text-api/pages/live-speech-recognition)

### Live Speech Recognition:
* The Gladia Audio Transcription WebSocket API allows developers to connect to a streaming audio transcription endpoint and receive real-time transcription results.
* After establishing connection to a WebSocket (`wss://api.gladia.io/audio/text/audio-transcription`), an initial JSON object with the configuration must be sent, specifying all those [parameters](https://docs.gladia.io/chapters/speech-to-text-api/pages/live-speech-recognition#initial-configuration-message)