import requests
import time
import os
from dotenv import load_dotenv
load_dotenv()
from azure.storage.blob import BlobServiceClient, generate_blob_sas, BlobSasPermissions
from datetime import datetime, timedelta
import os

# Configuration
connection_string = os.getenv('AZURE_STORAGE_CONNECTION_STRING')  # Your connection string
container_name = "stt-datasets-english-audios"  # Your container name
blob_name = "0003b9543cad190a9f9217200a768a23.wav"  # Name of your audio file

# Create BlobServiceClient
blob_service_client = BlobServiceClient.from_connection_string(connection_string)

# Define the expiration time for the SAS token
sas_token_expiration = datetime.utcnow() + timedelta(minutes=15)  # Token valid for 1 hour

# Generate SAS token for the blob
sas_token = generate_blob_sas(
    account_name=blob_service_client.account_name,
    container_name=container_name,
    blob_name=blob_name,
    account_key=blob_service_client.credential.account_key,
    permission=BlobSasPermissions(read=True),
    expiry=sas_token_expiration
)

# Construct the full URL with SAS token
blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
audio_url_with_sas = f"{blob_client.url}?{sas_token}"

print("Generated audio URL with SAS token:", audio_url_with_sas)


def make_fetch_request(url, headers, method='GET', data=None):
    if method == 'POST':
        response = requests.post(url, headers=headers, json=data)
    else:
        response = requests.get(url, headers=headers)
    return response.json()

gladia_key = os.getenv("GLADIA_API_KEY")
request_data = {"audio_url": audio_url_with_sas}
gladia_url = "https://api.gladia.io/v2/transcription/"

headers = {
    "x-gladia-key": gladia_key,
    "Content-Type": "application/json"
}

print("- Sending initial request to Gladia API...")
initial_response = make_fetch_request(gladia_url, headers, 'POST', request_data)

print("Initial response with Transcription ID:", initial_response)
result_url = initial_response.get("result_url")

if result_url:
    while True:
        print("Polling for results...")
        poll_response = make_fetch_request(result_url, headers)
        
        if poll_response.get("status") == "done":
            print("- Transcription done: \n")
            print(poll_response.get("result", {}).get("transcription", {}).get("full_transcript"))
            break
        else:
            print("Transcription status:", poll_response.get("status"))
        time.sleep(1)

