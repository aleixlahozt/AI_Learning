import logging
import os
import requests
from azure.storage.blob import BlobServiceClient, generate_blob_sas, BlobSasPermissions
from datetime import datetime, timedelta
import azure.functions as func
from azure.data.tables import TableServiceClient
from time import sleep  # Changed to sleep for better readability

def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Processing a row for transcription.')

    # Get parameters from the request
    row_id = req.params.get('row_id')
    file_url = req.params.get('file_url')

    # Configuration
    connection_string = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
    table_name = os.getenv('ENGLISH_TABLE_NAME')
    gladia_key = os.getenv("GLADIA_API_KEY")
    container_name = os.getenv("ENGLISH_AUDIOS_CONTAINER_NAME")

    # Create BlobServiceClient
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)

    # Generate SAS token for the blob
    sas_token = generate_blob_sas(
        account_name=blob_service_client.account_name,
        container_name=container_name,
        blob_name=file_url.split('/')[-1],  # Extract blob name from URL,
        account_key=blob_service_client.credential.account_key,
        permission=BlobSasPermissions(read=True),
        expiry=datetime.utcnow() + timedelta(minutes=15)  # Token valid for 15 minutes
    )

    audio_url_with_sas = f"{file_url}?{sas_token}"
    print("Generated audio URL with SAS token:", audio_url_with_sas)

    # Call Gladia API
    request_data = {"audio_url": audio_url_with_sas}
    gladia_url = "https://api.gladia.io/v2/transcription/"

    headers = {
        "x-gladia-key": gladia_key,
        "Content-Type": "application/json"
    }

    initial_response = requests.post(gladia_url, headers=headers, json=request_data)

    if initial_response.status_code != 200:
        return func.HttpResponse(
            "Error calling Gladia API.",
            status_code=500
        )

    result_url = initial_response.json().get("result_url")

    # Polling for the transcription result
    if result_url:
        while True:
            poll_response = requests.get(result_url, headers=headers)
            if poll_response.json().get("status") == "done":
                transcription = poll_response.json().get("result", {}).get("transcription", {}).get("full_transcript")
                # Update Azure Table with transcription
                table_client = TableServiceClient.from_connection_string(connection_string).get_table_client(table_name)
                table_client.update_entity({
                    "PartitionKey": "YourPartitionKey",  # Adjust according to your table schema
                    "RowKey": row_id,  # This should match the row ID you are processing
                    "gladia_transcription": transcription
                })
                return func.HttpResponse(f"Transcription updated: {transcription}")
            else:
                logging.info("Polling for transcription...")
                sleep(1)
    return func.HttpResponse("No result URL found.", status_code=400)
