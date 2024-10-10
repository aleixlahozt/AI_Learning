import azure.functions as func
import logging
import os
from azure.storage.blob import BlobServiceClient, generate_blob_sas, BlobSasPermissions
from datetime import datetime, timedelta
import requests
from azure.data.tables import TableServiceClient
from time import sleep

app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)

@app.route(route="healthcheck")
def health_check(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Health check request received.')

    # Perform any necessary checks (e.g., check database connectivity, external services, etc.)
    # For simplicity, we'll just return a 200 OK response.

    return func.HttpResponse(
        "Healthy",
        status_code=200
    )


@app.route(route="gladiastt001")
def gladiastt001(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Audio transcription request received by Function App')
    
    # Get parameters from the request
    row_id = req.params.get('row_id')
    file_url = req.params.get('file_url')
    partition_key = req.params.get('partition_key')

    # Validate parameters
    if not row_id or not file_url or not partition_key:
        return func.HttpResponse(
            "Missing parameters: row_id, file_url, and partition_key are required.",
            status_code=400
        )
    
    logging.info(f"file url: {file_url}")

    # Configuration
    connection_string = os.getenv("FUNCTIONAPP_AZURE_STORAGE_CONNECTION_STRING")
    table_name = os.getenv("FUNCTIONAPP_ENGLISH_TABLE_NAME")
    gladia_key = os.getenv("FUNCTIONAPP_GLADIA_API_KEY")
    container_name = os.getenv("FUNCTIONAPP_ENGLISH_AUDIOS_CONTAINER_NAME")

    # Create BlobServiceClient
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)

    # Generate SAS token for the blob
    sas_token = generate_blob_sas(
        account_name=blob_service_client.account_name,
        container_name=container_name,
        blob_name=file_url.split('/')[-1],  # Extract blob name from URL
        account_key=blob_service_client.credential.account_key,
        permission=BlobSasPermissions(read=True),
        expiry=datetime.utcnow() + timedelta(minutes=15)  # Token valid for 15 minutes
    )

    audio_url_with_sas = f"{file_url}?{sas_token}"
    logging.info("Generated audio URL with SAS token: %s", audio_url_with_sas)

    # Call Gladia API
    request_data = {"audio_url": audio_url_with_sas}
    gladia_url = "https://api.gladia.io/v2/transcription/"

    headers = {
        "x-gladia-key": gladia_key,
        "Content-Type": "application/json"
    }

    logging.info("- Sending initial request to Gladia API...")
    initial_response = make_fetch_request(gladia_url, headers, 'POST', request_data)

    # Handle initial response from Gladia
    if not initial_response.get("result_url"):
        return func.HttpResponse(
            f"Error calling Gladia API: {initial_response.get('message')}",
            status_code=500
        )

    logging.info("Initial response with Transcription ID: %s", initial_response)
    result_url = initial_response.get("result_url")

    # Polling for the transcription result
    max_retries = 60  # For example, 60 retries with 1 second sleep each
    retries = 0

    if result_url:
        while retries < max_retries:
            logging.info("Polling for results...")
            poll_response = make_fetch_request(result_url, headers)

            if poll_response.get("status") == "done":
                transcription = poll_response.get("result", {}).get("transcription", {}).get("full_transcript")
                logging.info(f"- Transcription done: {transcription}")

                # Update Azure Table with transcription
                table_client = TableServiceClient.from_connection_string(connection_string).get_table_client(table_name)
                table_client.upsert_entity({
                    "PartitionKey": partition_key,
                    "RowKey": row_id,
                    "gladia_transcription": transcription
                })
                return func.HttpResponse(f"Transcription updated: {transcription}")
            else:
                logging.info("Polling for transcription...")
                sleep(1)
                retries += 1

        return func.HttpResponse("Transcription polling timed out.", status_code=408)

    return func.HttpResponse("No result URL found.", status_code=400)


def make_fetch_request(url, headers, method='GET', data=None):
    try:
        if method == 'POST':
            response = requests.post(url, headers=headers, json=data)
        else:
            response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as http_err:
        logging.error(f"HTTP error occurred: {http_err}")
        return {"error": str(http_err)}
    except Exception as err:
        logging.error(f"Other error occurred: {err}")
        return {"error": str(err)}
