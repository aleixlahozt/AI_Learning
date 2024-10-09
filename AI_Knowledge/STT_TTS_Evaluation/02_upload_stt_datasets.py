import os
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv
load_dotenv()
# Configuration
connection_string = os.getenv('AZURE_STORAGE_CONNECTION_STRING')  # Set this in your environment
container_name_gt = "stt-datasets-gt"  # Replace with your container name
container_name_english_audios = "stt-datasets-english-audios"  # Replace with your container name

local_tsv_folder = "stt_datasets/gt"  # Local folder containing .tsv files
local_wav_folder = "stt_datasets/audios/english"  # Local folder containing .wav files

# Create a BlobServiceClient
blob_service_client = BlobServiceClient.from_connection_string(connection_string)

# Create the container if it doesn't exist
def create_container_if_not_exists(container_name):
    container_client = blob_service_client.get_container_client(container_name)
    try:
        container_client.create_container()
        print(f"Container '{container_name}' created.")
    except Exception as e:
        print(f"Container '{container_name}' already exists: {e}")
    
    return container_client

# Function to upload files
def upload_files(local_folder, container_name):
    container_client = create_container_if_not_exists(container_name)

    for root, _, files in os.walk(local_folder):
        for file in files:
            if file.endswith('.tsv') or file.endswith('.wav'):
                local_file_path = os.path.join(root, file)
                blob_client = container_client.get_blob_client(file)

                print(f"Uploading {file} to Blob Storage...")
                with open(local_file_path, "rb") as data:
                    blob_client.upload_blob(data, overwrite=True)
                print(f"{file} uploaded successfully.")


# Upload .tsv files
upload_files(local_tsv_folder, container_name_gt)

# Upload .wav files
upload_files(local_wav_folder, container_name_english_audios)

print("All files uploaded successfully.")
