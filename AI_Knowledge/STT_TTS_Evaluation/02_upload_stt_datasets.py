import os
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv
load_dotenv()
import configuration
from io import StringIO
import pandas as pd
from azure.data.tables import TableServiceClient, TableEntity

# STORAGE ACCOUNT
storage_account_name = configuration.storage_account_name
connection_string = os.getenv('AZURE_STORAGE_CONNECTION_STRING')  # Set this in your environment

# CONTAINERS
container_name_gt = configuration.container_name_gt
container_name_english_audios = configuration.container_name_english_audios

# LOCAL FOLDERS / PATHS
local_tsv_folder = configuration.local_tsv_folder
local_wav_folder = configuration.local_wav_folder

# GT FILENAMES
english_tsv_file = "english.tsv"  # Name of the TSV file
english_csv_file = "english_with_urls.csv"  # Name of the output CSV file

# AZURE TABLES
table_name = configuration.table_name_english
audio_container_url = f"https://{storage_account_name}.blob.core.windows.net/{container_name_english_audios}"  # Base URL for audio files

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
    uploaded_files = []
    container_client = create_container_if_not_exists(container_name)

    for root, _, files in os.walk(local_folder):
        for file in files:
            if file.endswith('.tsv') or file.endswith('.wav'):
                local_file_path = os.path.join(root, file)
                blob_client = container_client.get_blob_client(file)

                print(f"\tUploading {file} to Blob Storage...")
                with open(local_file_path, "rb") as data:
                    blob_client.upload_blob(data, overwrite=True)
                print(f"\t\t{file} uploaded successfully.")
                uploaded_files.append(file)
    return uploaded_files


# Upload .tsv files
print(f"0. Uploading tsv files to container -{container_name_gt}-")
_ = upload_files(local_tsv_folder, container_name_gt)
print("-"*50+"\n")

# Upload .wav files
print(f"1. Uploading wav files to container -{container_name_english_audios}-")
uploaded_wav_files = upload_files(local_wav_folder, container_name_english_audios)
print("-"*50+"\n")

# Read and Download the TSV file from Blob Storage
print(f"2. Downloading {english_tsv_file} file from blob storage")
blob_client = blob_service_client.get_blob_client(container_name_gt, english_tsv_file)
tsv_content = blob_client.download_blob().readall().decode('utf-8')
# Create a DataFrame from the TSV content
data = pd.read_csv(StringIO(tsv_content), sep='\t')
# Generate the DataFrame with required columns
data['file_url'] = audio_container_url + '/' + data['hash_name'] + '.wav'
result_df = data[['transcription', 'file_url']]
# Display the result DataFrame
print(result_df.head())

print("-"*50+"\n")
print(f"3. Creating Azure Table with name {table_name} and uploading a row for each audio...")
# Create a TableServiceClient
table_service_client = TableServiceClient.from_connection_string(conn_str=connection_string)

# Create the table if it doesn't exist
try:
    table_client = table_service_client.create_table(table_name=table_name)
    print(f"Table '{table_name}' created.")
except Exception as e:
    print(f"Table '{table_name}' may already exist or another error occurred: {e}")
    table_client = table_service_client.get_table_client(table_name=table_name)

# Iterate through each row in the DataFrame and add entities to the table
for index, row in result_df.iterrows():
    if os.path.basename(row["file_url"]) in uploaded_wav_files:
        entity = TableEntity()
        entity["PartitionKey"] = "transcriptionPartition"  # You can modify this as needed
        entity["RowKey"] = str(index)  # Unique identifier for each row
        entity["transcription"] = row["transcription"]
        entity["file_url"] = row["file_url"]
        entity["gladia_transcription"] = ""  # Initialize as empty
        entity["deepgram_transcription"] = ""  # Initialize as empty
        entity["azure_transcription"] = ""  # Initialize as empty
        entity["speechmatics_transcription"] = ""  # Initialize as empty
        
        # Insert entity into the table
        table_client.create_entity(entity)
        print(f"\tEntity for row {index} added.")
