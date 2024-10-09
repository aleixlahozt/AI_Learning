import os
import pandas as pd
from azure.storage.blob import BlobServiceClient
from io import StringIO
from dotenv import load_dotenv
load_dotenv()

# Configuration
connection_string = os.getenv('AZURE_STORAGE_CONNECTION_STRING')  # Set this in your environment
container_name = "stt-datasets-gt"  # Your container name
input_file_name = "english.tsv"  # Name of the TSV file
output_file_name = "english_with_urls.csv"  # Name of the output CSV file
audio_container_url = "https://sttdatasets001.blob.core.windows.net/stt-datasets-english-audios"  # Base URL for audio files

# Create a BlobServiceClient
blob_service_client = BlobServiceClient.from_connection_string(connection_string)

# Read the TSV file from Blob Storage
blob_client = blob_service_client.get_blob_client(container_name, input_file_name)

# Download the TSV file content
tsv_content = blob_client.download_blob().readall().decode('utf-8')

# Create a DataFrame from the TSV content
data = pd.read_csv(StringIO(tsv_content), sep='\t')

# Generate the DataFrame with required columns
data['file_url'] = audio_container_url + '/' + data['hash_name'] + '.wav'
result_df = data[['transcription', 'file_url']]

# Display the result DataFrame
print(result_df)

# Convert the DataFrame to CSV format
output_csv_content = result_df.to_csv(index=False)

# Upload the resulting CSV file back to Blob Storage
output_blob_client = blob_service_client.get_blob_client(container_name, output_file_name)
output_blob_client.upload_blob(output_csv_content, overwrite=True)

print(f"Uploaded {output_file_name} to {container_name} successfully.")
