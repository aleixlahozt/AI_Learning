import os
import requests
from azure.data.tables import TableServiceClient
from concurrent.futures import ThreadPoolExecutor, as_completed
import configuration
from dotenv import load_dotenv
load_dotenv()

# Configuration
connection_string = os.getenv('FUNCTIONAPP_AZURE_STORAGE_CONNECTION_STRING')
table_client = TableServiceClient.from_connection_string(connection_string).get_table_client(configuration.table_name_english)

# Set the desired concurrency limit
MAX_CONCURRENT_INVOCATIONS = 10  # Adjust this number as needed

def invoke_function(row):
    row_id = row['RowKey']  # Adjust this based on your table schema
    file_url = row['file_url']  # Get the file URL
    partition_key = row['PartitionKey']
    function_url = f"https://{configuration.function_app_name}.azurewebsites.net/api/{configuration.function_name_gladiastt}?row_id={row_id}&file_url={file_url}&partition_key={partition_key}"
    response = requests.post(function_url)
    return response.json()

def main():
    entities = list(table_client.list_entities())  # Fetch all entities
    results = []

    # Use ThreadPoolExecutor to limit concurrency
    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_INVOCATIONS) as executor:
        future_to_row = {executor.submit(invoke_function, row): row for row in entities}

        for future in as_completed(future_to_row):
            row = future_to_row[future]
            try:
                result = future.result()
                results.append(result)
                print(f"Row {row['RowKey']} processed: {result}")
            except Exception as e:
                print(f"Error processing row {row['RowKey']}: {e}")

    # You can further process results if needed
    return results

if __name__ == "__main__":
    main()
