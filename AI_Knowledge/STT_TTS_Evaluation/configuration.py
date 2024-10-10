import os

subscription_id = os.getenv("AZURE_SUBSCRIPTION_ID")  # Your Azure subscription ID
resource_group_name = "stt_evaluation"  # Replace with your desired resource group name
location = "eastus"  # Azure region for the resource group and storage account
storage_account_name = "sttdatasets001"  # Replace with your desired storage account name

# CONTAINERS
container_name_gt = "stt-datasets-gt"  # Replace with your container name
container_name_english_audios = "stt-datasets-english-audios"  # Replace with your container name

# LOCAL FOLDERS / PATHS
local_tsv_folder = "stt_datasets/gt"  # Local folder containing .tsv files
local_wav_folder = "stt_datasets/audios/english"  # Local folder containing .wav files

# AZURE TABLES
table_name_english = "EnglishTranscriptions"  # Name of your table

# AZURE FUNCTION
app_service_plan_name = "sttevaluationplan"
function_app_name = "sttevaluation001"
function_name_gladiastt = "gladiastt001"
runtime = "python"  # Runtime can be 'python', 'node', etc.
python_version = "Python|3.10"