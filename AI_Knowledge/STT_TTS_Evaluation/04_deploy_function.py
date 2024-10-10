from azure.identity import DefaultAzureCredential
from azure.mgmt.web import WebSiteManagementClient
from dotenv import load_dotenv
import os
import shutil

load_dotenv()
import configuration

# Configuration
subscription_id = os.getenv("AZURE_SUBSCRIPTION_ID")
resource_group_name = configuration.resource_group_name
function_app_name = configuration.function_app_name
function_name = configuration.function_name_gladiastt
location = configuration.location

# Authenticate
credential = DefaultAzureCredential()
web_client = WebSiteManagementClient(credential, subscription_id)

function_directory = f"azure_functions/{function_name}"  # Directory for the function code
zip_file_path = f"{function_directory}.zip"  # Path to the zip file

shutil.make_archive(function_directory, 'zip', function_directory)

# Upload the zip file directly to the function app
with open(zip_file_path, 'rb') as zip_file:
    # The proper method to upload a ZIP file directly to the function app
    web_client.web_apps.begin_create_or_update_source_control(
        resource_group_name,
        function_app_name,
        "production",
        zip_file.read()
    ).result()

# Load all environment variables from .env into a dictionary
app_settings = {key: os.getenv(key) for key in os.environ.keys()}  # Adjust prefixes as needed

# Update the application settings for the function app
web_client.web_apps.update_application_settings(
    resource_group_name,
    function_app_name,
    {"properties": app_settings}
)

print(f"Function '{function_name}' deployed successfully to '{function_app_name}'.")
