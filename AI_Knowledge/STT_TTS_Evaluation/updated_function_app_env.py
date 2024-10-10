from azure.identity import DefaultAzureCredential
from azure.mgmt.web import WebSiteManagementClient
from dotenv import load_dotenv
import os
import configuration

load_dotenv()
subscription_id = os.getenv("FUNCTIONAPP_AZURE_SUBSCRIPTION_ID")
resource_group_name = configuration.resource_group_name
function_app_name = configuration.function_app_name

# Authenticate
credential = DefaultAzureCredential()
web_client = WebSiteManagementClient(credential, subscription_id)

# Load all environment variables from .env into a dictionary
app_settings = {key: os.getenv(key) for key in os.environ.keys() if key.startswith("FUNCTIONAPP")}  # Adjust prefixes as needed

# Retrieve current application settings
current_settings = web_client.web_apps.list_application_settings(resource_group_name, function_app_name)

# # Combine existing settings with new ones, preserving existing values
combined_settings = current_settings.properties
combined_settings.update(app_settings)

# Update the application settings for the function app
web_client.web_apps.update_application_settings(
    resource_group_name,
    function_app_name,
    {"properties": combined_settings}
)

print("Function App settings updated successfully.")
