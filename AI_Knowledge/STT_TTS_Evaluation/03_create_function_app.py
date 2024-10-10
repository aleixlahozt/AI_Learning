import os
from dotenv import load_dotenv
import configuration
from azure.identity import AzureCliCredential
from azure.mgmt.resource import ResourceManagementClient
from azure.mgmt.storage import StorageManagementClient
from azure.mgmt.web import WebSiteManagementClient
from azure.mgmt.storage.models import StorageAccountCreateParameters, Sku, SkuName, Kind
load_dotenv()

subscription_id = os.getenv("AZURE_SUBSCRIPTION_ID")
resource_group_name = configuration.resource_group_name
location = configuration.location
storage_account_name = configuration.storage_account_name
function_app_name = configuration.function_app_name

# Authenticate using DefaultAzureCredential (ensure 'az login' is done or set up Service Principal)
credential = AzureCliCredential()

# Create clients
resource_client = ResourceManagementClient(credential, subscription_id)
storage_client = StorageManagementClient(credential, subscription_id)
web_client = WebSiteManagementClient(credential, subscription_id)

# Create function App
connection_string = os.getenv('AZURE_STORAGE_CONNECTION_STRING')  # Set this in your environment

app_service_plan = web_client.app_service_plans.begin_create_or_update(
    resource_group_name,
    configuration.app_service_plan_name,
    {
        "location": location,
        "sku": {
            "name": "B1",  # Basic tier
            "tier": "Free",
        },
        "kind": "FunctionApp",
        "reserved": True  # Required for Linux
    }
).result()
print(f"App Service Plan '{configuration.app_service_plan_name}' with B1 tier created successfully.")

function_app_async_operation = web_client.web_apps.begin_create_or_update(
    resource_group_name,
    function_app_name,
    {
        "location": location,
        "server_farm_id": app_service_plan.id,
        "kind": "functionapp",
        "properties": {
            "serverFarmId": app_service_plan.id,
            "siteConfig": {
                "linuxFxVersion": f"{configuration.python_version}",  # Specifies Python 3.10 runtime
                "appSettings": [
                    {"name": "AzureWebJobsStorage", "value": connection_string},
                    {"name": "FUNCTIONS_WORKER_RUNTIME", "value": f"{configuration.runtime}"},
                ]
            }
        }
    }
)
function_app = function_app_async_operation.result()
print(f"Function App '{function_app_name}' with {configuration.python_version} created successfully.")
