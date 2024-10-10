import os
from azure.identity import AzureCliCredential
from azure.mgmt.resource import ResourceManagementClient
from azure.mgmt.storage import StorageManagementClient
from azure.mgmt.storage.models import StorageAccountCreateParameters, Sku, SkuName, Kind

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

import configuration

# Define variables
subscription_id = os.getenv("AZURE_SUBSCRIPTION_ID")
resource_group_name = configuration.resource_group_name
location = configuration.location
storage_account_name = configuration.storage_account_name

# Authenticate using DefaultAzureCredential (ensure 'az login' is done or set up Service Principal)
credential = AzureCliCredential()

# Create clients
resource_client = ResourceManagementClient(credential, subscription_id)
storage_client = StorageManagementClient(credential, subscription_id)


# Create or check if the resource group exists
def create_resource_group():
    resource_group_params = {"location": location}
    resource_group = resource_client.resource_groups.create_or_update(
        resource_group_name,
        resource_group_params
    )
    print(f"Resource group '{resource_group_name}' created or already exists.")
    return resource_group

# Create or check if the storage account exists
def create_storage_account():
    # Check if the storage account exists
    accounts = list(storage_client.storage_accounts.list_by_resource_group(resource_group_name))
    account_exists = any(acc.name == storage_account_name for acc in accounts)

    if not account_exists:
        storage_async_operation = storage_client.storage_accounts.begin_create(
            resource_group_name,
            storage_account_name,
            StorageAccountCreateParameters(
                sku=Sku(name=SkuName.STANDARD_LRS),
                kind=Kind.STORAGE_V2,
                location=location,
            )
        )
        storage_account = storage_async_operation.result()
        print(f"Storage account '{storage_account_name}' created.")
    else:
        print(f"Storage account '{storage_account_name}' already exists.")

# Run the functions to create the resources
create_resource_group()
create_storage_account()
