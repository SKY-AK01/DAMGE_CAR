import argparse
import os
import sys
from dotenv import load_dotenv
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes
from azure.identity import DefaultAzureCredential

def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="Upload dataset to Azure ML.")
    parser.add_argument("--local_dir", required=True, help="Path to local dataset directory")
    parser.add_argument("--name", default=os.environ.get("AZURE_DATASET_NAME", "car_parts_seg_dataset"), help="Azure ML dataset name")
    parser.add_argument("--version", default=os.environ.get("AZURE_DATASET_VERSION", "1"), help="Dataset version")
    args = parser.parse_args()

    try:
        credential = DefaultAzureCredential()
        ml_client = MLClient(
            credential,
            os.environ.get("AZURE_SUBSCRIPTION_ID"),
            os.environ.get("AZURE_RESOURCE_GROUP"),
            os.environ.get("AZURE_WORKSPACE_NAME")
        )
    except Exception as e:
        print(f"[ERROR] Azure authentication failed: {e}")
        sys.exit(1)

    print(f"[*] Uploading '{args.local_dir}' as dataset '{args.name}' v{args.version}...")
    my_data = Data(
        path=args.local_dir,
        type=AssetTypes.URI_FOLDER,
        description="Car parts dataset",
        name=args.name,
        version=args.version
    )
    
    try:
        ml_client.data.create_or_update(my_data)
        print(f"[OK] Successfully registered dataset '{args.name}' version {args.version}.")
    except Exception as e:
        print(f"[ERROR] Failed to upload dataset: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
