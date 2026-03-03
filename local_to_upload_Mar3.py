import requests
import os

# --- Configuration ---
# Updated base URL and endpoint
FUNCTION_BASE_URL = "https://func-sla-catboost-train-uat-eastus.azurewebsites.net"
UPLOAD_ENDPOINT = f"{FUNCTION_BASE_URL}/api/upload-csv"

# Updated Local file path
LOCAL_FILE_PATH = r"C:\cosmic_case_sla_prediction\model_training_local\data\raw\training_raw_dataset_catboost.xlsx"

# The path inside the container
# Matches the filename from your local path and places it in the 'data' folder
TARGET_BLOB_NAME = "training_raw_dataset_catboost.xlsx"
TARGET_PREFIX = "data" 

def save_file_to_blob(local_path, blob_name, prefix):
    """
    Uploads a local Excel or CSV file to Azure Blob Storage via the Azure Function.
    This uses the 'upload-csv' endpoint which simply saves the file.
    """
    if not os.path.exists(local_path):
        print(f"❌ Error: {local_path} not found.")
        return

    # Parameters required by the Azure Function's 'upload_csv' logic
    params = {
        "blob": blob_name,
        "prefix": prefix,
        "overwrite": "true"
    }

    print(f"🚀 Uploading {os.path.basename(local_path)} to Azure Storage...")

    try:
        with open(local_path, "rb") as f:
            # Multipart/form-data upload using field "file" as preferred by the function
            # Content-type set for Excel (.xlsx)
            files = {
                "file": (
                    os.path.basename(local_path), 
                    f, 
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            }
            
            response = requests.post(
                UPLOAD_ENDPOINT,
                params=params,
                files=files,
                verify=True,
                timeout=60
            )

            if response.status_code == 200:
                print("✅ File saved successfully!")
                print("Server Response:", response.json())
            else:
                print(f"❌ Failed ({response.status_code}):", response.text)

    except Exception as e:
        print(f"❌ Connection Error: {e}")

if __name__ == "__main__":
    save_file_to_blob(LOCAL_FILE_PATH, TARGET_BLOB_NAME, TARGET_PREFIX)
