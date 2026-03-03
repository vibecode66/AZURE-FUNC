import requests
import os

# --- Configuration ---
# Pointing to the 'upload-csv' route instead of the training route
FUNCTION_BASE_URL = "https://func-sla-catboost-train-uat-eastus.azurewebsites.net"
UPLOAD_ENDPOINT = f"{FUNCTION_BASE_URL}/api/upload-csv"

# Local file configuration
LOCAL_FILE_PATH = r"C:\path\to\your\historical_data.xlsx"

# This defines the folder and filename inside your blob container
TARGET_BLOB_NAME = "training_data.xlsx"
TARGET_PREFIX = "data" 

def save_file_to_blob(local_path, blob_name, prefix):
    if not os.path.exists(local_path):
        print(f"❌ Error: {local_path} not found.")
        return

    # These parameters match the 'upload_csv' function logic: 
    # req.params.get("blob") and req.params.get("prefix")
    params = {
        "blob": blob_name,
        "prefix": prefix,
        "overwrite": "true"
    }

    print(f"🚀 Uploading {os.path.basename(local_path)} to storage (No training)...")

    try:
        with open(local_path, "rb") as f:
            # The 'upload_csv' function prefers multipart/form-data with field "file"
            files = {"file": (os.path.basename(local_path), f, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")}
            
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
