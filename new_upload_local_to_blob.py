import requests
import os

# --- Configuration ---
FUNCTION_BASE_URL = "https://func-sla-catboost-train-uat-eastus.azurewebsites.net"

# FIX: Aligned endpoint to match the route in function_app.py
UPLOAD_ENDPOINT = f"{FUNCTION_BASE_URL}/api/upload_blob"

LOCAL_FILE_PATH = r"C:\path\to\your\data.csv"
TARGET_BLOB_NAME = "data/input_records.csv"

def upload_csv_via_function(local_path, blob_path):
    if not os.path.exists(local_path):
        print(f"Error: File {local_path} not found.")
        return

    # Parameters aligned with what storage_upload expects
    params = {
        "blob": blob_path
    }

    print(f"Uploading {local_path} to {blob_path}...")

    with open(local_path, "rb") as f:
        try:
            response = requests.post(
                UPLOAD_ENDPOINT,
                params=params,
                data=f,
                headers={"Content-Type": "text/csv"},
                verify=True # Changed to True for standard production security
            )

            if response.status_code == 200:
                print("✅ Upload Successful!")
                print("Response:", response.json())
            else:
                print(f"❌ Upload Failed ({response.status_code})")
                print("Error Detail:", response.text)

        except Exception as e:
            print(f"❌ An error occurred: {e}")

if __name__ == "__main__":
    upload_csv_via_function(LOCAL_FILE_PATH, TARGET_BLOB_NAME)

