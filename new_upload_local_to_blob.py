import requests
import os

# --- Configuration (Matching your Azure Function Settings) ---
FUNCTION_BASE_URL = "https://func-sla-catboost-train-uat-eastus.azurewebsites.net"
TRAIN_ENDPOINT = f"{FUNCTION_BASE_URL}/api/catboost_train"

# IMPORTANT: This must match the 'raw_data' path inside your config.yaml in Blob Storage
# If your config.yaml says "data/input.csv", TARGET_BLOB_NAME must be "data/input.csv"
TARGET_BLOB_NAME = "data/training_data.csv"
LOCAL_FILE_PATH = r"C:\path\to\your\historical_data.csv"

# The identifier your logic uses (passed as a query param)
IDENTIFIER = "catboost_model_v1"

def upload_and_train(local_path, blob_path, identifier):
    if not os.path.exists(local_path):
        print(f"❌ Error: {local_path} not found.")
        return

    # 1. Prepare Parameters
    # 'blob' tells the mediator where to save the file
    # 'identifier' provides the context for your training logic
    params = {
        "blob": blob_path,
        "identifier": identifier
    }

    print(f"Step 1: Uploading {local_path} to Blob path: {blob_path}...")

    with open(local_path, "rb") as f:
        try:
            # We send the file to the function.
            # Your Azure Function's Managed Identity will handle the write to Storage.
            response = requests.post(
                TRAIN_ENDPOINT,
                params=params,
                data=f,
                headers={"Content-Type": "text/csv"},
                timeout=300 # Training can take a long time, so we use a high timeout
            )

            if response.status_code == 200:
                print("✅ Process Successful!")
                result = response.json()
                print("--- Azure Response ---")
                print(f"Message: {result.get('message')}")
                print(f"Features Used: {result.get('features_used')}")
            else:
                print(f"❌ Request Failed ({response.status_code})")
                print("Error Detail:", response.text)

        except requests.exceptions.Timeout:
            print("⚠️ Timeout: The training started, but the connection closed before it finished.")
            print("Check your Azure Logs; the model is likely still training in the background.")
        except Exception as e:
            print(f"❌ An error occurred: {e}")

if __name__ == "__main__":
    upload_and_train(LOCAL_FILE_PATH, TARGET_BLOB_NAME, IDENTIFIER)

