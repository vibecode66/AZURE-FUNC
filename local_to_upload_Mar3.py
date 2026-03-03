import requests
import os

# --- Configuration ---
FUNCTION_BASE_URL = "https://func-sla-catboost-train-uat-eastus.azurewebsites.net"
TRAIN_ENDPOINT = f"{FUNCTION_BASE_URL}/api/catboost_train"

# Local file configuration (Updated for Excel)
LOCAL_FILE_PATH = r"C:\path\to\your\historical_data.xlsx"

# REQUIRED: Must match the 'raw_data' path in your config.yaml
# Ensure it ends in .xlsx so the Azure Function knows to use pd.read_excel
TARGET_BLOB_NAME = "data/training_data.xlsx"

def upload_and_train(local_path, blob_path):
    if not os.path.exists(local_path):
        print(f"❌ Error: {local_path} not found.")
        return

    # 'blob' parameter is required by your Function's req.params.get('blob') logic
    params = {"blob": blob_path}

    # Updated headers for Excel files
    headers = {
        "Content-Type": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "User-Agent": "Python-Requests"
    }

    print(f"🚀 Uploading Excel data to '{blob_path}' and starting training...")

    try:
        with open(local_path, "rb") as f:
            # Sending raw binary data as the request body for req.get_body()
            response = requests.post(
                TRAIN_ENDPOINT,
                params=params,
                data=f,
                headers=headers,
                verify=True,
                timeout=900  # High timeout for training
            )

            if response.status_code == 200:
                print("✅ Success!")
                print("Response:", response.json())
            else:
                print(f"❌ Failed ({response.status_code}):", response.text)

    except Exception as e:
        print(f"❌ Connection Error: {e}")

if __name__ == "__main__":
    upload_and_train(LOCAL_FILE_PATH, TARGET_BLOB_NAME)
