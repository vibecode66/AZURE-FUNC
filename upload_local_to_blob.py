import requests
import os

# --- Configuration ---
# Replace with your actual Azure Function URL
FUNCTION_BASE_URL = "https://func-sla-catboost-train-uat-eastus.azurewebsites.net"
UPLOAD_ENDPOINT = f"{FUNCTION_BASE_URL}/api/storage/upload"

# The specific name of the function as it appears in the Azure Portal
AZURE_FUNCTION_NAME = "upload_local_to_blob"

# Local file details
LOCAL_FILE_PATH = r"C:\path\to\your\data.csv"
TARGET_BLOB_NAME = "data/input_records.csv"  # The path inside the container
CONTAINER_NAME = "your-container-name"  # Optional: if your function requires it


def upload_csv_via_function(local_path, blob_path, func_name):
    if not os.path.exists(local_path):
        print(f"Error: File {local_path} not found.")
        return

    # Parameters to tell the Azure Function where to put the file
    params = {
        "blob": blob_path,
        "overwrite": "true",
        "function_name": func_name  # Added function name here
    }

    # If your function uses a specific container variable
    if CONTAINER_NAME:
        params["container"] = CONTAINER_NAME

    print(f"Uploading {local_path} to {blob_path} via {func_name}...")

    with open(local_path, "rb") as f:
        try:
            # Send the file as raw binary data
            response = requests.post(
                UPLOAD_ENDPOINT,
                params=params,
                data=f,
                headers={"Content-Type": "text/csv"},
                verify=False  # Set to True if using valid production SSL
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
    # Now passing the AZURE_FUNCTION_NAME variable
    upload_csv_via_function(LOCAL_FILE_PATH, TARGET_BLOB_NAME, AZURE_FUNCTION_NAME)

