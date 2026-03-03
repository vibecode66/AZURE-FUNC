import streamlit as st
import pandas as pd
import io
import os
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient

# --- Hardcoded Configuration (Replaces os.getenv) ---
STORAGE_ACCOUNT_URL = "https://stslalightgbmuateu.blob.core.windows.net"
BLOB_CONTAINER_NAME = "train-data"
TARGET_FOLDER = "data"


def _get_credential() -> DefaultAzureCredential:
    """
    Uses DefaultAzureCredential.
    Locally: Will use your 'az login' session.
    In Azure: Will use the App's Managed Identity.
    """
    return DefaultAzureCredential()


def _get_container_client():
    """Get a container client using the hardcoded URL and Container Name"""
    credential = _get_credential()
    svc = BlobServiceClient(account_url=STORAGE_ACCOUNT_URL, credential=credential)
    return svc.get_container_client(BLOB_CONTAINER_NAME)


def upload_to_azure(file_bytes, blob_name):
    """Uploads the file bytes to the specific container and folder."""
    try:
        container_client = _get_container_client()

        # Path inside the container: e.g., data/my_file.xlsx
        full_blob_path = f"{TARGET_FOLDER}/{blob_name}"
        blob_client = container_client.get_blob_client(full_blob_path)

        # Upload binary data
        blob_client.upload_blob(file_bytes, overwrite=True)
        return True, full_blob_path
    except Exception as e:
        return False, str(e)


# --- Streamlit UI ---
st.set_page_config(page_title="Azure Data Uploader", page_icon="☁️")
st.title("📂 Excel to Azure Blob Storage")

# Display current targets for clarity
st.sidebar.header("Target Settings")
st.sidebar.text(f"Account: {STORAGE_ACCOUNT_URL.split('//')[1].split('.')[0]}")
st.sidebar.text(f"Container: {BLOB_CONTAINER_NAME}")
st.sidebar.text(f"Folder: {TARGET_FOLDER}")

uploaded_file = st.file_uploader("Choose an Excel file", type=["xlsx", "xls"])

if uploaded_file is not None:
    try:
        # Load into memory for preview
        df = pd.read_excel(uploaded_file)
        st.write("### Preview of Uploaded Data")
        st.dataframe(df.head(10))

        if st.button("🚀 Upload to Azure Storage"):
            with st.spinner("Uploading..."):
                # Ensure we read the file from the start
                uploaded_file.seek(0)
                file_bytes = uploaded_file.read()

                success, result = upload_to_azure(file_bytes, uploaded_file.name)

                if success:
                    st.success(f"✅ Successfully uploaded to: {result}")
                else:
                    st.error(f"❌ Upload Failed: {result}")
                    st.info("💡 Make sure you have run 'az login' in your terminal.")

    except Exception as e:
        st.error(f"Error processing file: {e}")

