import streamlit as st
import pandas as pd
import json
import requests
import urllib3
import base64
import os
import io
# Correct import for Blob Storage
from azure.storage.blob import BlobServiceClient

# Suppress the "InsecureRequestWarning"
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


# --- Helper Function to Encode Local Image ---
def get_base64_image(image_path):
    try:
        if os.path.exists(image_path):
            with open(image_path, "rb") as img_file:
                return base64.b64encode(img_file.read()).decode()
        return None
    except Exception:
        return None


# --- Function to Save/Update CSV on Azure Blob ---
def save_results_to_blob(df_to_save, conn_str, container, blob_name):
    try:
        blob_service_client = BlobServiceClient.from_connection_string(conn_str)
        blob_client = blob_service_client.get_blob_client(container=container, blob=blob_name)

        # Check if file exists to append; otherwise create new
        if blob_client.exists():
            existing_data = blob_client.download_blob().readall()
            existing_df = pd.read_csv(io.BytesIO(existing_data))
            # Append new record
            final_df = pd.concat([existing_df, df_to_save], ignore_index=True)
            # Optional: Deduplicate by TicketNumber if it exists
            if 'TicketNumber' in final_df.columns:
                final_df = final_df.drop_duplicates(subset=['TicketNumber'], keep='last')
        else:
            final_df = df_to_save

        # Convert DF to CSV string
        csv_data = final_df.to_csv(index=False)
        blob_client.upload_blob(csv_data, overwrite=True)
        return True
    except Exception as e:
        st.error(f"Failed to save to Azure: {e}")
        return False


# --- Page Config ---
st.set_page_config(
    page_title="Cosmic Case SLA Prediction",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Load the Image ---
img_path = os.path.join("assets", "microsoft-logo.png")
img_base64 = get_base64_image(img_path)

# --- Custom CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Segoe+UI&display=swap');
    html, body, [class*="css"] { font-family: 'Segoe UI', sans-serif; background-color: #faf9f8; color: #201f1e; }
    section[data-testid="stSidebar"] { background-color: #ffffff; border-right: 1px solid #e1dfdd; }
    h1 { font-family: 'Segoe UI', sans-serif; font-weight: 600; color: #201f1e; font-size: 1.5rem; margin-top: 0px; }
    .stButton > button { background-color: #ffffff; color: #323130; border: 1px solid #8a8886; border-radius: 2px; font-size: 14px; }
    div[data-testid="stHorizontalBlock"] button[kind="primary"] { background-color: #0078d4; color: white; border: none; }
    .header-logo { height: 50px; width: auto; margin-right: 10px; }
</style>
""", unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.markdown("### Connection")
    api_url = st.text_input("Inference Endpoint",
                            value="https://func-sla-catboost-train-uat-eastus.azurewebsites.net/predict")

    st.markdown("---")
    st.markdown("### Azure Storage Settings")
    az_conn_str = st.text_input("Connection String", type="password", help="Azure Blob Storage Connection String")
    az_container = st.text_input("Container Name", value="predictions")
    az_blob_name = st.text_input("CSV Filename", value="sla_output_results.csv")

    st.markdown("---")
    st.markdown("### Request options")
    timeout_val = st.number_input("Connection Timeout (seconds)", value=240)

# --- Main Content Header ---
col1, col2 = st.columns([0.08, 0.92])
with col1:
    if img_base64:
        st.markdown(f'<img src="data:image/png;base64,{img_base64}" class="header-logo">', unsafe_allow_html=True)
    else:
        st.write("🏢")
with col2:
    st.markdown("# Cosmic Case SLA Prediction with CatBoost")

# --- Initialize Session State ---
if "data" not in st.session_state: st.session_state.data = None
if "built_payload" not in st.session_state: st.session_state.built_payload = None
if "inference_result" not in st.session_state: st.session_state.inference_result = None
if "final_display_df" not in st.session_state: st.session_state.final_display_df = None
if "original_input_data" not in st.session_state: st.session_state.original_input_data = None

# --- File Upload Logic ---
uploaded_file = st.file_uploader("Upload inference CSV/Excel", type=["csv", "xlsx", "xls"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
        st.session_state.data = df
        st.success(f"File loaded: {len(df)} records found.")
        st.dataframe(df, use_container_width=True)
    except Exception as e:
        st.error(f"Error reading file: {e}")

# --- Prediction Logic ---
if st.session_state.data is not None:
    row_num = st.number_input("Select Row for Prediction", min_value=1, max_value=len(st.session_state.data), value=1)

    if st.button("Build json payload"):
        row_data = st.session_state.data.iloc[row_num - 1].to_dict()
        row_data_cleaned = json.loads(pd.Series(row_data).to_json(date_format='iso'))
        st.session_state.built_payload = json.dumps(row_data_cleaned, indent=2)
        st.session_state.original_input_data = pd.DataFrame([row_data])
        st.code(st.session_state.built_payload, language="json")

    if st.button("Inference", type="primary", disabled=not st.session_state.built_payload):
        try:
            payload_json = json.loads(st.session_state.built_payload)
            with st.spinner(f"Sending request to Azure..."):
                response = requests.post(
                    api_url,
                    json=payload_json,
                    headers={"Content-Type": "application/json"},
                    timeout=timeout_val,
                    verify=False
                )

                if response.status_code == 200:
                    st.session_state.inference_result = response.json()
                    st.success("Success!")

                    results_list = st.session_state.inference_result.get("results", [])
                    if results_list:
                        results_df = pd.DataFrame(results_list)
                        prediction_data = pd.DataFrame()

                        if 'Ticket ID' in results_df.columns:
                            prediction_data['TicketNumber'] = results_df['Ticket ID']

                        if 'Predicted Duration (Mins)' in results_df.columns:
                            prediction_data['Predicted Duration (Mins)'] = pd.to_numeric(
                                results_df['Predicted Duration (Mins)'], errors='coerce').round(0).astype('int')

                        if 'Received Date' in results_df.columns:
                            received_dt = pd.to_datetime(results_df['Received Date'], errors='coerce')
                            prediction_data['msdyn_receiveddate'] = received_dt.dt.strftime('%m/%d/%Y %H:%M:%S')
                            prediction_data['msdyn_receiveddate'] = received_dt

                        if 'Predicted Resolution Date' in results_df.columns:
                            pred_dt = pd.to_datetime(results_df['Predicted Resolution Date'], errors='coerce')
                            prediction_data['Predicted Resolution Date'] = pred_dt.dt.strftime('%m/%d/%Y %H:%M:%S')
                            prediction_data['_pred_dt'] = pred_dt

                        has_actual_date = False
                        if 'Actual Resolution Date' in results_df.columns:
                            actual_dt = pd.to_datetime(results_df['Actual Resolution Date'], errors='coerce')
                            if actual_dt.notna().any():
                                has_actual_date = True
                                prediction_data['Actual Resolution Date'] = actual_dt.dt.strftime('%m/%d/%Y %H:%M:%S')
                                prediction_data['_actual_dt'] = actual_dt

                        if has_actual_date:
                            delta_seconds = (
                                        prediction_data['_actual_dt'] - prediction_data['_pred_dt']).dt.total_seconds()
                            delta_minutes = delta_seconds / 60
                            prediction_data['SLA Status'] = delta_minutes.apply(
                                lambda x: 'Delay' if x > 0 else ('Early' if x < 0 else 'On Time')
                            )
                        else:
                            prediction_data['SLA Status'] = 'Pending'

                        if st.session_state.original_input_data is not None:
                            original_df = st.session_state.original_input_data.copy()
                            for col in original_df.columns:
                                if original_df[col].dtype == 'object':
                                    try:
                                        dt_col = pd.to_datetime(original_df[col], errors='coerce')
                                        if dt_col.notna().any():
                                            original_df[col] = dt_col.dt.strftime('%m/%d/%Y %H:%M:%S')
                                    except:
                                        pass

                            columns_to_exclude = [col for col in original_df.columns if col in prediction_data.columns]
                            original_df_filtered = original_df.drop(columns=columns_to_exclude, errors='ignore')
                            st.session_state.final_display_df = pd.concat([prediction_data, original_df_filtered],
                                                                          axis=1)
                        else:
                            st.session_state.final_display_df = prediction_data

                        # --- LOGIC TO SAVE TO AZURE BLOB ---
                        if az_conn_str and az_container:
                            # Clean the dataframe (remove helper columns starting with _)
                            clean_df = st.session_state.final_display_df.copy()
                            clean_df = clean_df[[c for c in clean_df.columns if not c.startswith('_')]]

                            if save_results_to_blob(clean_df, az_conn_str, az_container, az_blob_name):
                                st.success(f"File has been saved to Azure Blob: {az_blob_name}")
                        else:
                            st.warning("Prediction complete, but Azure settings are missing. Results not saved.")

                    else:
                        st.warning("No results found in the response.")
                else:
                    st.error(f"Error: Server returned {response.status_code}")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")

    # --- Display Results Section ---
    if st.session_state.final_display_df is not None:
        st.markdown("---")
        st.subheader("Prediction Results")
        display_df = st.session_state.final_display_df.copy()
        cols_to_display = [col for col in display_df.columns if not col.startswith('_')]
        display_df = display_df[cols_to_display]
        display_df = display_df.loc[:, ~display_df.columns.duplicated()]

        priority_columns = ["TicketNumber", "Predicted Duration (Mins)", "Received Date", "Predicted Resolution Date",
                            "Actual Resolution Date", "SLA Status"]
        remaining_columns = [col for col in cols_to_display if
                             col not in priority_columns and col in display_df.columns]
        final_column_order = [col for col in priority_columns if col in display_df.columns] + remaining_columns
        display_df = display_df[final_column_order]

        for col in ["Predicted Duration (Mins)"]:
            if col in display_df.columns:
                display_df[col] = display_df[col].astype(str).replace('nan', '').replace('<NA>', '')

        st.dataframe(display_df, use_container_width=True, hide_index=True,
                     column_config={col: st.column_config.TextColumn(col) for col in display_df.columns})
        st.info(f"Displaying {len(display_df.columns)} columns and {len(display_df)} row(s)")
else:
    st.info("Please upload a file to begin.")

