import streamlit as st
import pandas as pd
import json
import requests
import urllib3
import base64
import os
import io

# Suppress the "InsecureRequestWarning"
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


# =============================================================================
# HARDCODED CONFIGURATION
# =============================================================================
# (Azure Storage configurations removed)

# --- Helper Function to Encode Local Image ---
def get_base64_image(image_path):
    try:
        if os.path.exists(image_path):
            with open(image_path, "rb") as img_file:
                return base64.b64encode(img_file.read()).decode()
        return None
    except Exception:
        return None


# --- Page Config ---
st.set_page_config(page_title="Cosmic Case SLA Prediction", layout="wide")

# --- Load Logo ---
img_path = os.path.join("assets", "microsoft-logo.png")
img_base64 = get_base64_image(img_path)

# --- Sidebar ---
with st.sidebar:
    st.markdown("### Connection")
    api_url = st.text_input("Inference Endpoint",
                            value="https://func-sla-catboost-train-uat-eastus.azurewebsites.net/predict")

# --- Initialize Session State ---
for key in ["data", "built_payload", "final_display_df", "original_input_data"]:
    if key not in st.session_state: st.session_state[key] = None

# --- Main UI ---
st.markdown("# Cosmic Case SLA Prediction")

uploaded_file = st.file_uploader("Upload CSV/Excel", type=["csv", "xlsx"])
if uploaded_file:
    df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
    st.session_state.data = df
    st.dataframe(df.head(5))

# --- Prediction Logic ---
if st.session_state.data is not None:
    row_num = st.number_input("Select Row", min_value=1, max_value=len(st.session_state.data), value=1)

    if st.button("Build Payload"):
        row_data = st.session_state.data.iloc[row_num - 1].to_dict()
        st.session_state.original_input_data = pd.DataFrame([row_data])
        # Clean data for JSON
        row_json = json.loads(pd.Series(row_data).to_json(date_format='iso'))
        st.session_state.built_payload = json.dumps(row_json, indent=2)
        st.code(st.session_state.built_payload, language="json")

    if st.button("Run Inference", type="primary"):
        if not st.session_state.built_payload:
            st.error("Please build payload first.")
        else:
            try:
                with st.spinner("Calling Azure Function..."):
                    resp = requests.post(api_url, json=json.loads(st.session_state.built_payload), verify=False)

                    if resp.status_code == 200:
                        results = resp.json().get("results", [])
                        if results:
                            pred_df = pd.DataFrame(results)

                            # Merge with original features for full display
                            final_df = pd.concat([pred_df, st.session_state.original_input_data], axis=1)
                            # Remove duplicates if columns overlap
                            final_df = final_df.loc[:, ~final_df.columns.duplicated()]
                            st.session_state.final_display_df = final_df

                            st.success("Prediction complete.")
                        else:
                            st.warning("API returned success but no result data.")
                    else:
                        st.error(f"API Error: {resp.status_code}")
            except Exception as e:
                st.error(f"Error: {e}")

    if st.session_state.final_display_df is not None:
        st.subheader("Results")
        st.dataframe(st.session_state.final_display_df)

