import streamlit as st
import pandas as pd
import json
import requests
import urllib3
import base64
import os

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
        # Store the original input data for later use
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

                        # Extract key prediction columns
                        prediction_data = pd.DataFrame()

                        # Ticket ID
                        if 'Ticket ID' in results_df.columns:
                            prediction_data['TicketNumber'] = results_df['Ticket ID']

                        # Predicted Duration
                        if 'Predicted Duration (Mins)' in results_df.columns:
                            prediction_data['Predicted Duration (Mins)'] = pd.to_numeric(
                                results_df['Predicted Duration (Mins)'], errors='coerce').round(0).astype('int')

                        # Received Date
                        if 'Received Date' in results_df.columns:
                            received_dt = pd.to_datetime(results_df['Received Date'], errors='coerce')
                            prediction_data['Received Date'] = received_dt.dt.strftime('%m/%d/%Y %H:%M:%S')
                            prediction_data['_received_dt'] = received_dt  # Keep for calculation

                        # Predicted Resolution Date
                        if 'Predicted Resolution Date' in results_df.columns:
                            pred_dt = pd.to_datetime(results_df['Predicted Resolution Date'], errors='coerce')
                            prediction_data['Predicted Resolution Date'] = pred_dt.dt.strftime('%m/%d/%Y %H:%M:%S')
                            prediction_data['_pred_dt'] = pred_dt  # Keep for calculation

                        # Check if Actual Resolution Date exists in the response
                        has_actual_date = False
                        if 'Actual Resolution Date' in results_df.columns:
                            actual_dt = pd.to_datetime(results_df['Actual Resolution Date'], errors='coerce')
                            if actual_dt.notna().any():
                                has_actual_date = True
                                prediction_data['Actual Resolution Date'] = actual_dt.dt.strftime('%m/%d/%Y %H:%M:%S')
                                prediction_data['_actual_dt'] = actual_dt

                        # Calculate SLA Status based on what data is available
                        if has_actual_date:
                            # Compare Actual vs Predicted Resolution Date
                            delta_seconds = (
                                        prediction_data['_actual_dt'] - prediction_data['_pred_dt']).dt.total_seconds()
                            delta_minutes = delta_seconds / 60

                            # If actual is later than predicted = Delay, if earlier = Early
                            prediction_data['SLA Status'] = delta_minutes.apply(
                                lambda x: 'Delay' if x > 0 else ('Early' if x < 0 else 'On Time')
                            )
                        else:
                            # No actual data available yet
                            prediction_data['SLA Status'] = 'Pending'

                        # Add all original input columns from the JSON payload
                        if st.session_state.original_input_data is not None:
                            original_df = st.session_state.original_input_data.copy()

                            # Format datetime columns in original data
                            for col in original_df.columns:
                                if original_df[col].dtype == 'object':
                                    try:
                                        dt_col = pd.to_datetime(original_df[col], errors='coerce')
                                        if dt_col.notna().any():
                                            original_df[col] = dt_col.dt.strftime('%m/%d/%Y %H:%M:%S')
                                    except:
                                        pass

                            # Remove columns from original_df that already exist in prediction_data
                            # This prevents duplicate columns
                            columns_to_exclude = [col for col in original_df.columns if col in prediction_data.columns]
                            original_df_filtered = original_df.drop(columns=columns_to_exclude, errors='ignore')

                            # Combine prediction data with filtered original input data
                            st.session_state.final_display_df = pd.concat([prediction_data, original_df_filtered],
                                                                          axis=1)
                        else:
                            st.session_state.final_display_df = prediction_data

                    else:
                        st.warning("No results found in the response.")
                else:
                    st.error(f"Error: Server returned {response.status_code}")
                    st.error(f"Response: {response.text}")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
            import traceback

            st.error(traceback.format_exc())

    # --- Display Results Section ---
    if st.session_state.final_display_df is not None:
        st.markdown("---")
        st.subheader("Prediction Results")

        # Create display dataframe
        display_df = st.session_state.final_display_df.copy()

        # Remove internal columns (those starting with _)
        cols_to_display = [col for col in display_df.columns if not col.startswith('_')]
        display_df = display_df[cols_to_display]

        # Remove any duplicate columns that might have slipped through
        display_df = display_df.loc[:, ~display_df.columns.duplicated()]

        # Define priority column order (these will appear first if they exist)
        priority_columns = [
            "TicketNumber",
            "Predicted Duration (Mins)",
            "Received Date",
            "Predicted Resolution Date",
            "Actual Resolution Date",
            "SLA Status"
        ]

        # Build final column order: priority columns first, then remaining columns
        remaining_columns = [col for col in cols_to_display if
                             col not in priority_columns and col in display_df.columns]
        final_column_order = [col for col in priority_columns if col in display_df.columns] + remaining_columns

        display_df = display_df[final_column_order]

        # Convert numeric columns to string for left alignment
        numeric_cols = ["Predicted Duration (Mins)"]
        for col in numeric_cols:
            if col in display_df.columns:
                display_df[col] = display_df[col].astype(str).replace('nan', '').replace('<NA>', '')

        # Create column configuration for better display
        column_config = {}
        for col in display_df.columns:
            # Use text column for everything to ensure left alignment
            column_config[col] = st.column_config.TextColumn(col)

        # Display the dataframe
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            column_config=column_config
        )

        # Show column count
        st.info(f"Displaying {len(display_df.columns)} columns and {len(display_df)} row(s)")

else:
    st.info("Please upload a file to begin.")

