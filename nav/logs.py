import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from pytz import timezone
from database import fetch_logs, clear_logs

# Function to convert UTC to IST
def convert_utc_to_ist(utc_timestamp):
    utc = datetime.strptime(utc_timestamp, '%Y-%m-%d %H:%M:%S')
    utc = utc.replace(tzinfo=timezone('UTC'))
    ist = utc.astimezone(timezone('Asia/Kolkata'))
    return ist.strftime('%Y-%m-%d %H:%M:%S')

def show_logs():
    st.title("Prediction Logs")

    # Fetch logs from database
    logs = fetch_logs()

    if logs:
        st.subheader("Recent Predictions:")
        # Create a DataFrame for better formatting
        df_logs = pd.DataFrame(logs, columns=['model_name', 'input_data', 'result', 'timestamp'])

        # Convert timestamp from UTC to IST
        df_logs['timestamp'] = df_logs['timestamp'].apply(convert_utc_to_ist)

        # Display logs in a table
        st.table(df_logs.style.apply(lambda x: ['color: red' if x.name == 'model_name' else 'color: green' if 'Crop' in str(x) else 'color: blue' for _ in x], axis=1))

    else:
        st.write("No prediction logs found.")

    # Button to clear logs
    if st.button("Clear All Logs"):
        clear_logs()
        st.write("Logs cleared successfully.")
        # Reload the page to reflect cleared logs
        st.experimental_rerun()

# Main function to execute when script runs directly
if __name__ == "__main__":
    show_logs()
