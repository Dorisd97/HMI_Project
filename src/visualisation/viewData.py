import streamlit as st
import pandas as pd
import json
from src.config.config import CLEANED_JSON_PATH, PICKLE_FILE, CLEANED_JSON_PATH_1, BODY_CHAIN_OUTPUT_PATH

# Load JSON
with open(BODY_CHAIN_OUTPUT_PATH, 'r') as f:
    data = json.load(f)

# Normalize to DataFrame
df = pd.json_normalize(data)

# Title
st.title("Refined Enron Email Viewer")

# Display dataframe interactively
st.dataframe(df)


import pickle
# import pandas as pd
#
# df = pd.read_pickle(PICKLE_FILE)
#
# st.dataframe(df)


import streamlit as st
import json
import os

# Path to your JSON file
JSON_FILE_PATH = BODY_CHAIN_OUTPUT_PATH  # Change as needed


if not os.path.exists(JSON_FILE_PATH):
    st.error(f"File '{JSON_FILE_PATH}' not found.")
    st.stop()

# Read the JSON file
# Read the JSON file
try:
    with open(JSON_FILE_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
except Exception as e:
    st.error(f"Error loading JSON: {e}")
    st.stop()

def get_main_val(email, key):
    return email.get(key, "")

def flatten_email(email):
    parent = {
        "Main From": get_main_val(email, "From"),
        "Main To": get_main_val(email, "To"),
        "Main Subject": get_main_val(email, "Subject"),
        "Main Date": get_main_val(email, "Date"),
        "SourceFile": get_main_val(email, "SourceFile"),
    }
    rows = []
    body_chain = email.get("BodyChain", [])
    if not isinstance(body_chain, list):
        return []
    for idx, bc in enumerate(body_chain, 1):
        row = parent.copy()
        row.update({
            "Chain Index": idx,
            "Chain From": bc.get("From", ""),
            "Chain To": bc.get("To", ""),
            "Chain Cc": bc.get("Cc", ""),
            "Chain Subject": bc.get("Subject", ""),
            "Chain Body": bc.get("Body", "").replace('\n', ' '),
        })
        rows.append(row)
    return rows

rows = []
if isinstance(data, list):
    for email in data:
        rows.extend(flatten_email(email))
elif isinstance(data, dict):
    rows.extend(flatten_email(data))
else:
    st.error("Unsupported JSON structure.")
    st.stop()

if not rows:
    st.warning("No BodyChain found or empty data.")
    st.stop()

df = pd.DataFrame(rows)
st.dataframe(df, use_container_width=True, hide_index=True)