import streamlit as st
import pandas as pd
import json
from src.config.config import CLEANED_JSON_PATH_1

# Load JSON
with open(CLEANED_JSON_PATH_1, 'r') as f:
    data = json.load(f)

# Normalize to DataFrame
df = pd.json_normalize(data)

# Title
st.title("Refined Enron Email Viewer")

# Display dataframe interactively
st.dataframe(df)
