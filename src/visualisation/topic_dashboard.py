import streamlit as st
import json
import re
import os
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set Streamlit page config
st.set_page_config(page_title="Email Topic Analysis", layout="centered")

# Title
st.title("📊 Enron Email Topic Analysis Dashboard")

# Load and preprocess data
@st.cache_data
def load_emails():
    file_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'cleaned_enron.json')
    with open(file_path, "r", encoding="utf-8") as f:
        emails = json.load(f)
    return emails

emails = load_emails()

# Define topics to track
topics = ["dynegy", "merger", "california", "crisis", "lawsuit",
          "energy", "enron", "deal", "ferc", "ab1890"]

# Flatten and clean all bodies
bodies = [email.get("Body", "") for email in emails]
text = " ".join(bodies).lower()
text_clean = re.sub(r'[^a-z\s]', '', text)

# --- Bar Chart ---
st.header("📌 Total Topic Mentions (Bar Chart)")

topic_counts = Counter({topic: text_clean.count(topic) for topic in topics})
df_bar = pd.DataFrame(topic_counts.items(), columns=["Topic", "Frequency"]).sort_values(by="Frequency", ascending=False)

# Plot bar chart
fig1, ax1 = plt.subplots(figsize=(10, 6))
sns.barplot(data=df_bar, x="Frequency", y="Topic", ax=ax1)
ax1.set_title("Most Mentioned Topics in Emails")
st.pyplot(fig1)

# --- Heatmap: Mentions over Time ---
st.header("⏳ Topic Mentions Over Time (Heatmap)")

# Extract topic counts per email by date
records = []
for email in emails:
    row = {}
    row["Date"] = email.get("Date", "")
    body = email.get("Body", "").lower()
    clean_body = re.sub(r'[^a-z\s]', '', body)
    for topic in topics:
        row[topic] = clean_body.count(topic)
    records.append(row)

df_topics = pd.DataFrame(records)

# Fix date format and drop invalid
df_topics["Date"] = pd.to_datetime(df_topics["Date"], errors="coerce", dayfirst=True)
df_topics.dropna(subset=["Date"], inplace=True)

# ✅ Filter to active email period for clarity
df_topics = df_topics[(df_topics["Date"] >= "2000-01-01") & (df_topics["Date"] <= "2002-12-31")]

# Group by month
df_monthly = df_topics.groupby(df_topics["Date"].dt.to_period("M"))[topics].sum()
df_monthly.index = df_monthly.index.astype(str)

# Plot heatmap
fig2, ax2 = plt.subplots(figsize=(12, 6))
sns.heatmap(df_monthly.T, cmap="YlGnBu", annot=True, fmt="d", ax=ax2)
ax2.set_title("Topic Mentions Over Time")
st.pyplot(fig2)
