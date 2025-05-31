import streamlit as st
import pandas as pd
import json
import plotly.express as px
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
import re
from src.config.config import CLEANED_JSON_PATH

# ----------------------------
# Load and normalize JSON
# ----------------------------
@st.cache_data
def load_data(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    df = pd.json_normalize(data)
    return df

df = load_data(CLEANED_JSON_PATH)

# ----------------------------
# Clean/rename relevant columns for clarity
# ----------------------------
expected_cols = ['Message-ID', 'Date', 'From', 'To', 'Subject', 'Body', 'X-From', 'X-To']
missing = [col for col in expected_cols if col not in df.columns]
if missing:
    st.warning(f"Missing expected columns: {missing}")

# Optional rename for clarity
df = df.rename(columns={
    'From': 'Sender',
    'To': 'Recipient',
    'Subject': 'Subject',
    'Body': 'Body'
})

# ----------------------------
# App Title and Basic Display
# ----------------------------
st.title("📧 Enron Email Viewer & Visualizer")
st.markdown("Explore the structured Enron emails with search, filters, and visualizations.")

st.write(f"Total emails: {len(df)}")
st.dataframe(df[['Date', 'Sender', 'Recipient', 'Subject', 'Body']])

# ----------------------------
# Top Senders Chart
# ----------------------------
st.subheader("📨 Top 10 Email Senders")
if 'Sender' in df.columns:
    top_senders = df['Sender'].value_counts().head(10).reset_index()
    top_senders.columns = ['Sender', 'Email Count']
    fig_sender = px.bar(top_senders, x='Sender', y='Email Count', title='Top 10 Email Senders')
    st.plotly_chart(fig_sender)
else:
    st.warning("Column 'Sender' not found.")

# ----------------------------
# Word Frequency from Subject
# ----------------------------
st.subheader("🗂️ Top Words in Subjects")

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'\W+', ' ', text)
    return text

if 'Subject' in df.columns:
    all_subjects = ' '.join(df['Subject'].dropna().astype(str).apply(clean_text))
    word_freq = Counter(all_subjects.split())
    word_df = pd.DataFrame(word_freq.most_common(20), columns=['Word', 'Frequency'])
    fig_words = px.bar(word_df, x='Word', y='Frequency', title='Top 20 Words in Email Subjects')
    st.plotly_chart(fig_words)
else:
    st.warning("Column 'Subject' not found.")

# ----------------------------
# Word Cloud
# ----------------------------
st.subheader("☁️ Word Cloud from Subjects")

if 'Subject' in df.columns and all_subjects.strip():
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_subjects)
    fig, ax = plt.subplots()
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
    st.pyplot(fig)
else:
    st.warning("No subject data to generate word cloud.")

# ----------------------------
# Filter by Sender
# ----------------------------
st.subheader("🔎 Filter Emails by Sender")

if 'Sender' in df.columns:
    unique_senders = df['Sender'].dropna().unique()
    selected_sender = st.selectbox("Select sender", ["All"] + sorted(unique_senders))

    if selected_sender != "All":
        filtered_df = df[df['Sender'] == selected_sender]
    else:
        filtered_df = df

    st.write(f"Showing {len(filtered_df)} emails")
    st.dataframe(filtered_df[['Date', 'Sender', 'Recipient', 'Subject', 'Body']])
