import os
import json
import re
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# Page setup
st.set_page_config(page_title="Full Emotion Tracker", layout="wide")
st.title("🧠 Full Emotion Analysis of Enron Emails")

# Load data
data_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'cleaned_enron.json')
with open(data_path, 'r', encoding='utf-8') as f:
    emails = json.load(f)

# Stopwords
stop_words = set([
    "the", "and", "to", "of", "a", "in", "for", "on", "that", "is", "with",
    "this", "by", "it", "be", "we", "are", "was", "as", "an", "at", "from", "not"
])

# Emotion mapping
emotion_keywords = {
    "joy": ["happy", "glad", "excited", "pleased", "celebrate", "smile"],
    "anger": ["angry", "furious", "hate", "fight", "rage", "resent"],
    "fear": ["scared", "fear", "afraid", "worry", "panic", "anxious"],
    "trust": ["trust", "agree", "support", "reliable", "sure", "loyal"],
    "sadness": ["sad", "unhappy", "cry", "regret", "grief", "depressed"],
    "surprise": ["surprise", "shocked", "unexpected", "amazed", "wow"],
    "disgust": ["disgust", "nasty", "gross", "horrible", "offensive"],
    "anticipation": ["expect", "hope", "eager", "plan", "soon", "await"]
}

keyword_to_emotion = {word: emotion for emotion, words in emotion_keywords.items() for word in words}
emotion_records = []

# Tag emotions in email text
for email in emails:
    body = email.get("Body", "").lower()
    clean = re.sub(r"[^a-z\s]", " ", body)
    tokens = [w for w in clean.split() if len(w) > 3 and w not in stop_words]
    for word in tokens:
        if word in keyword_to_emotion:
            emotion_records.append({
                "Word": word,
                "Emotion": keyword_to_emotion[word],
                "Date": email.get("Date", ""),
                "From": email.get("From", ""),
                "To": email.get("To", ""),
                "Subject": email.get("Subject", ""),
                "Snippet": body[:300].replace("\n", " ") + "..."
            })

df_emotion = pd.DataFrame(emotion_records)

# Emotion frequency summary
emotion_summary = df_emotion["Emotion"].value_counts().reset_index()
emotion_summary.columns = ["Emotion", "Count"]

# Negative emotion filter
negative_emotions = ["anger", "fear", "sadness", "disgust"]
df_negative = df_emotion[df_emotion["Emotion"].isin(negative_emotions)]

# --- UI layout ---

st.subheader("📋 Emotion-Tagged Words")
st.dataframe(df_emotion.head(50), use_container_width=True)

st.markdown("## 📊 Emotion Distribution")
col1, col2 = st.columns([2, 3])

with col1:
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(data=emotion_summary, x="Emotion", y="Count", palette="coolwarm", ax=ax)
    ax.set_title("Emotions in Emails", fontsize=12)
    ax.set_xlabel("")
    ax.set_ylabel("Count")
    ax.tick_params(axis='x', labelrotation=45)
    st.pyplot(fig)

with col2:
    st.markdown("#### 🧠 Emotion Insight Summary")
    st.markdown("""
    - **Anger**, **fear**, **sadness**, and **disgust** reflect emotional tension.
    - **Joy**, **trust**, and **anticipation** appear in positive or hopeful messages.
    - This helps identify **internal morale** and **organizational climate shifts** over time.
    """)

st.markdown("---")

st.subheader("🧨 Emails Containing Negative Emotions")
st.dataframe(df_negative.head(30), use_container_width=True)

# Optional footer styling
st.markdown("""
<style>
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)
