import os
import json
import re
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# --- Page setup ---
st.set_page_config(page_title="Full Emotion Tracker", layout="wide")
st.title("🧠 Full Emotion Analysis of Enron Emails")

# ✅ Load dataset
data_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'cleaned_enron.json')
with open(data_path, 'r', encoding='utf-8') as f:
    emails = json.load(f)

# --- Stopwords ---
stop_words = set([
    "the", "and", "to", "of", "a", "in", "for", "on", "that", "is", "with", "this", "by", "it", "be", "we", "are", "was",
    "as", "an", "at", "from", "not", "have", "has", "or", "but", "can", "if", "i", "you", "he", "she", "they", "will",
    "would", "could", "should", "our", "your", "their", "them", "his", "her", "us", "my", "me", "do", "does", "did",
    "been", "being", "so", "no", "yes", "just", "than", "then", "out", "said", "nbsp", "message", "original",
    "attached", "thank", "regards", "mail", "subject", "comments", "journal", "reuters", "press", "quote",
    "newsletter", "distribution", "conference", "document", "draft", "cc", "bcc", "etc", "http", "https", "com", "org"
])

# --- Emotion keywords ---
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

# --- Emotion tagging ---
emotion_records = []
email_emotion_counter = Counter()

for email in emails:
    body = email.get("Body", "").lower()
    clean = re.sub(r"[^a-z\s]", " ", body)
    tokens = [w for w in clean.split() if len(w) > 3 and w not in stop_words]

    # Save word-level emotion tagging for table
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

    # Count each emotion once per email
    email_emotions = set()
    for emotion, keywords in emotion_keywords.items():
        if any(keyword in tokens for keyword in keywords):
            email_emotions.add(emotion)
    email_emotion_counter.update(email_emotions)

df_emotion = pd.DataFrame(emotion_records)
df_bar = pd.DataFrame(email_emotion_counter.items(), columns=["Emotion", "Count"]).sort_values(by="Count", ascending=False)
negative_emotions = ["anger", "fear", "sadness", "disgust"]
df_negative = df_emotion[df_emotion["Emotion"].isin(negative_emotions)]

# --- UI ---
st.subheader("📋 Emotion-Tagged Words")
st.dataframe(df_emotion.head(50), use_container_width=True)

st.markdown("## 📊 Emotion Distribution")
col1, col2 = st.columns([2, 3])

with col1:
    fig, ax = plt.subplots(figsize=(6, 3))
    sns.barplot(data=df_bar, x="Emotion", y="Count", palette="pastel", ax=ax)
    ax.set_title("Emotion Distribution (One Count per Email)", fontsize=12)
    ax.set_xlabel("")
    ax.set_ylabel("Email Count")
    ax.tick_params(axis='x', labelrotation=45)
    st.pyplot(fig)

with col2:
    st.markdown("#### 🧠 Emotion Insight Summary")
    st.markdown("""
    - **Anger**, **fear**, **sadness**, and **disgust** reflect emotional tension.
    - **Joy**, **trust**, and **anticipation** highlight optimism or coordination.
    - This helps understand shifts in tone and crisis communication patterns.
    """)

st.markdown("---")
st.subheader("🧨 Emails Containing Negative Emotions")
st.dataframe(df_negative.head(30), use_container_width=True)

# --- Hide footer ---
st.markdown("""
<style>
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)
