import os
import json
import re
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from wordcloud import WordCloud
import numpy as np

# --- Page setup ---
st.set_page_config(page_title="Full Emotion Tracker", layout="wide")
st.title("🧠 Full Emotion Analysis of Enron Emails")

# ✅ Load dataset
data_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'cleaned_enron.json')
with open(data_path, 'r', encoding='utf-8') as f:
    emails = json.load(f)

# --- Expanded stopwords ---
stop_words = set([
    "the", "and", "to", "of", "a", "in", "for", "on", "that", "is", "with",
    "this", "by", "it", "be", "we", "are", "was", "as", "an", "at", "from", "not",
    "have", "has", "or", "but", "can", "if", "i", "you", "he", "she", "they",
    "will", "would", "could", "should", "our", "your", "their", "them", "his",
    "her", "us", "my", "me", "do", "does", "did", "been", "being", "so", "no",
    "yes", "just", "than", "then", "out", "enron", "said", "nbsp", "message",
    "original", "attached", "thank", "regards", "mail", "subject", "comments",
    "journal", "reuters", "press", "quote", "newsletter", "distribution",
    "conference", "document", "draft", "senate", "tariff", "forwarded", "cc",
    "bcc", "etc", "http", "https", "com", "org"
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
emotion_summary = df_emotion["Emotion"].value_counts().reset_index()
emotion_summary.columns = ["Emotion", "Count"]
negative_emotions = ["anger", "fear", "sadness", "disgust"]
df_negative = df_emotion[df_emotion["Emotion"].isin(negative_emotions)]

# --- Emotion tables and charts ---
st.subheader("📋 Emotion-Tagged Words")
st.dataframe(df_emotion.head(50), use_container_width=True)

st.markdown("## 📊 Emotion Distribution")
col1, col2 = st.columns([2, 3])
with col1:
    fig, ax = plt.subplots(figsize=(6, 3))
    sns.barplot(data=emotion_summary, x="Emotion", y="Count", palette="Pastel1", ax=ax)
    ax.set_title("Emotions in Emails", fontsize=12)
    ax.set_xlabel("")
    ax.set_ylabel("Count")
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

# --- Word Cloud ---
st.markdown("---")
st.subheader("☁️ Word Cloud (Cleaned + Circular + Light Colors)")

# Prepare text
all_text = " ".join(email.get("Body", "").lower() for email in emails)
all_text = re.sub(r"[^a-z\s]", " ", all_text)
tokens = [word for word in all_text.split() if word not in stop_words and len(word) > 3]
word_freq = Counter(tokens)

# Create a circular mask
x, y = np.ogrid[:300, :300]
mask = (x - 150) ** 2 + (y - 150) ** 2 > 150**2
circle_mask = 255 * mask.astype(int)

# Create and show WordCloud
wc = WordCloud(
    width=800,
    height=400,
    background_color="white",
    colormap="copper",  # light/nude
    mask=circle_mask,
    contour_color='gray',
    contour_width=1
).generate_from_frequencies(word_freq)

fig, ax = plt.subplots(figsize=(6, 3))  # 🔽 Smaller figure size
ax.imshow(wc, interpolation="bilinear")
ax.axis("off")
st.pyplot(fig)

# --- Footer Hide ---
st.markdown("""
<style>
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)
