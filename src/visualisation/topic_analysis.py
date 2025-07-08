import os
import json
import re
import nltk
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from nltk.corpus import stopwords

nltk.download('stopwords', quiet=True)

stop_words = set(stopwords.words('english'))

st.set_page_config(page_title="Enron Email Emotion Analysis", layout="wide")
st.title("📬 Enron Email Word & Emotion Tagging")

data_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'cleaned_enron.json')
with open(data_path, 'r', encoding='utf-8') as f:
    emails = json.load(f)

bodies = [email.get("Body", "") for email in emails]
text = " ".join(bodies).lower()
text = re.sub(r"[^a-z\s]", " ", text)
tokens = text.split()
filtered_words = [w for w in tokens if w not in stop_words and len(w) > 3]

word_freq = Counter(filtered_words)
df_words = pd.DataFrame(word_freq.items(), columns=["Word", "Frequency"]).sort_values(by="Frequency", ascending=False)
df_words = df_words[df_words["Frequency"] > 5]

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

def get_emotion(word):
    for emotion, keywords in emotion_keywords.items():
        if word in keywords:
            return emotion
    return "neutral"

df_words["Emotion"] = df_words["Word"].apply(get_emotion)

st.subheader("🔠 Top Words with Detected Emotions")
st.dataframe(df_words.head(50), use_container_width=True)

emotion_summary = df_words["Emotion"].value_counts().reset_index()
emotion_summary.columns = ["Emotion", "Count"]
emotion_summary = emotion_summary[emotion_summary["Emotion"] != "neutral"]

st.subheader("🧠 Emotion Distribution (Keyword-based)")
fig, ax = plt.subplots(figsize=(10, 5))
sns.barplot(data=emotion_summary, x="Emotion", y="Count", palette="Set2", ax=ax)
ax.set_title("Emotion Frequencies in Enron Emails")
ax.set_xlabel("Emotion")
ax.set_ylabel("Word Count")
st.pyplot(fig)
