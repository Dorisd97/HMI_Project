import os
import json
import re
import streamlit as st
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter

# --- Page setup ---
st.set_page_config(page_title="Enron Wordcloud", layout="wide")
st.title("☁️ Enron Wordcloud")

# ✅ Load dataset
data_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'cleaned_enron.json')
with open(data_path, 'r', encoding='utf-8') as f:
    emails = json.load(f)

# --- Define stopwords ---
stop_words = set([
    "the", "and", "to", "of", "a", "in", "for", "on", "that", "is", "with", "this", "by", "it", "be", "we", "are", "was",
    "as", "an", "at", "from", "not", "have", "has", "or", "but", "can", "if", "i", "you", "he", "she", "they", "will",
    "would", "could", "should", "our", "your", "their", "them", "his", "her", "us", "my", "me", "do", "does", "did",
    "been", "being", "so", "no", "yes", "just", "than", "then", "out", "said", "nbsp", "message", "original",
    "attached", "thank", "regards", "mail", "subject", "comments", "journal", "reuters", "press", "quote",
    "newsletter", "distribution", "conference", "document", "draft", "cc", "bcc", "etc", "http", "https", "com", "org"
])

# --- Clean and extract tokens ---
all_text = " ".join(email.get("Body", "").lower() for email in emails)
all_text = re.sub(r"[^a-z\s]", " ", all_text)
tokens = [word for word in all_text.split() if word not in stop_words and len(word) > 3]
word_freq = Counter(tokens)

# --- Generate Word Cloud ---
wc = WordCloud(
    width=800,
    height=400,
    background_color="white",
    colormap="copper",
    collocations=False
).generate_from_frequencies(word_freq)

# --- Display ---
fig, ax = plt.subplots(figsize=(3, 4), dpi=100)
ax.imshow(wc, interpolation="bilinear")
ax.axis("off")
st.pyplot(fig)

# --- Hide Streamlit footer ---
st.markdown("""
    <style>
    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)
