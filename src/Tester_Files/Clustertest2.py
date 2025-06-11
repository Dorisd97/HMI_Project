import streamlit as st
import json
import pandas as pd
import numpy as np
import re
from typing import Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt

# --- Helper functions ---

@st.cache_data
def load_subject_summary(file_bytes: bytes) -> pd.DataFrame:
    """
    Load only subject + summary from a JSON payload (as bytes).
    """
    data = json.loads(file_bytes.decode('utf-8'))
    df = pd.DataFrame(data)
    df['subject'] = df.get('subject', '').fillna('')
    df['summary'] = df.get('summary', '').fillna('')
    df['text'] = df['subject'] + ' ' + df['summary']
    return df[['subject', 'summary', 'text']]

@st.cache_data
def vectorize(texts: pd.Series) -> Tuple[np.ndarray, TfidfVectorizer]:
    """
    Vectorize text using TF-IDF.
    """
    vec = TfidfVectorizer(
        max_features=1000,
        stop_words='english',
        ngram_range=(1,2),
        min_df=2,
        max_df=0.8
    )
    return vec.fit_transform(texts).toarray(), vec


def generate_summary(text: str, sentence_count: int) -> str:
    """
    Naively split by sentence-ending punctuation and return the first N sentences.
    """
    # normalize whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    # split into sentences
    sents = re.split(r'(?<=[\.!?])\s+', text)
    # filter out very short sentences
    sents = [s for s in sents if len(s.split()) >= 3]
    if not sents:
        return "Not enough content to generate a summary."
    return " ".join(sents[:sentence_count])


def find_best_k(features: np.ndarray, k_min=2, k_max=10) -> int:
    """
    Determine optimal k by silhouette score, plotting scores for inspection.
    """
    scores = []
    Ks = list(range(k_min, k_max + 1))
    for k in Ks:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(features)
        scores.append(silhouette_score(features, labels))
    fig, ax = plt.subplots()
    ax.plot(Ks, scores, marker='o')
    ax.set_xlabel('Number of clusters k')
    ax.set_ylabel('Silhouette Score')
    ax.set_title('Selecting k via Silhouette')
    st.pyplot(fig)
    return Ks[int(np.argmax(scores))]


def run_kmeans(features: np.ndarray, k: int) -> Tuple[np.ndarray, KMeans]:
    """
    Fit K-Means and return labels and model.
    """
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(features)
    return labels, km


def plot_clusters_2d(features: np.ndarray, labels: np.ndarray):
    """Plot a 2D PCA projection of the clusters."""
    pca = PCA(n_components=2, random_state=42)
    pts = pca.fit_transform(features)
    fig, ax = plt.subplots()
    scatter = ax.scatter(pts[:,0], pts[:,1], c=labels, cmap='tab10', alpha=0.7)
    ax.set_title('PCA projection of clusters')
    legend1 = ax.legend(*scatter.legend_elements(), title="Cluster")
    ax.add_artist(legend1)
    st.pyplot(fig)


# --- Streamlit app ---

def main():
    st.set_page_config(page_title="Subject+Summary Clustering & Summaries", layout="wide")
    st.title("📧 Cluster on Subject & Summary + Cluster Summaries")

    # Sidebar
    st.sidebar.header("Configuration")
    summary_sentences = st.sidebar.slider(
        "Sentences per cluster summary", 1, 10, 3,
        help="How many sentences to include in each cluster summary"
    )
    k_manual = st.sidebar.checkbox("Pick k manually", value=False)
    if k_manual:
        manual_k = st.sidebar.slider("Number of clusters", 2, 20, 5)
    else:
        manual_k = None

    uploaded = st.file_uploader("Upload your JSON file", type="json")
    if not uploaded:
        st.info("Please upload a JSON file containing 'subject' and 'summary' fields.")
        st.stop()

    raw_bytes = uploaded.read()
    df = load_subject_summary(raw_bytes)
    st.write(f"Loaded **{len(df)}** emails.")

    # Vectorize
    with st.spinner("Vectorizing text…"):
        X, vectorizer = vectorize(df['text'])

    # Choose k
    if manual_k:
        k = manual_k
    else:
        st.info("Finding best k automatically…")
        k = find_best_k(X, k_min=2, k_max=12)
        st.success(f"→ Best k = **{k}**")

    # Cluster
    labels, km_model = run_kmeans(X, k)
    df['cluster'] = labels

    # Cluster sizes
    st.subheader("Cluster Sizes")
    size_df = (
        df['cluster'].value_counts().sort_index()
          .rename_axis('cluster').reset_index(name='count')
    )
    st.bar_chart(size_df.set_index('cluster'))

    # 2D PCA
    st.subheader("2D PCA Plot of Clusters")
    plot_clusters_2d(X, labels)

    # Sample assignments
    st.subheader("Sample Cluster Assignments")
    st.dataframe(df[['cluster','subject','summary']].head(10), use_container_width=True)

    # Summaries per cluster
    st.header("📖 Cluster Summaries")
    for cluster_id in sorted(df['cluster'].unique()):
        texts = df[df['cluster']==cluster_id]['text'].tolist()
        combined = ' '.join(texts)
        summary = generate_summary(combined, sentence_count=summary_sentences)
        st.subheader(f"Cluster {cluster_id} Summary")
        st.write(summary)

    # Keywords per cluster for quick topic sense
    st.header("🔑 Cluster Keywords")
    feature_names = vectorizer.get_feature_names_out()
    centers = km_model.cluster_centers_
    for i, center in enumerate(centers):
        top_idxs = center.argsort()[-5:][::-1]
        keywords = [feature_names[idx] for idx in top_idxs]
        st.write(f"Cluster {i}: {', '.join(keywords)}")

    # Download full assignments
    st.download_button(
        "Download full assignments as CSV",
        df.to_csv(index=False),
        file_name="clustered_subject_summary.csv",
        mime="text/csv"
    )

if __name__ == "__main__":
    main()
