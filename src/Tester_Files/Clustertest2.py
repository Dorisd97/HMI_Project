import streamlit as st
import json
import pandas as pd
import numpy as np
import re

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
    We cache on the raw bytes, which are hashable.
    """
    data = json.loads(file_bytes.decode('utf-8'))
    df = pd.DataFrame(data)
    df['subject'] = df.get('subject', '').fillna('')
    df['summary'] = df.get('summary', '').fillna('')
    df['text'] = df['subject'] + ' ' + df['summary']
    return df[['subject', 'summary', 'text']]

@st.cache_data
def vectorize(texts: pd.Series) -> np.ndarray:
    vec = TfidfVectorizer(
        max_features=1000,
        stop_words='english',
        ngram_range=(1,2),
        min_df=2,
        max_df=0.8
    )
    return vec.fit_transform(texts).toarray()

@st.cache_data
def generate_summary(text: str, sentence_count: int) -> str:
    """
    Try extractive summary via Sumy LexRank; fallback to naive first-N sentences.
    """
    try:
        from sumy.parsers.plaintext import PlaintextParser
        from sumy.nlp.tokenizers import Tokenizer
        from sumy.summarizers.lex_rank import LexRankSummarizer
        parser = PlaintextParser.from_string(text, Tokenizer("english"))
        summarizer = LexRankSummarizer()
        sentences = summarizer(parser.document, sentence_count)
        if sentences:
            return " ".join(str(s) for s in sentences)
    except Exception:
        pass
    # Fallback: simple split on sentence boundaries
    sents = re.split(r'(?<=[.!?])\s+', text.strip())
    if not sents:
        return "Not enough content to generate a summary."
    return " ".join(sents[:sentence_count])


def find_best_k(features: np.ndarray, k_min=2, k_max=10) -> int:
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


def run_kmeans(features: np.ndarray, k: int) -> np.ndarray:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    return km.fit_predict(features)


def plot_clusters_2d(features: np.ndarray, labels: np.ndarray):
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
    st.set_page_config(page_title="Subject+Summary Clustering & Storytelling", layout="wide")
    st.title("📧 Cluster on Subject & Summary + Generate Cluster Summaries")

    # Sidebar controls
    st.sidebar.header("Configuration")
    summary_sentences = st.sidebar.slider(
        "Sentences per cluster summary (fallback if extractive fails)",
        min_value=1,
        max_value=10,
        value=3,
        help="Number of sentences in each cluster's summary"
    )
    k_manual = st.sidebar.checkbox("Pick k manually", value=False)
    if k_manual:
        manual_k = st.sidebar.slider("Number of clusters", 2, 20, 5)
    else:
        manual_k = None

    uploaded = st.file_uploader("Upload your JSON file", type="json")
    if not uploaded:
        st.info("Please upload a JSON file containing `subject` and `summary` fields.")
        st.stop()

    raw_bytes = uploaded.read()
    df = load_subject_summary(raw_bytes)
    st.write(f"Loaded **{len(df)}** emails.")

    with st.spinner("Vectorizing text…"):
        X = vectorize(df['text'])

    if manual_k:
        k = manual_k
    else:
        st.info("Finding best k automatically…")
        k = find_best_k(X, k_min=2, k_max=12)
        st.success(f"→ Best k = **{k}**")

    labels = run_kmeans(X, k)
    df['cluster'] = labels

    # Display cluster sizes
    st.subheader("Cluster Sizes")
    size_df = (
        df['cluster']
        .value_counts()
        .sort_index()
        .rename_axis('cluster')
        .reset_index(name='count')
    )
    st.bar_chart(size_df.set_index('cluster'))

    # 2D PCA scatter
    st.subheader("2D PCA Plot of Clusters")
    plot_clusters_2d(X, labels)

    # Sample assignments
    st.subheader("Sample Cluster Assignments")
    st.dataframe(df[['cluster', 'subject', 'summary']].head(10), use_container_width=True)

    # Generate and display summaries for each cluster
    st.header("📖 Cluster Summaries")
    for cluster_id in sorted(df['cluster'].unique()):
        texts = df[df['cluster'] == cluster_id]['text'].tolist()
        combined = " ".join(texts)
        summary = generate_summary(combined, sentence_count=summary_sentences)
        st.subheader(f"Cluster {cluster_id} Summary")
        st.write(summary)

    # Download results
    st.download_button(
        "Download full assignments as CSV",
        df.to_csv(index=False),
        file_name="clustered_subject_summary.csv",
        mime="text/csv"
    )

if __name__ == "__main__":
    main()
