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

# ---- LLM setup ----
@st.cache_resource
def load_llm(model_path: str, n_ctx=2048):
    from llama_cpp import Llama
    return Llama(model_path=model_path, n_ctx=n_ctx, n_threads=4)  # Set n_threads as per your CPU

def generate_summary_llm(texts, llm, cluster_id=None, max_tokens=200):
    # Concatenate up to 40 emails for summarization (to fit context)
    sample = texts[:40] if len(texts) > 40 else texts
    prompt = (
        f"Summarize the following {len(sample)} email subjects and summaries "
        "as a clear and concise paragraph that explains the main topics and purpose of this cluster:\n\n"
        + "\n".join(sample)
        + "\n\nSummary:"
    )
    out = llm(prompt, max_tokens=max_tokens, stop=["\n\n", "\nCluster"])
    return out['choices'][0]['text'].strip()

# ---- Helper functions ----

@st.cache_data
def load_subject_summary(file_bytes: bytes) -> pd.DataFrame:
    data = json.loads(file_bytes.decode('utf-8'))
    df = pd.DataFrame(data)
    df['subject'] = df.get('subject', '').fillna('')
    df['summary'] = df.get('summary', '').fillna('')
    df['text'] = df['subject'] + ' ' + df['summary']
    return df[['subject', 'summary', 'text']]

@st.cache_data
def vectorize(texts: pd.Series) -> Tuple[np.ndarray, TfidfVectorizer]:
    vec = TfidfVectorizer(
        max_features=1000,
        stop_words='english',
        ngram_range=(1,2),
        min_df=2,
        max_df=0.8
    )
    return vec.fit_transform(texts).toarray(), vec

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

def run_kmeans(features: np.ndarray, k: int) -> Tuple[np.ndarray, KMeans]:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(features)
    return labels, km

def plot_clusters_2d(features: np.ndarray, labels: np.ndarray):
    pca = PCA(n_components=2, random_state=42)
    pts = pca.fit_transform(features)
    fig, ax = plt.subplots()
    scatter = ax.scatter(pts[:,0], pts[:,1], c=labels, cmap='tab10', alpha=0.7)
    ax.set_title('PCA projection of clusters')
    legend1 = ax.legend(*scatter.legend_elements(), title="Cluster")
    ax.add_artist(legend1)
    st.pyplot(fig)

# ---- Streamlit App ----

def main():
    st.set_page_config(page_title="Subject+Summary Clustering & LLM Summaries", layout="wide")
    st.title("📧 Email Clustering & LLM Summaries (Mistral)")

    # ---- Model path config ----
    st.sidebar.header("LLM Model")
    model_path = st.sidebar.text_input(
        "Path to Mistral .gguf model file",
        value="mistral-7b-instruct-v0.2.Q4_K_M.gguf"
    )
    if not model_path:
        st.stop()
    llm = load_llm(model_path)

    # ---- Sidebar for config ----
    st.sidebar.header("Clustering")
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

    with st.spinner("Vectorizing text…"):
        X, vectorizer = vectorize(df['text'])

    if manual_k:
        k = manual_k
    else:
        st.info("Finding best k automatically…")
        k = find_best_k(X, k_min=2, k_max=12)
        st.success(f"→ Best k = **{k}**")

    labels, km_model = run_kmeans(X, k)
    df['cluster'] = labels

    st.subheader("Cluster Sizes")
    size_df = (
        df['cluster'].value_counts().sort_index()
          .rename_axis('cluster').reset_index(name='count')
    )
    st.bar_chart(size_df.set_index('cluster'))

    st.subheader("2D PCA Plot of Clusters")
    plot_clusters_2d(X, labels)

    st.subheader("Sample Cluster Assignments")
    st.dataframe(df[['cluster','subject','summary']].head(10), use_container_width=True)

    # ---- Summaries per cluster with LLM ----
    st.header("📖 LLM-generated Cluster Summaries")
    for cluster_id in sorted(df['cluster'].unique()):
        texts = df[df['cluster']==cluster_id]['text'].tolist()
        with st.spinner(f"Generating summary for Cluster {cluster_id}…"):
            summary = generate_summary_llm(texts, llm, cluster_id=cluster_id)
        st.subheader(f"Cluster {cluster_id} Summary")
        st.write(summary)

    # ---- Keywords per cluster ----
    st.header("🔑 Cluster Keywords")
    feature_names = vectorizer.get_feature_names_out()
    centers = km_model.cluster_centers_
    for i, center in enumerate(centers):
        top_idxs = center.argsort()[-5:][::-1]
        keywords = [feature_names[idx] for idx in top_idxs]
        st.write(f"Cluster {i}: {', '.join(keywords)}")

    st.download_button(
        "Download full assignments as CSV",
        df.to_csv(index=False),
        file_name="clustered_subject_summary.csv",
        mime="text/csv"
    )

if __name__ == "__main__":
    main()
