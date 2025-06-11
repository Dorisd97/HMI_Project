import streamlit as st
import json
import pandas as pd
import numpy as np

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

def find_best_k(features: np.ndarray, k_min=2, k_max=10) -> int:
    scores = []
    Ks = list(range(k_min, k_max + 1))
    for k in Ks:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(features)
        scores.append(silhouette_score(features, labels))
    # Plot silhouette vs. k
    fig, ax = plt.subplots()
    ax.plot(Ks, scores, marker='o')
    ax.set_xlabel('Number of clusters k')
    ax.set_ylabel('Silhouette Score')
    ax.set_title('Selecting k via Silhouette')
    st.pyplot(fig)
    # Return the k with the highest score
    best_k = Ks[int(np.argmax(scores))]
    return best_k

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
    st.set_page_config(page_title="Subject+Summary Clustering", layout="wide")
    st.title("📧 Cluster on Subject & Summary Only")

    uploaded = st.file_uploader("Upload your JSON file", type="json")
    if not uploaded:
        st.info("Please upload a JSON file containing `subject` and `summary` fields.")
        st.stop()

    # Read uploaded file as bytes and pass to our loader
    raw_bytes = uploaded.read()
    df = load_subject_summary(raw_bytes)
    st.write(f"Loaded {len(df)} emails.")

    # Vectorize text
    with st.spinner("Vectorizing text…"):
        X = vectorize(df['text'])

    # Choose k
    k_manual = st.checkbox("Pick k manually", value=False)
    if k_manual:
        k = st.slider("Number of clusters", 2, 20, 5)
    else:
        st.write("Finding best k automatically…")
        k = find_best_k(X, k_min=2, k_max=12)
        st.success(f"→ Best k = {k}")

    # Run K-Means
    labels = run_kmeans(X, k)
    df['cluster'] = labels

    # Display cluster sizes
    st.subheader("Cluster sizes")
    size_df = (
        df['cluster']
        .value_counts()
        .sort_index()
        .rename_axis('cluster')
        .reset_index(name='count')
    )
    st.bar_chart(size_df.set_index('cluster'))

    # 2D PCA scatter
    st.subheader("2D PCA plot")
    plot_clusters_2d(X, labels)

    # Sample assignments table
    st.subheader("Sample of cluster assignments")
    st.dataframe(df[['cluster','subject','summary']].head(10), use_container_width=True)

    # Download full results
    st.download_button(
        "Download full assignments as CSV",
        df.to_csv(index=False),
        file_name="clustered_subject_summary.csv",
        mime="text/csv"
    )

if __name__ == "__main__":
    main()
