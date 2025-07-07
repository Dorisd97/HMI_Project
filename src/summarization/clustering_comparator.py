# Fix for Streamlit + PyTorch compatibility
import os

from src.config.config import PROCESSED_JSON_OUTPUT

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Workaround for the torch._classes issue
import sys
from unittest.mock import Mock

sys.modules["torch._classes"] = Mock()

import streamlit as st
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import umap
import hdbscan
from collections import Counter
import json
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# Import sentence transformers after the fix
try:
    from sentence_transformers import SentenceTransformer

    SENTENCE_TRANSFORMERS_AVAILABLE = True
except Exception as e:
    print(f"Warning: Could not import sentence_transformers: {e}")
    SENTENCE_TRANSFORMERS_AVAILABLE = False

# Page config
st.set_page_config(
    page_title="Email Clustering Analysis",
    page_icon="📧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {padding-top: 0rem;}
    .stTabs [data-baseweb="tab-list"] {gap: 24px;}
    .stTabs [data-baseweb="tab"] {height: 50px; padding-left: 20px; padding-right: 20px;}
    .metric-card {background-color: #f0f2f6; padding: 20px; border-radius: 10px; margin: 10px 0;}
</style>
""", unsafe_allow_html=True)


# Cache the model loading
@st.cache_resource
def load_sentence_transformer():
    if SENTENCE_TRANSFORMERS_AVAILABLE:
        try:
            return SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            st.error(f"Error loading sentence transformer: {e}")
            return None
    return None


class EmailTopicClusterer:
    """
    Comprehensive email clustering system for Streamlit app.
    """

    def __init__(self, emails_data):
        self.emails = emails_data
        self.df = pd.DataFrame(emails_data)
        self.embeddings = None
        self.clusters = None

    def prepare_text_features(self):
        """Prepare text features from multiple fields for clustering."""
        self.df['combined_text'] = (
                self.df['subject'].fillna('') + ' ' +
                self.df['summary'].fillna('') * 2 + ' ' +
                self.df['classification'].fillna('') + ' ' +
                self.df['entities'].apply(lambda x: ' '.join(
                    x.get('topics', []) +
                    x.get('organizations', []) +
                    x.get('people', [])
                ) if isinstance(x, dict) else '')
        )
        return self.df['combined_text']

    def method1_tfidf_kmeans(self, n_clusters=20, max_features=5000):
        """TF-IDF + K-Means Clustering"""
        texts = self.prepare_text_features()

        # TF-IDF Vectorization
        tfidf = TfidfVectorizer(
            max_features=max_features,
            min_df=5,
            max_df=0.95,
            ngram_range=(1, 2),
            stop_words='english'
        )
        tfidf_matrix = tfidf.fit_transform(texts)

        # K-Means clustering
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init=10,
            max_iter=300
        )
        clusters = kmeans.fit_predict(tfidf_matrix)

        # Get top terms per cluster
        feature_names = tfidf.get_feature_names_out()
        cluster_centers = kmeans.cluster_centers_

        cluster_topics = {}
        for i in range(n_clusters):
            top_indices = cluster_centers[i].argsort()[-10:][::-1]
            top_terms = [feature_names[idx] for idx in top_indices]
            cluster_topics[i] = top_terms

        # Calculate metrics
        silhouette = silhouette_score(tfidf_matrix, clusters)
        davies_bouldin = davies_bouldin_score(tfidf_matrix.toarray(), clusters)

        return {
            'clusters': clusters,
            'cluster_topics': cluster_topics,
            'silhouette_score': silhouette,
            'davies_bouldin_score': davies_bouldin,
            'method': 'TF-IDF + K-Means',
            'embeddings': tfidf_matrix.toarray()
        }

    def method2_lda_clustering(self, n_topics=20, n_clusters=15):
        """LDA Topic Modeling + Clustering"""
        texts = self.prepare_text_features()

        # TF-IDF for LDA
        tfidf = TfidfVectorizer(
            max_features=5000,
            min_df=5,
            max_df=0.95,
            stop_words='english'
        )
        tfidf_matrix = tfidf.fit_transform(texts)

        # LDA
        lda = LatentDirichletAllocation(
            n_components=n_topics,
            random_state=42,
            learning_method='batch',
            max_iter=50
        )
        topic_distributions = lda.fit_transform(tfidf_matrix)

        # Cluster based on topic distributions
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(topic_distributions)

        # Get topic words
        feature_names = tfidf.get_feature_names_out()
        topic_words = {}
        for topic_idx, topic in enumerate(lda.components_):
            top_indices = topic.argsort()[-10:][::-1]
            top_words = [feature_names[i] for i in top_indices]
            topic_words[topic_idx] = top_words

        # Map clusters to dominant topics
        cluster_topics = {}
        for cluster_id in range(n_clusters):
            cluster_mask = clusters == cluster_id
            if cluster_mask.any():
                cluster_topic_dist = topic_distributions[cluster_mask].mean(axis=0)
                dominant_topics = cluster_topic_dist.argsort()[-3:][::-1]
                cluster_topics[cluster_id] = [word for t in dominant_topics for word in topic_words[t][:3]]

        silhouette = silhouette_score(topic_distributions, clusters)

        return {
            'clusters': clusters,
            'topic_distributions': topic_distributions,
            'cluster_topics': cluster_topics,
            'silhouette_score': silhouette,
            'method': 'LDA + K-Means',
            'embeddings': topic_distributions
        }

    def method3_sentence_transformers(self, min_cluster_size=10):
        """Sentence Transformers + HDBSCAN"""
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            st.error("Sentence Transformers not available. Please check installation.")
            return None

        texts = self.prepare_text_features().tolist()

        # Generate embeddings
        with st.spinner('Generating semantic embeddings...'):
            model = load_sentence_transformer()
            if model is None:
                return None
            embeddings = model.encode(texts, batch_size=32, show_progress_bar=False)

        # Reduce dimensions with UMAP
        with st.spinner('Reducing dimensions...'):
            reducer = umap.UMAP(
                n_neighbors=15,
                n_components=10,
                min_dist=0.0,
                metric='cosine',
                random_state=42
            )
            reduced_embeddings = reducer.fit_transform(embeddings)

        # Cluster with HDBSCAN
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=5,
            metric='euclidean',
            cluster_selection_epsilon=0.0,
            prediction_data=True
        )
        clusters = clusterer.fit_predict(reduced_embeddings)

        # Extract representative documents per cluster
        cluster_topics = {}
        unique_clusters = set(clusters) - {-1}

        for cluster_id in unique_clusters:
            cluster_mask = clusters == cluster_id
            cluster_docs = self.df[cluster_mask]

            all_topics = []
            for _, row in cluster_docs.iterrows():
                if isinstance(row['entities'], dict):
                    all_topics.extend(row['entities'].get('topics', []))
                all_topics.extend(row['classification'].split(', '))

            topic_counts = Counter(all_topics)
            cluster_topics[cluster_id] = [t[0] for t in topic_counts.most_common(10)]

        self.embeddings = embeddings

        # Metrics
        valid_mask = clusters != -1
        if valid_mask.sum() > 1:
            silhouette = silhouette_score(reduced_embeddings[valid_mask],
                                          clusters[valid_mask])
        else:
            silhouette = -1

        return {
            'clusters': clusters,
            'embeddings': embeddings,
            'reduced_embeddings': reduced_embeddings,
            'cluster_topics': cluster_topics,
            'silhouette_score': silhouette,
            'noise_ratio': (clusters == -1).sum() / len(clusters),
            'method': 'Sentence Transformers + HDBSCAN'
        }

    def get_cluster_summary(self, result, top_n=5):
        """Get detailed summary of each cluster."""
        clusters = result['clusters']
        cluster_summaries = {}

        unique_clusters = np.unique(clusters)
        for cluster_id in unique_clusters:
            if cluster_id == -1:
                continue

            cluster_mask = clusters == cluster_id
            cluster_emails = self.df[cluster_mask]

            summary = {
                'size': len(cluster_emails),
                'percentage': len(cluster_emails) / len(self.df) * 100,
                'top_subjects': cluster_emails['subject'].value_counts().head(top_n).to_dict(),
                'top_senders': cluster_emails['from'].value_counts().head(top_n).to_dict(),
                'classifications': cluster_emails['classification'].value_counts().head(top_n).to_dict(),
                'date_range': {
                    'start': cluster_emails['date'].min(),
                    'end': cluster_emails['date'].max()
                },
                'topics': result['cluster_topics'].get(cluster_id, [])[:top_n]
            }

            cluster_summaries[f'Cluster_{cluster_id}'] = summary

        return cluster_summaries

    def create_cluster_visualization(self, result):
        """Create interactive cluster visualization using Plotly."""
        clusters = result['clusters']

        # Get embeddings
        if 'embeddings' in result:
            embeddings = result['embeddings']
        else:
            embeddings = result.get('reduced_embeddings', np.random.rand(len(clusters), 2))

        # Reduce to 2D for visualization
        if embeddings.shape[1] > 2:
            reducer = umap.UMAP(n_components=2, random_state=42)
            coords = reducer.fit_transform(embeddings)
        else:
            coords = embeddings

        # Create dataframe for plotting
        plot_df = pd.DataFrame({
            'x': coords[:, 0],
            'y': coords[:, 1],
            'cluster': clusters,
            'subject': self.df['subject'],
            'from': self.df['from'],
            'date': self.df['date']
        })

        # Create interactive scatter plot
        fig = px.scatter(
            plot_df,
            x='x',
            y='y',
            color='cluster',
            hover_data=['subject', 'from', 'date'],
            title=f'Email Clusters Visualization ({result["method"]})',
            color_continuous_scale='viridis' if clusters.min() >= 0 else None
        )

        fig.update_layout(
            height=600,
            hovermode='closest',
            plot_bgcolor='white'
        )

        return fig


def main():
    st.title("📧 Email Topic Clustering Analysis")
    st.markdown("Advanced clustering analysis for your email dataset")

    # Sidebar
    with st.sidebar:
        st.header("Configuration")

        # File path input
        json_file_path = st.text_input(
            "JSON File Path",
            value="PROCESSED_JSON_OUTPUT",
            help="Enter the path to your processed JSON file"
        )

        # Load data button
        if st.button("Load Email Data", type="primary"):
            st.session_state.load_data = True

    # Main content
    if 'load_data' in st.session_state and st.session_state.load_data:
        try:
            # Load data
            with st.spinner(f"Loading data from {PROCESSED_JSON_OUTPUT}..."):
                with open(PROCESSED_JSON_OUTPUT, 'r', encoding='utf-8') as f:
                    data = json.load(f)

            st.success(f"Successfully loaded {len(data)} emails!")

            # Initialize clusterer
            clusterer = EmailTopicClusterer(data)

            # Display basic statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Emails", len(data))
            with col2:
                st.metric("Unique Senders", clusterer.df['from'].nunique())
            with col3:
                st.metric("Date Range", f"{clusterer.df['date'].min()[:10]} to {clusterer.df['date'].max()[:10]}")
            with col4:
                st.metric("Unique Classifications", len(set(', '.join(clusterer.df['classification']).split(', '))))

            # Tabs for different clustering methods
            if SENTENCE_TRANSFORMERS_AVAILABLE:
                tab1, tab2, tab3, tab4 = st.tabs(
                    ["TF-IDF + K-Means", "LDA Topic Modeling", "Sentence Transformers", "Compare Methods"])
            else:
                tab1, tab2, tab4 = st.tabs(["TF-IDF + K-Means", "LDA Topic Modeling", "Compare Methods"])
                st.warning("Sentence Transformers not available. Install with: pip install sentence-transformers")

            with tab1:
                st.header("TF-IDF + K-Means Clustering")
                st.markdown("Fast and interpretable clustering based on term frequency")

                col1, col2 = st.columns(2)
                with col1:
                    n_clusters_tfidf = st.slider("Number of Clusters", 5, 50, 20, key="tfidf_clusters")
                with col2:
                    max_features = st.slider("Max Features", 1000, 10000, 5000, step=1000)

                if st.button("Run TF-IDF Clustering", key="run_tfidf"):
                    with st.spinner("Running TF-IDF + K-Means..."):
                        result_tfidf = clusterer.method1_tfidf_kmeans(n_clusters_tfidf, max_features)
                        st.session_state.result_tfidf = result_tfidf

                if 'result_tfidf' in st.session_state:
                    result = st.session_state.result_tfidf

                    # Display metrics
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Silhouette Score", f"{result['silhouette_score']:.3f}")
                    with col2:
                        st.metric("Davies-Bouldin Score", f"{result['davies_bouldin_score']:.3f}")

                    # Visualization
                    st.plotly_chart(clusterer.create_cluster_visualization(result), use_container_width=True)

                    # Cluster topics
                    st.subheader("Cluster Topics")
                    for cluster_id, topics in result['cluster_topics'].items():
                        st.write(f"**Cluster {cluster_id}**: {', '.join(topics[:5])}")

            with tab2:
                st.header("LDA Topic Modeling + Clustering")
                st.markdown("Discover latent topics and cluster based on topic distributions")

                col1, col2 = st.columns(2)
                with col1:
                    n_topics = st.slider("Number of Topics", 10, 50, 25, key="lda_topics")
                with col2:
                    n_clusters_lda = st.slider("Number of Clusters", 5, 30, 15, key="lda_clusters")

                if st.button("Run LDA Clustering", key="run_lda"):
                    with st.spinner("Running LDA Topic Modeling..."):
                        result_lda = clusterer.method2_lda_clustering(n_topics, n_clusters_lda)
                        st.session_state.result_lda = result_lda

                if 'result_lda' in st.session_state:
                    result = st.session_state.result_lda

                    st.metric("Silhouette Score", f"{result['silhouette_score']:.3f}")

                    # Visualization
                    st.plotly_chart(clusterer.create_cluster_visualization(result), use_container_width=True)

                    # Topic distribution heatmap
                    st.subheader("Topic Distribution Heatmap")
                    topic_dist = result['topic_distributions'][:100]  # Show first 100 emails
                    fig = px.imshow(topic_dist.T,
                                    labels=dict(x="Email Index", y="Topic", color="Probability"),
                                    title="Topic Distribution across Emails (First 100)")
                    st.plotly_chart(fig, use_container_width=True)

            if SENTENCE_TRANSFORMERS_AVAILABLE:
                with tab3:
                    st.header("Sentence Transformers + HDBSCAN")
                    st.markdown("State-of-the-art semantic clustering with automatic cluster detection")

                    min_cluster_size = st.slider("Minimum Cluster Size", 10, 200, 50,
                                                 help="Smaller values create more clusters")

                    if st.button("Run Semantic Clustering", key="run_transformers"):
                        result_trans = clusterer.method3_sentence_transformers(min_cluster_size)
                        if result_trans:
                            st.session_state.result_trans = result_trans

                    if 'result_trans' in st.session_state:
                        result = st.session_state.result_trans

                        # Display metrics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Silhouette Score", f"{result['silhouette_score']:.3f}")
                        with col2:
                            st.metric("Number of Clusters",
                                      len(set(result['clusters'])) - (1 if -1 in result['clusters'] else 0))
                        with col3:
                            st.metric("Noise Ratio", f"{result['noise_ratio']:.2%}")

                        # Visualization
                        st.plotly_chart(clusterer.create_cluster_visualization(result), use_container_width=True)

                        # Cluster summaries
                        st.subheader("Cluster Analysis")
                        summaries = clusterer.get_cluster_summary(result)

                        for cluster_name, summary in summaries.items():
                            with st.expander(
                                    f"{cluster_name} ({summary['size']} emails - {summary['percentage']:.1f}%)"):
                                col1, col2 = st.columns(2)

                                with col1:
                                    st.write("**Top Topics:**")
                                    for topic in summary['topics']:
                                        st.write(f"• {topic}")

                                    st.write("**Date Range:**")
                                    st.write(f"{summary['date_range']['start']} to {summary['date_range']['end']}")

                                with col2:
                                    st.write("**Top Senders:**")
                                    for sender, count in list(summary['top_senders'].items())[:3]:
                                        st.write(f"• {sender}: {count}")

                                    st.write("**Classifications:**")
                                    for cls, count in list(summary['classifications'].items())[:3]:
                                        st.write(f"• {cls}: {count}")

            with tab4:
                st.header("Method Comparison")
                st.markdown("Compare the performance of different clustering methods")

                if st.button("Run All Methods", key="run_all"):
                    with st.spinner("Running all clustering methods... This may take a few minutes."):
                        # Run all methods with default parameters
                        results = {}

                        # TF-IDF
                        results['TF-IDF + K-Means'] = clusterer.method1_tfidf_kmeans(n_clusters=20)

                        # LDA
                        results['LDA + Clustering'] = clusterer.method2_lda_clustering(n_topics=25, n_clusters=20)

                        # Sentence Transformers (if available)
                        if SENTENCE_TRANSFORMERS_AVAILABLE:
                            trans_result = clusterer.method3_sentence_transformers(min_cluster_size=50)
                            if trans_result:
                                results['Sentence Transformers'] = trans_result

                        st.session_state.all_results = results

                if 'all_results' in st.session_state:
                    results = st.session_state.all_results

                    # Comparison table
                    comparison_data = []
                    for method, result in results.items():
                        row = {
                            'Method': method,
                            'Silhouette Score': f"{result['silhouette_score']:.3f}",
                            'Number of Clusters': len(set(result['clusters'])) - (1 if -1 in result['clusters'] else 0)
                        }
                        if 'davies_bouldin_score' in result:
                            row['Davies-Bouldin Score'] = f"{result['davies_bouldin_score']:.3f}"
                        if 'noise_ratio' in result:
                            row['Noise Ratio'] = f"{result['noise_ratio']:.2%}"
                        comparison_data.append(row)

                    comparison_df = pd.DataFrame(comparison_data)
                    st.dataframe(comparison_df, use_container_width=True)

                    # Best method
                    best_method = max(results.items(), key=lambda x: x[1]['silhouette_score'])
                    st.success(
                        f"Best performing method: **{best_method[0]}** (Silhouette Score: {best_method[1]['silhouette_score']:.3f})")

                    # Export results
                    st.subheader("Export Results")
                    selected_method = st.selectbox("Select method to export", list(results.keys()))

                    if st.button("Prepare Export"):
                        selected_result = results[selected_method]
                        clusterer.df['cluster'] = selected_result['clusters']

                        # Create downloadable CSV
                        csv = clusterer.df.to_csv(index=False)
                        st.download_button(
                            label="Download Clustered Emails CSV",
                            data=csv,
                            file_name="clustered_emails.csv",
                            mime="text/csv"
                        )

                        # Create cluster summary JSON
                        summaries = clusterer.get_cluster_summary(selected_result)
                        summary_json = json.dumps(summaries, indent=2)
                        st.download_button(
                            label="Download Cluster Summaries JSON",
                            data=summary_json,
                            file_name="cluster_summaries.json",
                            mime="application/json"
                        )

        except FileNotFoundError:
            st.error(f"File not found: {json_file_path}")
            st.info("Please make sure the file path is correct and the file exists.")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.info("Please check your data format and try again.")

    else:
        # Landing page
        st.info("👈 Please enter your JSON file path in the sidebar and click 'Load Email Data' to begin.")

        st.markdown("""
        ### Features:
        - **Multiple Clustering Methods**: TF-IDF, LDA, Sentence Transformers
        - **Interactive Visualizations**: Explore clusters in 2D space
        - **Detailed Analytics**: Cluster summaries, topic extraction, quality metrics
        - **Export Results**: Download clustered data and summaries

        ### Required JSON Format:
        ```json
        {
            "to": "",
            "from": "email@example.com",
            "date": "30.11.2001 15:30:37",
            "subject": "Email Subject",
            "summary": "Email summary text...",
            "tone_analysis": "Professional",
            "classification": "Business, Legal",
            "entities": {
                "people": [],
                "organizations": ["Company"],
                "topics": ["topic1", "topic2"]
            },
            "email_id": 1
        }
        ```
        """)


if __name__ == "__main__":
    main()