import streamlit as st
import json
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import LatentDirichletAllocation, PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
from typing import List, Dict, Tuple, Any
import io
import base64

# Set page config
st.set_page_config(
    page_title="Email Topic Clustering Dashboard",
    page_icon="📧",
    layout="wide",
    initial_sidebar_state="expanded"
)


class EmailTopicClusterer:
    """
    A comprehensive email clustering system for Streamlit
    """

    def __init__(self, json_file_path: str):
        """Initialize the clusterer with email data"""
        self.emails = self.load_emails(json_file_path)
        self.df = pd.DataFrame(self.emails)
        self.clustering_results = {}

    def load_emails(self, file_path: str) -> List[Dict]:
        """Load emails from JSON file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
            return []

    def preprocess_text(self, text: str) -> str:
        """Clean and preprocess text for clustering"""
        if not text:
            return ""
        text = re.sub(r'\s+', ' ', text.lower().strip())
        text = re.sub(r'[^\w\s\-\.]', ' ', text)
        return text

    def prepare_text_features(self) -> np.ndarray:
        """Prepare text features using TF-IDF on summaries and subjects"""
        text_content = []
        for email in self.emails:
            combined_text = f"{email.get('subject', '')} {email.get('summary', '')}"
            text_content.append(self.preprocess_text(combined_text))

        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.8
        )

        tfidf_matrix = self.tfidf_vectorizer.fit_transform(text_content)
        return tfidf_matrix.toarray()

    def prepare_entity_features(self) -> np.ndarray:
        """Prepare features based on extracted entities"""
        all_entities = set()
        entity_categories = ['people', 'organizations', 'locations', 'projects', 'legal', 'topics']

        for email in self.emails:
            entities = email.get('entities', {})
            for category in entity_categories:
                if category in entities:
                    all_entities.update(entities[category])

        all_entities = sorted(list(all_entities))

        entity_features = []
        for email in self.emails:
            email_vector = [0] * len(all_entities)
            entities = email.get('entities', {})

            for category in entity_categories:
                if category in entities:
                    for entity in entities[category]:
                        if entity in all_entities:
                            idx = all_entities.index(entity)
                            email_vector[idx] = 1

            entity_features.append(email_vector)

        self.entity_names = all_entities
        return np.array(entity_features)

    def find_optimal_clusters(self, features: np.ndarray, max_k: int = 20) -> Tuple[int, List]:
        """Find optimal number of clusters using elbow method and silhouette score"""
        inertias = []
        silhouette_scores = []
        k_range = range(2, min(max_k + 1, len(features)))

        progress_bar = st.progress(0)
        status_text = st.empty()

        for i, k in enumerate(k_range):
            progress_bar.progress((i + 1) / len(k_range))
            status_text.text(f'Testing {k} clusters...')

            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(features)

            inertias.append(kmeans.inertia_)
            if len(set(cluster_labels)) > 1:
                silhouette_scores.append(silhouette_score(features, cluster_labels))
            else:
                silhouette_scores.append(0)

        progress_bar.empty()
        status_text.empty()

        if len(silhouette_scores) > 0:
            optimal_k = k_range[np.argmax(silhouette_scores)]
        else:
            optimal_k = 5

        return optimal_k, list(zip(k_range, inertias, silhouette_scores))

    def text_based_clustering(self, n_clusters: int = None) -> Dict[str, Any]:
        """Perform clustering based on text content"""
        text_features = self.prepare_text_features()

        if n_clusters is None:
            n_clusters, metrics = self.find_optimal_clusters(text_features)

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(text_features)

        feature_names = self.tfidf_vectorizer.get_feature_names_out()
        cluster_centers = kmeans.cluster_centers_

        cluster_topics = {}
        for i, center in enumerate(cluster_centers):
            top_indices = center.argsort()[-10:][::-1]
            top_terms = [feature_names[idx] for idx in top_indices]
            cluster_topics[i] = top_terms

        silhouette_avg = silhouette_score(text_features, cluster_labels)

        result = {
            'method': 'text_based',
            'n_clusters': n_clusters,
            'labels': cluster_labels,
            'cluster_topics': cluster_topics,
            'silhouette_score': silhouette_avg,
            'features_shape': text_features.shape
        }

        self.clustering_results['text_based'] = result
        return result

    def entity_based_clustering(self, n_clusters: int = None) -> Dict[str, Any]:
        """Perform clustering based on entity features"""
        entity_features = self.prepare_entity_features()

        if entity_features.shape[1] == 0:
            st.warning("No entity features found!")
            return None

        if n_clusters is None:
            n_clusters, metrics = self.find_optimal_clusters(entity_features, max_k=15)

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(entity_features)

        cluster_centers = kmeans.cluster_centers_
        cluster_entities = {}
        for i, center in enumerate(cluster_centers):
            top_indices = center.argsort()[-10:][::-1]
            top_entities = [self.entity_names[idx] for idx in top_indices if center[idx] > 0.1]
            cluster_entities[i] = top_entities

        silhouette_avg = silhouette_score(entity_features, cluster_labels)

        result = {
            'method': 'entity_based',
            'n_clusters': n_clusters,
            'labels': cluster_labels,
            'cluster_entities': cluster_entities,
            'silhouette_score': silhouette_avg,
            'features_shape': entity_features.shape
        }

        self.clustering_results['entity_based'] = result
        return result

    def lda_topic_modeling(self, n_topics: int = 10) -> Dict[str, Any]:
        """Perform LDA topic modeling"""
        text_features = self.prepare_text_features()

        lda = LatentDirichletAllocation(
            n_components=n_topics,
            random_state=42,
            max_iter=20,
            learning_method='online'
        )

        doc_topic_probs = lda.fit_transform(text_features)
        topic_labels = np.argmax(doc_topic_probs, axis=1)

        feature_names = self.tfidf_vectorizer.get_feature_names_out()
        topics = {}
        for topic_idx, topic in enumerate(lda.components_):
            top_indices = topic.argsort()[-10:][::-1]
            top_words = [feature_names[idx] for idx in top_indices]
            topics[topic_idx] = top_words

        result = {
            'method': 'lda',
            'n_topics': n_topics,
            'labels': topic_labels,
            'topics': topics,
            'doc_topic_probs': doc_topic_probs,
            'perplexity': lda.perplexity(text_features)
        }

        self.clustering_results['lda'] = result
        return result

    def hybrid_clustering(self, n_clusters: int = None) -> Dict[str, Any]:
        """Combine text and entity features for clustering"""
        text_features = self.prepare_text_features()
        entity_features = self.prepare_entity_features()

        scaler_text = StandardScaler()
        scaler_entity = StandardScaler()

        text_features_norm = scaler_text.fit_transform(text_features)
        entity_features_norm = scaler_entity.fit_transform(entity_features)

        combined_features = np.hstack([
            text_features_norm * 0.7,
            entity_features_norm * 0.3
        ])

        if n_clusters is None:
            n_clusters, metrics = self.find_optimal_clusters(combined_features, max_k=15)

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(combined_features)

        silhouette_avg = silhouette_score(combined_features, cluster_labels)

        result = {
            'method': 'hybrid',
            'n_clusters': n_clusters,
            'labels': cluster_labels,
            'silhouette_score': silhouette_avg,
            'features_shape': combined_features.shape
        }

        self.clustering_results['hybrid'] = result
        return result


def create_plotly_cluster_visualization(clusterer, method):
    """Create interactive Plotly visualizations"""
    if method not in clusterer.clustering_results:
        return None

    result = clusterer.clustering_results[method]
    labels = result['labels']

    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Cluster Size Distribution', 'Classification Distribution',
                        'Silhouette Score Comparison', 'Email Timeline'),
        specs=[[{"type": "bar"}, {"type": "bar"}],
               [{"type": "bar"}, {"type": "scatter"}]]
    )

    # Cluster size distribution
    cluster_sizes = Counter(labels)
    fig.add_trace(
        go.Bar(x=list(cluster_sizes.keys()), y=list(cluster_sizes.values()),
               name='Cluster Sizes', marker_color='lightblue'),
        row=1, col=1
    )

    # Classification distribution
    classifications = Counter([email.get('classification', 'Unknown') for email in clusterer.emails])
    fig.add_trace(
        go.Bar(x=list(classifications.keys()), y=list(classifications.values()),
               name='Classifications', marker_color='lightgreen'),
        row=1, col=2
    )

    # Silhouette scores comparison
    methods_with_scores = {k: v.get('silhouette_score', 0)
                           for k, v in clusterer.clustering_results.items()
                           if 'silhouette_score' in v}
    if methods_with_scores:
        fig.add_trace(
            go.Bar(x=list(methods_with_scores.keys()), y=list(methods_with_scores.values()),
                   name='Silhouette Scores', marker_color='coral'),
            row=2, col=1
        )

    # Email timeline (if dates are available)
    try:
        dates = [email.get('date', '') for email in clusterer.emails]
        dates_parsed = pd.to_datetime(dates, format='%d.%m.%Y %H:%M:%S', errors='coerce')
        valid_dates = dates_parsed.dropna()

        if len(valid_dates) > 0:
            date_counts = valid_dates.value_counts().sort_index()
            fig.add_trace(
                go.Scatter(x=date_counts.index, y=date_counts.values,
                           mode='lines+markers', name='Email Timeline',
                           line=dict(color='purple')),
                row=2, col=2
            )
    except:
        pass

    fig.update_layout(height=800, showlegend=False, title_text=f"Clustering Analysis - {method.title()}")
    return fig


def main():
    st.title("📧 Email Topic Clustering Dashboard")
    st.markdown("---")

    # Sidebar for configuration
    st.sidebar.header("Configuration")

    # File path input
    file_path = st.sidebar.text_input(
        "Enter JSON file path:",
        value="D:\Projects\HMI\HMI_Project\data\enron_full_analysis_results.json",  # Default value - change this to your file path
        help="Enter the path to your JSON file containing email data"
    )

    if not file_path:
        st.warning("Please enter a file path to continue.")
        return

    # Initialize clusterer
    try:
        with st.spinner("Loading email data..."):
            clusterer = EmailTopicClusterer(file_path)

        if not clusterer.emails:
            st.error("No emails loaded. Please check your file path and format.")
            return

        st.success(f"✅ Loaded {len(clusterer.emails)} emails successfully!")

    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return

    # Dataset overview
    st.header("📊 Dataset Overview")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Emails", len(clusterer.emails))

    with col2:
        classifications = [email.get('classification', 'Unknown') for email in clusterer.emails]
        st.metric("Unique Classifications", len(set(classifications)))

    with col3:
        senders = [email.get('from', 'Unknown') for email in clusterer.emails]
        st.metric("Unique Senders", len(set(senders)))

    with col4:
        # Count total entities
        total_entities = 0
        for email in clusterer.emails:
            entities = email.get('entities', {})
            for category in entities.values():
                total_entities += len(category)
        st.metric("Total Entities", total_entities)

    # Sample data preview
    st.subheader("📧 Sample Email Data")
    sample_df = pd.DataFrame(clusterer.emails[:5])
    st.dataframe(sample_df[['email_id', 'from', 'to', 'subject', 'classification']], use_container_width=True)

    # Clustering configuration
    st.header("🔧 Clustering Configuration")

    clustering_methods = st.multiselect(
        "Select clustering methods to run:",
        ['Text-based', 'Entity-based', 'LDA Topic Modeling', 'Hybrid'],
        default=['Text-based', 'Hybrid']
    )

    col1, col2 = st.columns(2)
    with col1:
        custom_clusters = st.checkbox("Use custom number of clusters")
    with col2:
        if custom_clusters:
            n_clusters = st.slider("Number of clusters", 2, 20, 8)
        else:
            n_clusters = None

    # Run clustering
    if st.button("🚀 Run Clustering Analysis", type="primary"):

        progress_placeholder = st.empty()
        results_placeholder = st.empty()

        with st.spinner("Running clustering analysis..."):

            # Run selected methods
            if 'Text-based' in clustering_methods:
                progress_placeholder.info("Running text-based clustering...")
                clusterer.text_based_clustering(n_clusters)

            if 'Entity-based' in clustering_methods:
                progress_placeholder.info("Running entity-based clustering...")
                clusterer.entity_based_clustering(n_clusters)

            if 'LDA Topic Modeling' in clustering_methods:
                progress_placeholder.info("Running LDA topic modeling...")
                clusterer.lda_topic_modeling(n_clusters or 10)

            if 'Hybrid' in clustering_methods:
                progress_placeholder.info("Running hybrid clustering...")
                clusterer.hybrid_clustering(n_clusters)

        progress_placeholder.success("✅ Clustering analysis completed!")

        # Display results
        st.header("📈 Clustering Results")

        # Results summary
        st.subheader("Summary")
        results_data = []
        for method, result in clusterer.clustering_results.items():
            results_data.append({
                'Method': method.replace('_', ' ').title(),
                'Clusters/Topics': result.get('n_clusters', result.get('n_topics', 'N/A')),
                'Silhouette Score': f"{result.get('silhouette_score', 0):.3f}" if 'silhouette_score' in result else 'N/A',
                'Perplexity': f"{result.get('perplexity', 0):.3f}" if 'perplexity' in result else 'N/A'
            })

        results_df = pd.DataFrame(results_data)
        st.dataframe(results_df, use_container_width=True)

        # Method-specific results
        for method in clusterer.clustering_results.keys():
            st.subheader(f"📊 {method.replace('_', ' ').title()} Results")

            # Create tabs for different views
            tab1, tab2, tab3 = st.tabs(["Visualizations", "Cluster Details", "Sample Emails"])

            with tab1:
                fig = create_plotly_cluster_visualization(clusterer, method)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)

            with tab2:
                result = clusterer.clustering_results[method]

                if 'cluster_topics' in result:
                    st.write("**Top terms per cluster:**")
                    for cluster_id, topics in result['cluster_topics'].items():
                        st.write(f"**Cluster {cluster_id}:** {', '.join(topics[:5])}")

                elif 'cluster_entities' in result:
                    st.write("**Top entities per cluster:**")
                    for cluster_id, entities in result['cluster_entities'].items():
                        if entities:
                            st.write(f"**Cluster {cluster_id}:** {', '.join(entities[:5])}")

                elif 'topics' in result:
                    st.write("**LDA Topics:**")
                    for topic_id, words in result['topics'].items():
                        st.write(f"**Topic {topic_id}:** {', '.join(words[:5])}")

            with tab3:
                # Show sample emails for each cluster
                labels = result['labels']
                cluster_samples = defaultdict(list)

                for idx, label in enumerate(labels):
                    if len(cluster_samples[label]) < 3:  # Limit to 3 samples per cluster
                        cluster_samples[label].append({
                            'Cluster': label,
                            'Subject': clusterer.emails[idx]['subject'][:80] + "...",
                            'From': clusterer.emails[idx]['from'],
                            'Classification': clusterer.emails[idx].get('classification', 'Unknown')
                        })

                # Convert to DataFrame for display
                samples_data = []
                for cluster_emails in cluster_samples.values():
                    samples_data.extend(cluster_emails)

                if samples_data:
                    samples_df = pd.DataFrame(samples_data)
                    st.dataframe(samples_df, use_container_width=True)

        # Export results
        st.header("💾 Export Results")
        if st.button("Export Clustering Results"):
            # Create export data
            export_data = {}
            for method, result in clusterer.clustering_results.items():
                export_result = result.copy()
                if 'labels' in export_result:
                    export_result['labels'] = export_result['labels'].tolist()
                if 'doc_topic_probs' in export_result:
                    export_result['doc_topic_probs'] = export_result['doc_topic_probs'].tolist()
                export_data[method] = export_result

            # Convert to JSON string
            json_string = json.dumps(export_data, indent=2)

            # Create download button
            st.download_button(
                label="Download Results as JSON",
                data=json_string,
                file_name="clustering_results.json",
                mime="application/json"
            )


if __name__ == "__main__":
    main()
