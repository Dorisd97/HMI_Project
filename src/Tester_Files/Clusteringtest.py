import streamlit as st
import json
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import LatentDirichletAllocation, PCA
from sklearn.manifold import TSNE
import umap
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

    def create_cluster_visualization_data(self, method: str, reduction_method: str = 'pca') -> Dict[str, Any]:
        """Create 2D visualization data for clusters using dimensionality reduction"""
        if method not in self.clustering_results:
            return None

        result = self.clustering_results[method]
        labels = result['labels']

        # Get the appropriate features for the method
        if method == 'text_based':
            features = self.prepare_text_features()
        elif method == 'entity_based':
            features = self.prepare_entity_features()
        elif method == 'hybrid':
            text_features = self.prepare_text_features()
            entity_features = self.prepare_entity_features()
            scaler_text = StandardScaler()
            scaler_entity = StandardScaler()
            text_features_norm = scaler_text.fit_transform(text_features)
            entity_features_norm = scaler_entity.fit_transform(entity_features)
            features = np.hstack([text_features_norm * 0.7, entity_features_norm * 0.3])
        else:  # LDA
            features = self.prepare_text_features()

        # Apply dimensionality reduction
        if reduction_method == 'pca':
            reducer = PCA(n_components=2, random_state=42)
            coords_2d = reducer.fit_transform(features)
            explained_variance = reducer.explained_variance_ratio_
        elif reduction_method == 'tsne':
            reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(features) - 1))
            coords_2d = reducer.fit_transform(features)
            explained_variance = None
        else:  # umap
            reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=min(15, len(features) - 1))
            coords_2d = reducer.fit_transform(features)
            explained_variance = None

        # Create DataFrame for plotting
        plot_data = pd.DataFrame({
            'x': coords_2d[:, 0],
            'y': coords_2d[:, 1],
            'cluster': labels,
            'email_id': [email['email_id'] for email in self.emails],
            'subject': [email['subject'][:50] + "..." if len(email['subject']) > 50
                        else email['subject'] for email in self.emails],
            'from': [email['from'] for email in self.emails],
            'to': [email['to'] for email in self.emails],
            'classification': [email.get('classification', 'Unknown') for email in self.emails]
        })

        return {
            'plot_data': plot_data,
            'explained_variance': explained_variance,
            'reduction_method': reduction_method,
            'n_clusters': result.get('n_clusters', result.get('n_topics', len(set(labels))))
        }


def create_cluster_scatter_plot(clusterer, method, reduction_method='pca'):
    """Create interactive scatter plot of clusters"""
    viz_data = clusterer.create_cluster_visualization_data(method, reduction_method)

    if viz_data is None:
        return None

    plot_data = viz_data['plot_data']

    # Create color palette for clusters
    n_clusters = viz_data['n_clusters']
    colors = px.colors.qualitative.Set3[:n_clusters] if n_clusters <= len(
        px.colors.qualitative.Set3) else px.colors.qualitative.Plotly

    # Create the scatter plot
    fig = px.scatter(
        plot_data,
        x='x',
        y='y',
        color='cluster',
        hover_data=['email_id', 'subject', 'from', 'classification'],
        title=f'{method.replace("_", " ").title()} Clusters ({reduction_method.upper()} Visualization)',
        labels={
            'x': f'{reduction_method.upper()} Component 1',
            'y': f'{reduction_method.upper()} Component 2',
            'cluster': 'Cluster'
        },
        color_discrete_sequence=colors
    )

    # Update layout for better visualization
    fig.update_traces(
        marker=dict(size=8, opacity=0.7, line=dict(width=1, color='white')),
        hovertemplate='<b>%{hovertext}</b><br>' +
                      'Cluster: %{marker.color}<br>' +
                      'Email ID: %{customdata[0]}<br>' +
                      'Subject: %{customdata[1]}<br>' +
                      'From: %{customdata[2]}<br>' +
                      'Classification: %{customdata[3]}<extra></extra>',
        hovertext=plot_data['subject']
    )

    fig.update_layout(
        width=800,
        height=600,
        title_x=0.5,
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02
        )
    )

    # Add explained variance to title if PCA
    if viz_data['explained_variance'] is not None:
        explained_var_text = f" (Explained Variance: {viz_data['explained_variance'][0]:.2%}, {viz_data['explained_variance'][1]:.2%})"
        fig.update_layout(title=fig.layout.title.text + explained_var_text)

    return fig


def create_cluster_summary_plots(clusterer, method):
    """Create summary plots for cluster analysis"""
    if method not in clusterer.clustering_results:
        return None

    result = clusterer.clustering_results[method]
    labels = result['labels']

    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Cluster Size Distribution', 'Classification by Cluster',
                        'Sender Distribution (Top 10)', 'Timeline Distribution'),
        specs=[[{"type": "bar"}, {"type": "bar"}],
               [{"type": "bar"}, {"type": "scatter"}]]
    )

    # 1. Cluster size distribution
    cluster_sizes = Counter(labels)
    fig.add_trace(
        go.Bar(
            x=list(cluster_sizes.keys()),
            y=list(cluster_sizes.values()),
            name='Cluster Sizes',
            marker_color='lightblue',
            text=list(cluster_sizes.values()),
            textposition='auto'
        ),
        row=1, col=1
    )

    # 2. Classification distribution by cluster
    cluster_classifications = defaultdict(Counter)
    for idx, label in enumerate(labels):
        classification = clusterer.emails[idx].get('classification', 'Unknown')
        cluster_classifications[label][classification] += 1

    # Get all unique classifications
    all_classifications = set()
    for cluster_data in cluster_classifications.values():
        all_classifications.update(cluster_data.keys())

    # Create stacked bar for classifications
    for i, classification in enumerate(all_classifications):
        cluster_ids = sorted(cluster_classifications.keys())
        heights = [cluster_classifications[cluster][classification] for cluster in cluster_ids]

        fig.add_trace(
            go.Bar(
                x=cluster_ids,
                y=heights,
                name=classification[:15] + "..." if len(classification) > 15 else classification,
                legendgroup="classifications"
            ),
            row=1, col=2
        )

    # 3. Top senders distribution
    senders = [email.get('from', 'Unknown') for email in clusterer.emails]
    sender_counts = Counter(senders).most_common(10)

    fig.add_trace(
        go.Bar(
            x=[sender[1] for sender in sender_counts],
            y=[sender[0].split('@')[0] if '@' in sender[0] else sender[0] for sender in sender_counts],
            orientation='h',
            name='Top Senders',
            marker_color='lightgreen'
        ),
        row=2, col=1
    )

    # 4. Email timeline
    try:
        dates = [email.get('date', '') for email in clusterer.emails]
        dates_parsed = pd.to_datetime(dates, format='%d.%m.%Y %H:%M:%S', errors='coerce')
        valid_dates = dates_parsed.dropna()

        if len(valid_dates) > 0:
            # Group by month for better visualization
            monthly_counts = valid_dates.dt.to_period('M').value_counts().sort_index()

            fig.add_trace(
                go.Scatter(
                    x=[str(period) for period in monthly_counts.index],
                    y=monthly_counts.values,
                    mode='lines+markers',
                    name='Monthly Email Count',
                    line=dict(color='purple', width=3),
                    marker=dict(size=6)
                ),
                row=2, col=2
            )
    except Exception as e:
        # If date parsing fails, show a simple message
        fig.add_annotation(
            text="Date parsing failed",
            xref="x4", yref="y4",
            x=0.5, y=0.5,
            showarrow=False,
            row=2, col=2
        )

    fig.update_layout(
        height=800,
        showlegend=True,
        title_text=f"Cluster Analysis Summary - {method.replace('_', ' ').title()}",
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02
        )
    )

    # Update axes labels
    fig.update_xaxes(title_text="Cluster ID", row=1, col=1)
    fig.update_yaxes(title_text="Number of Emails", row=1, col=1)

    fig.update_xaxes(title_text="Cluster ID", row=1, col=2)
    fig.update_yaxes(title_text="Number of Emails", row=1, col=2)

    fig.update_xaxes(title_text="Email Count", row=2, col=1)
    fig.update_yaxes(title_text="Sender", row=2, col=1)

    fig.update_xaxes(title_text="Time Period", row=2, col=2)
    fig.update_yaxes(title_text="Email Count", row=2, col=2)

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
            tab1, tab2, tab3, tab4 = st.tabs(["Cluster Plot", "Summary Charts", "Cluster Details", "Sample Emails"])

            with tab1:
                st.subheader("🎯 Interactive Cluster Visualization")

                # Dimensionality reduction method selection
                col1, col2 = st.columns([1, 3])
                with col1:
                    reduction_method = st.selectbox(
                        "Visualization method:",
                        options=['pca', 'tsne', 'umap'],
                        index=0,
                        key=f"reduction_{method}",
                        help="PCA: Linear, preserves global structure\nT-SNE: Non-linear, preserves local structure\nUMAP: Balance of local and global structure"
                    )

                # Create and display cluster scatter plot
                with st.spinner(f"Creating {reduction_method.upper()} visualization..."):
                    cluster_plot = create_cluster_scatter_plot(clusterer, method, reduction_method)
                    if cluster_plot:
                        st.plotly_chart(cluster_plot, use_container_width=True)

                        # Add explanation
                        st.info(f"""
                        **How to read this plot:**
                        - Each point represents one email
                        - Colors represent different clusters
                        - Hover over points to see email details
                        - Points close together are more similar
                        - {reduction_method.upper()} reduces high-dimensional data to 2D for visualization
                        """)
                    else:
                        st.error("Could not create cluster visualization")

            with tab2:
                st.subheader("📈 Summary Analysis")
                summary_plot = create_cluster_summary_plots(clusterer, method)
                if summary_plot:
                    st.plotly_chart(summary_plot, use_container_width=True)

            with tab3:
                st.subheader("🏷️ Cluster Characteristics")
                result = clusterer.clustering_results[method]

                if 'cluster_topics' in result:
                    st.write("**🔤 Top terms per cluster:**")
                    for cluster_id, topics in result['cluster_topics'].items():
                        with st.expander(f"Cluster {cluster_id} - Top Terms"):
                            st.write(", ".join(topics))

                elif 'cluster_entities' in result:
                    st.write("**🏢 Top entities per cluster:**")
                    for cluster_id, entities in result['cluster_entities'].items():
                        if entities:
                            with st.expander(f"Cluster {cluster_id} - Top Entities"):
                                st.write(", ".join(entities))

                elif 'topics' in result:
                    st.write("**📝 LDA Topics:**")
                    for topic_id, words in result['topics'].items():
                        with st.expander(f"Topic {topic_id}"):
                            st.write(", ".join(words))

            with tab4:
                st.subheader("📧 Sample Emails by Cluster")
                # Show sample emails for each cluster
                labels = result['labels']
                cluster_samples = defaultdict(list)

                for idx, label in enumerate(labels):
                    if len(cluster_samples[label]) < 5:  # Limit to 5 samples per cluster
                        cluster_samples[label].append({
                            'Cluster': label,
                            'Email ID': clusterer.emails[idx]['email_id'],
                            'Subject': clusterer.emails[idx]['subject'],
                            'From': clusterer.emails[idx]['from'],
                            'To': clusterer.emails[idx]['to'],
                            'Classification': clusterer.emails[idx].get('classification', 'Unknown'),
                            'Summary': clusterer.emails[idx].get('summary', '')[:200] + "..."
                        })

                # Display samples by cluster
                for cluster_id in sorted(cluster_samples.keys()):
                    with st.expander(f"Cluster {cluster_id} ({Counter(labels)[cluster_id]} emails total)"):
                        cluster_df = pd.DataFrame(cluster_samples[cluster_id])
                        st.dataframe(cluster_df, use_container_width=True, hide_index=True)

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
