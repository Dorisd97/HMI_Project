import streamlit as st
import pandas as pd
import json
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import numpy as np
from datetime import datetime
import re
from collections import Counter, defaultdict
import warnings

warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Enron Email Network Analysis",
    page_icon="📧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .cluster-header {
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        color: white;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
    }
    .story-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border: 1px solid #e9ecef;
        margin: 1rem 0;
    }
    .timeline-event {
        background-color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_sentence_transformer():
    """Load sentence transformer model with caching"""
    return SentenceTransformer('all-MiniLM-L6-v2')


class EnronEmailAnalyzer:
    def __init__(self):
        self.emails = None
        self.network_graph = None
        self.clusters = None
        self.embeddings = None
        self.sentence_model = None

    def load_data(self, uploaded_file):
        """Load email data from JSON file"""
        try:
            if uploaded_file is not None:
                data = json.load(uploaded_file)
                if isinstance(data, list):
                    self.emails = pd.DataFrame(data)
                else:
                    # If it's a single object, wrap in list
                    self.emails = pd.DataFrame([data])

                # Convert date strings to datetime
                self.emails['date'] = pd.to_datetime(self.emails['date'], format='%d.%m.%Y %H:%M:%S')
                return True
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return False
        return False

    def create_network_graph(self):
        """Create network graph from email data"""
        G = nx.Graph()

        # Add nodes and edges
        for _, email in self.emails.iterrows():
            sender = email['from']
            recipients = [email['to']] if isinstance(email['to'], str) else email['to']

            # Add sender node
            G.add_node(sender, node_type='person')

            # Add recipient nodes and edges
            for recipient in recipients:
                if pd.notna(recipient):
                    G.add_node(recipient, node_type='person')
                    if G.has_edge(sender, recipient):
                        G[sender][recipient]['weight'] += 1
                    else:
                        G.add_edge(sender, recipient, weight=1)

            # Add organization nodes
            if 'entities' in email and 'organizations' in email['entities']:
                for org in email['entities']['organizations']:
                    G.add_node(org, node_type='organization')
                    G.add_edge(sender, org, weight=0.5)

        self.network_graph = G
        return G

    def visualize_network(self):
        """Create interactive network visualization"""
        if self.network_graph is None:
            return None

        G = self.network_graph

        # Calculate layout
        pos = nx.spring_layout(G, k=1, iterations=50)

        # Prepare edge traces
        edge_x = []
        edge_y = []
        edge_weights = []

        for edge in G.edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_weights.append(edge[2].get('weight', 1))

        edge_trace = go.Scatter(x=edge_x, y=edge_y,
                                line=dict(width=0.5, color='#888'),
                                hoverinfo='none',
                                mode='lines')

        # Prepare node traces
        node_x = []
        node_y = []
        node_text = []
        node_color = []
        node_size = []
        node_hover = []

        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(node.split('@')[0] if '@' in node else node)

            # Color by node type
            node_type = G.nodes[node].get('node_type', 'unknown')
            if node_type == 'person':
                node_color.append('#1f77b4')
            elif node_type == 'organization':
                node_color.append('#ff7f0e')
            else:
                node_color.append('#2ca02c')

            # Size by degree
            degree = G.degree(node)
            node_size.append(min(degree * 3 + 8, 25))
            node_hover.append(f"{node}<br>Connections: {degree}")

        node_trace = go.Scatter(x=node_x, y=node_y,
                                mode='markers+text',
                                hoverinfo='text',
                                hovertext=node_hover,
                                text=node_text,
                                textposition="middle center",
                                textfont=dict(size=8),
                                marker=dict(size=node_size,
                                            color=node_color,
                                            line=dict(width=1, color='white')))

        # Create figure
        fig = go.Figure(data=[edge_trace, node_trace],
                        layout=go.Layout(
                            title='Email Communication Network',
                            title_font_size=16,
                            showlegend=False,
                            hovermode='closest',
                            margin=dict(b=20, l=5, r=5, t=40),
                            annotations=[dict(
                                text="🔵 People | 🟠 Organizations<br>Node size = connection count",
                                showarrow=False,
                                xref="paper", yref="paper",
                                x=0.005, y=-0.002)],
                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            height=600))

        return fig

    def prepare_text_for_embedding(self):
        """Prepare comprehensive text features for sentence embedding"""
        text_features = []

        for _, email in self.emails.iterrows():
            combined_text = ""

            # Add subject and summary (main content)
            subject = str(email.get('subject', ''))
            summary = str(email.get('summary', ''))
            combined_text += f"{subject}. {summary}. "

            # Add topics with emphasis
            if 'entities' in email and 'topics' in email['entities']:
                topics = " ".join(email['entities']['topics'])
                combined_text += f"Topics: {topics}. "

            # Add key organizations and people
            if 'entities' in email:
                if 'organizations' in email['entities']:
                    orgs = " ".join(email['entities']['organizations'])
                    combined_text += f"Organizations: {orgs}. "

                if 'people' in email['entities']:
                    people = " ".join(email['entities']['people'])
                    combined_text += f"People: {people}. "

            # Add classification context
            classification = str(email.get('classification', ''))
            tone = str(email.get('tone_analysis', ''))
            combined_text += f"Type: {classification} {tone}."

            text_features.append(combined_text.strip())

        return text_features

    def find_optimal_clusters(self, embeddings, max_k=15):
        """Find optimal number of clusters using multiple methods"""
        n_samples = len(embeddings)
        max_k = min(max_k, n_samples // 2)  # Ensure we don't have too many clusters

        if max_k < 2:
            return 2

        # Method 1: Silhouette Analysis
        silhouette_scores = []
        k_range = range(2, max_k + 1)

        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(embeddings)

            # Skip if we have singleton clusters
            if len(set(cluster_labels)) < k:
                silhouette_scores.append(-1)
                continue

            score = silhouette_score(embeddings, cluster_labels)
            silhouette_scores.append(score)

        # Find k with highest silhouette score
        if silhouette_scores and max(silhouette_scores) > 0:
            optimal_k_silhouette = k_range[np.argmax(silhouette_scores)]
        else:
            optimal_k_silhouette = 3

        # Method 2: DBSCAN for density-based clustering
        # Try different eps values
        eps_values = np.arange(0.1, 1.0, 0.1)
        best_dbscan_score = -1
        best_dbscan_labels = None

        for eps in eps_values:
            dbscan = DBSCAN(eps=eps, min_samples=2)
            labels = dbscan.fit_predict(embeddings)

            # Check if we got meaningful clusters (not all noise or single cluster)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = list(labels).count(-1)

            if n_clusters >= 2 and n_noise < len(labels) * 0.5:  # Less than 50% noise
                try:
                    score = silhouette_score(embeddings, labels)
                    if score > best_dbscan_score:
                        best_dbscan_score = score
                        best_dbscan_labels = labels
                except:
                    continue

        # Combine results - prefer silhouette if reasonable, otherwise use DBSCAN
        if best_dbscan_labels is not None and best_dbscan_score > 0.3:
            return 'dbscan', best_dbscan_labels
        else:
            return 'kmeans', optimal_k_silhouette

    def perform_smart_clustering(self):
        """Perform intelligent clustering using sentence transformers"""
        if self.emails is None:
            return None

        # Load sentence transformer model
        if self.sentence_model is None:
            self.sentence_model = load_sentence_transformer()

        # Prepare text features
        text_features = self.prepare_text_for_embedding()

        # Create embeddings
        with st.spinner("Creating semantic embeddings..."):
            self.embeddings = self.sentence_model.encode(text_features, show_progress_bar=False)

        # Find optimal clustering
        with st.spinner("Finding optimal clusters..."):
            cluster_method, cluster_result = self.find_optimal_clusters(self.embeddings)

        if cluster_method == 'dbscan':
            cluster_labels = cluster_result
            n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        else:  # kmeans
            optimal_k = cluster_result
            kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(self.embeddings)
            n_clusters = optimal_k

        # Add cluster labels to dataframe
        self.emails['cluster'] = cluster_labels

        # Calculate cluster quality metrics
        cluster_stats = self.calculate_cluster_stats(cluster_labels)

        # Store clustering results
        self.clusters = {
            'labels': cluster_labels,
            'embeddings': self.embeddings,
            'method': cluster_method,
            'n_clusters': n_clusters,
            'stats': cluster_stats
        }

        return cluster_labels, cluster_stats

    def calculate_cluster_stats(self, cluster_labels):
        """Calculate statistics for each cluster to identify the densest/most meaningful ones"""
        stats = {}
        unique_clusters = [c for c in set(cluster_labels) if c != -1]  # Exclude noise cluster

        for cluster_id in unique_clusters:
            cluster_emails = self.emails[self.emails['cluster'] == cluster_id]
            cluster_embeddings = self.embeddings[cluster_labels == cluster_id]

            # Calculate internal cohesion (average pairwise similarity)
            if len(cluster_embeddings) > 1:
                from sklearn.metrics.pairwise import cosine_similarity
                similarities = cosine_similarity(cluster_embeddings)
                # Get upper triangle (excluding diagonal)
                upper_triangle = similarities[np.triu_indices_from(similarities, k=1)]
                cohesion = np.mean(upper_triangle) if len(upper_triangle) > 0 else 0
            else:
                cohesion = 0

            # Calculate temporal span
            date_range = (cluster_emails['date'].max() - cluster_emails['date'].min()).days

            # Count unique participants
            unique_senders = cluster_emails['from'].nunique()

            # Extract dominant topics
            all_topics = []
            for _, email in cluster_emails.iterrows():
                if 'entities' in email and 'topics' in email['entities']:
                    all_topics.extend(email['entities']['topics'])

            dominant_topics = Counter(all_topics).most_common(3)

            # Calculate density score (combination of cohesion, size, and temporal concentration)
            size_score = min(len(cluster_emails) / 10, 1.0)  # Normalize by 10 emails
            temporal_score = 1.0 / (1.0 + date_range / 30)  # Prefer shorter time spans
            density_score = (cohesion * 0.4 + size_score * 0.3 + temporal_score * 0.3)

            stats[cluster_id] = {
                'size': len(cluster_emails),
                'cohesion': cohesion,
                'date_range': date_range,
                'unique_senders': unique_senders,
                'dominant_topics': dominant_topics,
                'density_score': density_score,
                'start_date': cluster_emails['date'].min(),
                'end_date': cluster_emails['date'].max()
            }

        return stats

    def get_top_clusters(self, top_n=5):
        """Get the top N densest/most meaningful clusters"""
        if self.clusters is None or 'stats' not in self.clusters:
            return []

        # Sort clusters by density score
        sorted_clusters = sorted(
            self.clusters['stats'].items(),
            key=lambda x: x[1]['density_score'],
            reverse=True
        )

        return sorted_clusters[:top_n]

    def analyze_cluster_story(self, cluster_id):
        """Analyze the story within a specific cluster with enhanced insights"""
        cluster_emails = self.emails[self.emails['cluster'] == cluster_id].copy()
        cluster_emails = cluster_emails.sort_values('date')

        # Enhanced story analysis
        story_analysis = {
            'cluster_id': cluster_id,
            'total_emails': len(cluster_emails),
            'unique_senders': cluster_emails['from'].nunique(),
            'date_range': (cluster_emails['date'].min(), cluster_emails['date'].max()),
            'duration_days': (cluster_emails['date'].max() - cluster_emails['date'].min()).days,
            'timeline': cluster_emails[['date', 'from', 'to', 'subject', 'summary']].to_dict('records'),
            'key_topics': [],
            'participants': Counter(cluster_emails['from']).most_common(10),
            'organizations': Counter(),
            'people': Counter(),
            'classifications': Counter(cluster_emails['classification']),
            'tone_analysis': Counter(cluster_emails['tone_analysis']),
            'email_frequency': self.calculate_email_frequency(cluster_emails),
            'story_summary': self.generate_story_summary(cluster_emails)
        }

        # Extract comprehensive entities
        all_topics = []
        for _, email in cluster_emails.iterrows():
            if 'entities' in email and 'topics' in email['entities']:
                all_topics.extend(email['entities']['topics'])

            if 'entities' in email:
                if 'organizations' in email['entities']:
                    story_analysis['organizations'].update(email['entities']['organizations'])
                if 'people' in email['entities']:
                    story_analysis['people'].update(email['entities']['people'])

        story_analysis['key_topics'] = Counter(all_topics).most_common(10)

        return story_analysis

    def calculate_email_frequency(self, cluster_emails):
        """Calculate email frequency over time"""
        daily_counts = cluster_emails.groupby(cluster_emails['date'].dt.date).size()
        return daily_counts.to_dict()

    def generate_story_summary(self, cluster_emails):
        """Generate a concise story summary"""
        if len(cluster_emails) == 0:
            return "No emails in this cluster."

        # Get the most common topics and organizations
        all_topics = []
        all_orgs = []

        for _, email in cluster_emails.iterrows():
            if 'entities' in email and 'topics' in email['entities']:
                all_topics.extend(email['entities']['topics'])
            if 'entities' in email and 'organizations' in email['entities']:
                all_orgs.extend(email['entities']['organizations'])

        top_topics = [item[0] for item in Counter(all_topics).most_common(3)]
        top_orgs = [item[0] for item in Counter(all_orgs).most_common(2)]

        # Create summary
        duration = (cluster_emails['date'].max() - cluster_emails['date'].min()).days
        start_date = cluster_emails['date'].min().strftime('%Y-%m-%d')
        end_date = cluster_emails['date'].max().strftime('%Y-%m-%d')

        summary = f"This cluster contains {len(cluster_emails)} emails spanning {duration} days "
        summary += f"from {start_date} to {end_date}. "

        if top_topics:
            summary += f"Main topics include: {', '.join(top_topics)}. "

        if top_orgs:
            summary += f"Key organizations involved: {', '.join(top_orgs)}. "

        summary += f"Primary participants: {', '.join([p[0] for p in Counter(cluster_emails['from']).most_common(3)])}."

        return summary


def main():
    st.markdown('<h1 class="main-header">🤖 AI-Powered Enron Email Story Discovery</h1>', unsafe_allow_html=True)

    analyzer = EnronEmailAnalyzer()

    # Sidebar for data upload and controls
    with st.sidebar:
        st.header("📁 Data Upload")
        uploaded_file = st.file_uploader("Upload JSON email data", type=['json'])

        if uploaded_file:
            if analyzer.load_data(uploaded_file):
                st.success(f"✅ Loaded {len(analyzer.emails)} emails")

                st.header("🎛️ Analysis Controls")

                st.info("🧠 Using AI to automatically find optimal clusters and extract stories")

                if st.button("🚀 Run AI Analysis", type="primary"):
                    with st.spinner("Running intelligent analysis..."):
                        analyzer.create_network_graph()
                        cluster_labels, cluster_stats = analyzer.perform_smart_clustering()
                    st.success("✨ AI Analysis complete!")
                    st.balloons()

    if analyzer.emails is not None:
        # Main content tabs
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🕸️ Network Graph", "🎯 Smart Clusters", "📚 Top Stories"])

        with tab1:
            st.header("📊 Dataset Overview")

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Total Emails", len(analyzer.emails))
                st.markdown('</div>', unsafe_allow_html=True)

            with col2:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Unique Senders", analyzer.emails['from'].nunique())
                st.markdown('</div>', unsafe_allow_html=True)

            with col3:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                date_range = analyzer.emails['date'].max() - analyzer.emails['date'].min()
                st.metric("Date Range (Days)", date_range.days)
                st.markdown('</div>', unsafe_allow_html=True)

            with col4:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Classifications", analyzer.emails['classification'].nunique())
                st.markdown('</div>', unsafe_allow_html=True)

            # Email timeline
            st.subheader("📅 Email Timeline")
            daily_counts = analyzer.emails.groupby(analyzer.emails['date'].dt.date).size()
            fig_timeline = px.line(x=daily_counts.index, y=daily_counts.values,
                                   title="Email Volume Over Time")
            fig_timeline.update_layout(xaxis_title="Date", yaxis_title="Number of Emails")
            st.plotly_chart(fig_timeline, use_container_width=True)

            # Communication patterns
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("👥 Top Email Senders")
                top_senders = analyzer.emails['from'].value_counts().head(10)
                fig_senders = px.bar(x=top_senders.values, y=top_senders.index,
                                     orientation='h', title="Most Active Communicators")
                st.plotly_chart(fig_senders, use_container_width=True)

            with col2:
                st.subheader("📋 Email Classifications")
                classifications = analyzer.emails['classification'].value_counts()
                fig_class = px.pie(values=classifications.values, names=classifications.index,
                                   title="Email Type Distribution")
                st.plotly_chart(fig_class, use_container_width=True)

        with tab2:
            st.header("🕸️ Communication Network")

            if analyzer.network_graph is not None:
                fig_network = analyzer.visualize_network()
                st.plotly_chart(fig_network, use_container_width=True)

                # Network statistics
                G = analyzer.network_graph
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Nodes", G.number_of_nodes())
                with col2:
                    st.metric("Edges", G.number_of_edges())
                with col3:
                    if G.number_of_nodes() > 0:
                        st.metric("Avg Degree", f"{2 * G.number_of_edges() / G.number_of_nodes():.2f}")
                with col4:
                    st.metric("Density", f"{nx.density(G):.3f}")

                # Key network insights
                st.subheader("🔍 Network Insights")
                if G.number_of_nodes() > 0:
                    # Most connected nodes
                    degree_centrality = nx.degree_centrality(G)
                    top_connected = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:5]

                    st.write("**Most Connected People:**")
                    for node, centrality in top_connected:
                        if '@' in node:  # It's a person
                            st.write(f"• {node}: {G.degree(node)} connections")
            else:
                st.info("Click 'Run AI Analysis' in the sidebar to generate the network graph.")

        with tab3:
            st.header("🎯 AI-Discovered Clusters")

            if analyzer.clusters is not None:
                # Cluster overview
                st.subheader("🤖 Clustering Results")

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Clusters Found", analyzer.clusters['n_clusters'])
                with col2:
                    st.metric("Method Used", analyzer.clusters['method'].upper())
                with col3:
                    noise_count = sum(1 for label in analyzer.clusters['labels'] if label == -1)
                    st.metric("Unclustered Emails", noise_count)

                # Cluster distribution
                cluster_counts = analyzer.emails[analyzer.emails['cluster'] != -1][
                    'cluster'].value_counts().sort_index()
                fig_clusters = px.bar(x=cluster_counts.index, y=cluster_counts.values,
                                      title="Email Distribution Across Clusters",
                                      labels={'x': 'Cluster ID', 'y': 'Number of Emails'})
                st.plotly_chart(fig_clusters, use_container_width=True)

                # Semantic visualization using PCA
                if analyzer.embeddings is not None:
                    st.subheader("🗺️ Semantic Cluster Map")
                    pca = PCA(n_components=2, random_state=42)
                    embeddings_2d = pca.fit_transform(analyzer.embeddings)

                    # Create DataFrame for plotting
                    plot_df = pd.DataFrame({
                        'x': embeddings_2d[:, 0],
                        'y': embeddings_2d[:, 1],
                        'cluster': analyzer.emails['cluster'].astype(str),
                        'subject': analyzer.emails['subject']
                    })

                    fig_semantic = px.scatter(plot_df, x='x', y='y', color='cluster',
                                              hover_data=['subject'],
                                              title="Semantic Similarity Map (PCA Projection)",
                                              labels={'x': f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)',
                                                      'y': f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)'})
                    st.plotly_chart(fig_semantic, use_container_width=True)

                # Cluster quality metrics
                st.subheader("📈 Cluster Quality Analysis")
                if 'stats' in analyzer.clusters:
                    stats_df = pd.DataFrame(analyzer.clusters['stats']).T
                    stats_df = stats_df.sort_values('density_score', ascending=False)

                    # Display top clusters by quality
                    st.write("**Clusters Ranked by Density Score:**")
                    for idx, (cluster_id, stats) in enumerate(stats_df.iterrows()):
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric(f"Cluster {cluster_id}", f"Score: {stats['density_score']:.3f}")
                        with col2:
                            st.metric("Size", stats['size'])
                        with col3:
                            st.metric("Cohesion", f"{stats['cohesion']:.3f}")
                        with col4:
                            st.metric("Duration", f"{stats['date_range']} days")
            else:
                st.info("Click 'Run AI Analysis' in the sidebar to discover clusters.")

        with tab4:
            st.header("📚 Automated Story Discovery")

            if analyzer.clusters is not None:
                top_clusters = analyzer.get_top_clusters(top_n=5)

                if top_clusters:
                    st.subheader("🏆 Top Stories by AI Analysis")
                    st.write("Stories are automatically ranked by density, cohesion, and significance.")

                    for rank, (cluster_id, cluster_stats) in enumerate(top_clusters, 1):
                        story = analyzer.analyze_cluster_story(cluster_id)

                        # Story card
                        st.markdown(f"""
                        <div class="story-card">
                            <h3>📖 Story #{rank}: Cluster {cluster_id}</h3>
                            <p><strong>Density Score:</strong> {cluster_stats['density_score']:.3f}</p>
                        </div>
                        """, unsafe_allow_html=True)

                        # Story metrics
                        col1, col2, col3, col4, col5 = st.columns(5)
                        with col1:
                            st.metric("📧 Emails", story['total_emails'])
                        with col2:
                            st.metric("👥 Participants", story['unique_senders'])
                        with col3:
                            st.metric("⏱️ Duration", f"{story['duration_days']} days")
                        with col4:
                            st.metric("🏢 Organizations", len(story['organizations']))
                        with col5:
                            st.metric("🎯 Topics", len(story['key_topics']))

                        # Story summary
                        st.write("**📝 Auto-Generated Summary:**")
                        st.write(story['story_summary'])

                        # Expandable details
                        with st.expander(f"🔍 Detailed Analysis - Story #{rank}"):

                            # Timeline visualization
                            st.write("**📅 Timeline of Events:**")
                            timeline_df = pd.DataFrame(story['timeline'])
                            timeline_df['date'] = pd.to_datetime(timeline_df['date'])

                            # Create timeline chart
                            fig_timeline = px.scatter(timeline_df, x='date', y='from',
                                                      hover_data=['subject'],
                                                      title=f"Timeline for Cluster {cluster_id}")
                            fig_timeline.update_traces(marker_size=10)
                            st.plotly_chart(fig_timeline, use_container_width=True)

                            # Key insights in columns
                            col1, col2, col3 = st.columns(3)

                            with col1:
                                st.write("**🎯 Top Topics:**")
                                for topic, count in story['key_topics'][:5]:
                                    st.write(f"• {topic} ({count})")

                            with col2:
                                st.write("**🏢 Organizations:**")
                                for org, count in story['organizations'].most_common(5):
                                    st.write(f"• {org} ({count})")

                            with col3:
                                st.write("**👤 Key People:**")
                                for person, count in story['people'].most_common(5):
                                    st.write(f"• {person} ({count})")

                            # Detailed timeline
                            st.write("**📋 Chronological Events:**")
                            for event in story['timeline'][:10]:  # Show first 10 events
                                st.markdown(f"""
                                <div class="timeline-event">
                                    <strong>{pd.to_datetime(event['date']).strftime('%Y-%m-%d %H:%M')}</strong><br>
                                    <strong>From:</strong> {event['from']}<br>
                                    <strong>To:</strong> {event['to']}<br>
                                    <strong>Subject:</strong> {event['subject']}<br>
                                    <strong>Summary:</strong> {event['summary'][:200]}...
                                </div>
                                """, unsafe_allow_html=True)

                        st.write("---")
                else:
                    st.write("No significant clusters found. Try with more email data.")
            else:
                st.info("Click 'Run AI Analysis' in the sidebar to discover stories automatically.")

    else:
        st.info("👆 Please upload a JSON file containing email data to begin AI-powered analysis.")

        # Show example data format and new features
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📋 Expected Data Format")
            st.code('''
{
  "to": "recipient@enron.com",
  "from": "sender@enron.com", 
  "date": "08.11.2000 04:08:00",
  "subject": "Email subject",
  "summary": "Email summary...",
  "tone_analysis": "Professional",
  "classification": "Internal Communication",
  "entities": {
    "people": ["Person 1", "Person 2"],
    "organizations": ["Org 1", "Org 2"],
    "topics": ["Topic 1", "Topic 2"]
  }
}
            ''', language='json')

        with col2:
            st.subheader("🚀 New AI Features")
            st.markdown("""
            - **🤖 Automatic Cluster Detection**: No manual cluster count needed
            - **🧠 Semantic Understanding**: Uses sentence transformers for better analysis
            - **📊 Density-Based Ranking**: Finds the most meaningful stories automatically
            - **⚡ Smart Algorithms**: Combines K-means and DBSCAN for optimal results
            - **📈 Quality Metrics**: Cohesion, temporal analysis, and significance scoring
            - **📚 Auto Story Generation**: Generates comprehensive story summaries
            """)


if __name__ == "__main__":
    main()
