import streamlit as st
import pandas as pd
import numpy as np
import json
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import community
from pyvis.network import Network
import streamlit.components.v1 as components
from datetime import datetime, timedelta
import re
import warnings

warnings.filterwarnings('ignore')


# Download required NLTK data
@st.cache_resource
def download_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')


download_nltk_data()


class EmailAnalyzer:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.emails_df = None
        self.similarity_matrix = None
        self.graph = None
        self.embeddings = None

    def clean_text(self, text):
        """Clean and preprocess text data"""
        if not isinstance(text, str):
            return ""

        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text.lower())

        # Tokenize
        tokens = word_tokenize(text)

        # Remove stopwords and lemmatize
        cleaned_tokens = [
            self.lemmatizer.lemmatize(token)
            for token in tokens
            if token not in self.stop_words and len(token) > 2
        ]

        return ' '.join(cleaned_tokens)

    @st.cache_data
    def load_and_preprocess_data(_self, data):
        """Load and preprocess email data"""
        emails = data['emails']

        # Convert to DataFrame
        df = pd.DataFrame(emails)

        # Parse dates
        df['date'] = pd.to_datetime(df['date'], format='%d.%m.%Y %H:%M:%S', errors='coerce')

        # Clean text fields
        df['cleaned_summary'] = df['summary'].apply(_self.clean_text)
        df['cleaned_subject'] = df['subject'].apply(_self.clean_text)

        # Combine text for analysis
        df['combined_text'] = df['cleaned_subject'] + ' ' + df['cleaned_summary']

        # Extract additional features
        df['num_recipients'] = df['to'].apply(lambda x: len(x.split(',')) if isinstance(x, str) else 0)
        df['hour'] = df['date'].dt.hour
        df['day_of_week'] = df['date'].dt.dayofweek
        df['month'] = df['date'].dt.month

        _self.emails_df = df
        return df

    @st.cache_data
    def generate_embeddings(_self, df):
        """Generate TF-IDF embeddings for emails"""
        vectorizer = TfidfVectorizer(
            max_features=1000,
            min_df=2,
            max_df=0.8,
            ngram_range=(1, 2)
        )

        embeddings = vectorizer.fit_transform(df['combined_text'].fillna(''))
        _self.embeddings = embeddings.toarray()

        return _self.embeddings, vectorizer

    @st.cache_data
    def calculate_similarity_matrix(_self, embeddings):
        """Calculate cosine similarity matrix"""
        similarity_matrix = cosine_similarity(embeddings)
        _self.similarity_matrix = similarity_matrix
        return similarity_matrix

    def build_graph(self, similarity_matrix, threshold=0.1):
        """Build network graph from similarity matrix"""
        G = nx.Graph()

        # Add nodes
        for i, row in self.emails_df.iterrows():
            G.add_node(
                i,
                title=row['subject'][:50] + '...' if len(row['subject']) > 50 else row['subject'],
                sender=row['from'],
                date=row['date'].strftime('%Y-%m-%d %H:%M') if pd.notna(row['date']) else 'Unknown',
                classification=row['classification'],
                tone=row['tone_analysis'],
                size=10 + row['num_recipients'] * 2
            )

        # Add edges based on similarity
        n = len(similarity_matrix)
        for i in range(n):
            for j in range(i + 1, n):
                if similarity_matrix[i][j] > threshold:
                    G.add_edge(i, j, weight=similarity_matrix[i][j])

        self.graph = G
        return G

    def detect_communities(self, graph):
        """Detect communities using Louvain algorithm"""
        partition = community.best_partition(graph, resolution=1.0)

        # Add community info to nodes
        for node in graph.nodes():
            graph.nodes[node]['community'] = partition[node]

        return partition

    def calculate_centrality_metrics(self, graph):
        """Calculate various centrality metrics"""
        metrics = {}

        if len(graph.nodes()) > 0:
            metrics['degree_centrality'] = nx.degree_centrality(graph)
            metrics['betweenness_centrality'] = nx.betweenness_centrality(graph)
            metrics['closeness_centrality'] = nx.closeness_centrality(graph)
            metrics['eigenvector_centrality'] = nx.eigenvector_centrality(graph, max_iter=1000)

        return metrics


def create_network_visualization(graph, communities):
    """Create interactive network visualization using Plotly"""
    if len(graph.nodes()) == 0:
        return go.Figure()

    # Use spring layout for positioning
    pos = nx.spring_layout(graph, k=1, iterations=50)

    # Prepare node traces by community
    node_traces = []
    community_colors = px.colors.qualitative.Set3

    for community_id in set(communities.values()):
        community_nodes = [node for node, comm in communities.items() if comm == community_id]

        if community_nodes:
            node_x = [pos[node][0] for node in community_nodes]
            node_y = [pos[node][1] for node in community_nodes]

            node_info = []
            node_sizes = []
            for node in community_nodes:
                node_data = graph.nodes[node]
                info = f"Email ID: {node}<br>"
                info += f"Subject: {node_data.get('title', 'N/A')}<br>"
                info += f"Sender: {node_data.get('sender', 'N/A')}<br>"
                info += f"Date: {node_data.get('date', 'N/A')}<br>"
                info += f"Classification: {node_data.get('classification', 'N/A')}<br>"
                info += f"Tone: {node_data.get('tone', 'N/A')}<br>"
                info += f"Community: {community_id}"
                node_info.append(info)
                node_sizes.append(node_data.get('size', 10))

            node_trace = go.Scatter(
                x=node_x, y=node_y,
                mode='markers',
                hoverinfo='text',
                hovertext=node_info,
                marker=dict(
                    size=node_sizes,
                    color=community_colors[community_id % len(community_colors)],
                    line=dict(width=2, color='white')
                ),
                name=f'Community {community_id}',
                showlegend=True
            )
            node_traces.append(node_trace)

    # Create edge traces
    edge_x = []
    edge_y = []

    for edge in graph.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines',
        showlegend=False
    )

    # Create figure
    fig = go.Figure(data=[edge_trace] + node_traces)

    fig.update_layout(
        title='Email Network Graph',
        showlegend=True,
        hovermode='closest',
        margin=dict(b=20, l=5, r=5, t=40),
        annotations=[dict(
            text="Email network visualization with community detection",
            showarrow=False,
            xref="paper", yref="paper",
            x=0.005, y=-0.002,
            xanchor='left', yanchor='bottom',
            font=dict(color="#000", size=12)
        )],
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=600
    )

    return fig


def create_similarity_heatmap(similarity_matrix, emails_df):
    """Create similarity heatmap"""
    fig = px.imshow(
        similarity_matrix,
        title="Email Similarity Heatmap",
        color_continuous_scale="Viridis",
        aspect="auto"
    )

    fig.update_layout(
        xaxis_title="Email Index",
        yaxis_title="Email Index",
        height=600
    )

    return fig


def create_temporal_analysis(emails_df):
    """Create temporal analysis visualizations"""
    if emails_df['date'].isna().all():
        return go.Figure()

    # Filter out NaN dates
    df_clean = emails_df.dropna(subset=['date'])

    if df_clean.empty:
        return go.Figure()

    # Email volume over time
    df_clean['date_only'] = df_clean['date'].dt.date
    daily_counts = df_clean.groupby('date_only').size().reset_index(name='count')

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=daily_counts['date_only'],
        y=daily_counts['count'],
        mode='lines+markers',
        name='Daily Email Volume',
        line=dict(color='blue', width=2),
        marker=dict(size=6)
    ))

    fig.update_layout(
        title="Email Volume Over Time",
        xaxis_title="Date",
        yaxis_title="Number of Emails",
        height=400
    )

    return fig


def create_classification_analysis(emails_df):
    """Create classification distribution analysis"""
    classification_counts = emails_df['classification'].value_counts()

    fig = go.Figure(data=[
        go.Bar(
            x=classification_counts.index,
            y=classification_counts.values,
            marker_color='lightblue'
        )
    ])

    fig.update_layout(
        title="Email Classification Distribution",
        xaxis_title="Classification",
        yaxis_title="Count",
        height=400
    )

    return fig


def create_sender_network(emails_df):
    """Create sender-recipient network"""
    G = nx.DiGraph()

    for _, row in emails_df.iterrows():
        sender = row['from']
        recipients = row['to'].split(',') if isinstance(row['to'], str) else []

        for recipient in recipients:
            recipient = recipient.strip()
            if recipient:
                if G.has_edge(sender, recipient):
                    G[sender][recipient]['weight'] += 1
                else:
                    G.add_edge(sender, recipient, weight=1)

    # Create visualization
    if len(G.nodes()) == 0:
        return go.Figure()

    pos = nx.spring_layout(G, k=2, iterations=50)

    # Node trace
    node_x = [pos[node][0] for node in G.nodes()]
    node_y = [pos[node][1] for node in G.nodes()]
    node_text = [f"User: {node}<br>Degree: {G.degree(node)}" for node in G.nodes()]
    node_sizes = [G.degree(node) * 5 + 10 for node in G.nodes()]

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        hovertext=node_text,
        text=[node.split('@')[0] for node in G.nodes()],
        textposition="middle center",
        marker=dict(size=node_sizes, color='lightcoral', line=dict(width=2))
    )

    # Edge traces
    edge_x = []
    edge_y = []

    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color='#888'),
        hoverinfo='none',
        mode='lines'
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        title='Sender-Recipient Network',
        showlegend=False,
        hovermode='closest',
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=600
    )

    return fig


def main():
    st.set_page_config(
        page_title="Email Relationship Analysis",
        page_icon="📧",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("📧 Email Relationship Analysis Dashboard")
    st.markdown("---")

    # Initialize analyzer
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = EmailAnalyzer()

    analyzer = st.session_state.analyzer

    # Sidebar for data upload and parameters
    with st.sidebar:
        st.header("Configuration")

        # File upload
        uploaded_file = st.file_uploader(
            "Upload Email JSON Data",
            type=['json'],
            help="Upload preprocessed email data in JSON format"
        )

        if uploaded_file is not None:
            try:
                data = json.load(uploaded_file)

                # Load and preprocess data
                with st.spinner("Processing email data..."):
                    emails_df = analyzer.load_and_preprocess_data(data)

                st.success(f"Loaded {len(emails_df)} emails")

                # Analysis parameters
                st.subheader("Analysis Parameters")

                similarity_threshold = st.slider(
                    "Similarity Threshold",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.1,
                    step=0.05,
                    help="Minimum similarity score to create connections"
                )

                # Date range filter
                if not emails_df['date'].isna().all():
                    date_range = st.date_input(
                        "Date Range",
                        value=(
                            emails_df['date'].min().date() if pd.notna(
                                emails_df['date'].min()) else datetime.now().date(),
                            emails_df['date'].max().date() if pd.notna(
                                emails_df['date'].max()) else datetime.now().date()
                        ),
                        help="Filter emails by date range"
                    )
                else:
                    date_range = None

                # Classification filter
                classifications = emails_df['classification'].unique()
                selected_classifications = st.multiselect(
                    "Email Classifications",
                    options=classifications,
                    default=classifications,
                    help="Filter by email classification"
                )

            except Exception as e:
                st.error(f"Error loading data: {str(e)}")
                return
        else:
            st.info("Please upload an email dataset to begin analysis")
            return

    # Filter data based on selections
    filtered_df = emails_df.copy()

    if date_range and not emails_df['date'].isna().all():
        filtered_df = filtered_df[
            (filtered_df['date'].dt.date >= date_range[0]) &
            (filtered_df['date'].dt.date <= date_range[1])
            ]

    if selected_classifications:
        filtered_df = filtered_df[
            filtered_df['classification'].isin(selected_classifications)
        ]

    # Main content tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview",
        "🕸️ Network Analysis",
        "🔥 Similarity Analysis",
        "📈 Temporal Analysis",
        "👥 Communication Patterns"
    ])

    with tab1:
        st.header("Dataset Overview")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Emails", len(filtered_df))

        with col2:
            unique_senders = filtered_df['from'].nunique()
            st.metric("Unique Senders", unique_senders)

        with col3:
            if not filtered_df['date'].isna().all():
                date_span = (filtered_df['date'].max() - filtered_df['date'].min()).days
                st.metric("Date Span (days)", date_span)
            else:
                st.metric("Date Span", "N/A")

        with col4:
            avg_recipients = filtered_df['num_recipients'].mean()
            st.metric("Avg Recipients", f"{avg_recipients:.1f}")

        # Classification distribution
        st.subheader("Email Classification Distribution")
        fig_class = create_classification_analysis(filtered_df)
        st.plotly_chart(fig_class, use_container_width=True)

        # Sample emails
        st.subheader("Sample Emails")
        sample_emails = filtered_df.head(5)[['date', 'from', 'subject', 'classification', 'tone_analysis']]
        st.dataframe(sample_emails, use_container_width=True)

    with tab2:
        st.header("Network Analysis")

        if len(filtered_df) > 1:
            with st.spinner("Generating embeddings and building network..."):
                # Generate embeddings
                embeddings, vectorizer = analyzer.generate_embeddings(filtered_df)

                # Calculate similarity matrix
                similarity_matrix = analyzer.calculate_similarity_matrix(embeddings)

                # Build graph
                graph = analyzer.build_graph(similarity_matrix, similarity_threshold)

                # Detect communities
                communities = analyzer.detect_communities(graph)

                # Calculate centrality metrics
                centrality_metrics = analyzer.calculate_centrality_metrics(graph)

            col1, col2 = st.columns([3, 1])

            with col1:
                st.subheader("Email Network Graph")
                fig_network = create_network_visualization(graph, communities)
                st.plotly_chart(fig_network, use_container_width=True)

            with col2:
                st.subheader("Network Statistics")
                st.metric("Nodes", graph.number_of_nodes())
                st.metric("Edges", graph.number_of_edges())
                st.metric("Communities", len(set(communities.values())))

                if graph.number_of_nodes() > 0:
                    density = nx.density(graph)
                    st.metric("Graph Density", f"{density:.3f}")

                # Top central emails
                if centrality_metrics and 'degree_centrality' in centrality_metrics:
                    st.subheader("Most Central Emails")
                    top_central = sorted(
                        centrality_metrics['degree_centrality'].items(),
                        key=lambda x: x[1],
                        reverse=True
                    )[:5]

                    for email_id, centrality in top_central:
                        if email_id < len(filtered_df):
                            email_subject = filtered_df.iloc[email_id]['subject'][:30] + "..."
                            st.write(f"**{email_id}**: {email_subject}")
                            st.write(f"Centrality: {centrality:.3f}")
        else:
            st.warning("Need at least 2 emails for network analysis")

    with tab3:
        st.header("Similarity Analysis")

        if len(filtered_df) > 1:
            if hasattr(analyzer, 'similarity_matrix') and analyzer.similarity_matrix is not None:
                # Similarity heatmap
                st.subheader("Email Similarity Heatmap")
                fig_heatmap = create_similarity_heatmap(analyzer.similarity_matrix, filtered_df)
                st.plotly_chart(fig_heatmap, use_container_width=True)

                # Most similar email pairs
                st.subheader("Most Similar Email Pairs")
                n = len(analyzer.similarity_matrix)
                similarity_pairs = []

                for i in range(n):
                    for j in range(i + 1, n):
                        similarity_pairs.append({
                            'Email 1': i,
                            'Email 2': j,
                            'Similarity': analyzer.similarity_matrix[i][j],
                            'Subject 1': filtered_df.iloc[i]['subject'][:50] + "...",
                            'Subject 2': filtered_df.iloc[j]['subject'][:50] + "..."
                        })

                similarity_df = pd.DataFrame(similarity_pairs)
                similarity_df = similarity_df.sort_values('Similarity', ascending=False).head(10)
                st.dataframe(similarity_df, use_container_width=True)
            else:
                st.info("Run network analysis first to generate similarity matrix")
        else:
            st.warning("Need at least 2 emails for similarity analysis")

    with tab4:
        st.header("Temporal Analysis")

        if not filtered_df['date'].isna().all():
            # Email volume over time
            fig_temporal = create_temporal_analysis(filtered_df)
            st.plotly_chart(fig_temporal, use_container_width=True)

            # Time-based patterns
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Email Volume by Hour")
                hourly_counts = filtered_df['hour'].value_counts().sort_index()
                fig_hourly = px.bar(
                    x=hourly_counts.index,
                    y=hourly_counts.values,
                    title="Email Volume by Hour of Day"
                )
                st.plotly_chart(fig_hourly, use_container_width=True)

            with col2:
                st.subheader("Email Volume by Day of Week")
                day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
                daily_counts = filtered_df['day_of_week'].value_counts().sort_index()
                fig_daily = px.bar(
                    x=[day_names[i] for i in daily_counts.index],
                    y=daily_counts.values,
                    title="Email Volume by Day of Week"
                )
                st.plotly_chart(fig_daily, use_container_width=True)
        else:
            st.warning("No valid date information available for temporal analysis")

    with tab5:
        st.header("Communication Patterns")

        # Sender-recipient network
        st.subheader("Sender-Recipient Network")
        fig_sender_network = create_sender_network(filtered_df)
        st.plotly_chart(fig_sender_network, use_container_width=True)

        # Top communicators
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Most Active Senders")
            sender_counts = filtered_df['from'].value_counts().head(10)
            fig_senders = px.bar(
                x=sender_counts.values,
                y=sender_counts.index,
                orientation='h',
                title="Top Email Senders"
            )
            st.plotly_chart(fig_senders, use_container_width=True)

        with col2:
            st.subheader("Email Length Distribution")
            filtered_df['content_length'] = filtered_df['summary'].str.len()
            fig_length = px.histogram(
                filtered_df,
                x='content_length',
                title="Email Content Length Distribution",
                bins=30
            )
            st.plotly_chart(fig_length, use_container_width=True)

    # Export functionality
    st.markdown("---")
    st.subheader("Export Results")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("Export Filtered Data"):
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="filtered_email_data.csv",
                mime="text/csv"
            )

    with col2:
        if hasattr(analyzer, 'similarity_matrix') and analyzer.similarity_matrix is not None:
            if st.button("Export Similarity Matrix"):
                similarity_df = pd.DataFrame(analyzer.similarity_matrix)
                csv = similarity_df.to_csv(index=False)
                st.download_button(
                    label="Download Similarity Matrix",
                    data=csv,
                    file_name="similarity_matrix.csv",
                    mime="text/csv"
                )

    with col3:
        if hasattr(analyzer, 'graph') and analyzer.graph is not None:
            if st.button("Export Graph Data"):
                # Export as edge list
                edge_list = []
                for edge in analyzer.graph.edges(data=True):
                    edge_list.append({
                        'source': edge[0],
                        'target': edge[1],
                        'weight': edge[2].get('weight', 1)
                    })

                edge_df = pd.DataFrame(edge_list)
                csv = edge_df.to_csv(index=False)
                st.download_button(
                    label="Download Graph Edges",
                    data=csv,
                    file_name="graph_edges.csv",
                    mime="text/csv"
                )


if __name__ == "__main__":
    main()