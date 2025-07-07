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
from sklearn.cluster import KMeans, DBSCAN
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
from typing import List, Dict, Tuple, Optional
import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading

# Optional LangChain imports (install if needed)
try:
    from langchain.llms import Ollama
    from langchain.embeddings import OllamaEmbeddings
    from langchain.prompts import PromptTemplate
    from langchain.chains import LLMChain

    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    st.warning("LangChain not available. Install with: pip install langchain")

warnings.filterwarnings('ignore')


@st.cache_resource
def download_nltk_data():
    """Download required NLTK data"""
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')


download_nltk_data()


class AdvancedEmailAnalyzer:
    """Enhanced email analyzer with advanced NLP and semantic analysis capabilities"""

    def __init__(self, use_ollama: bool = False):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.emails_df = None
        self.similarity_matrix = None
        self.semantic_similarity_matrix = None
        self.graph = None
        self.embeddings = None
        self.semantic_embeddings = None
        self.use_ollama = use_ollama and LANGCHAIN_AVAILABLE

        # Initialize Ollama if available and requested
        if self.use_ollama:
            self._initialize_ollama()

    def _initialize_ollama(self):
        """Initialize Ollama models for semantic analysis"""
        try:
            self.llm = Ollama(model="mistral", temperature=0.1)
            self.embedding_model = OllamaEmbeddings(model="mistral")

            # Create analysis chains
            self._setup_analysis_chains()

            st.success("✅ Ollama models initialized successfully")
        except Exception as e:
            st.error(f"❌ Failed to initialize Ollama: {str(e)}")
            self.use_ollama = False

    def _setup_analysis_chains(self):
        """Setup LangChain analysis chains"""
        # Topic extraction chain
        topic_prompt = PromptTemplate(
            input_variables=["text"],
            template="""
            Analyze the following email content and extract the main topics and themes.
            Return a list of 3-5 key topics, separated by commas.

            Email content: {text}

            Topics:
            """
        )
        self.topic_chain = LLMChain(llm=self.llm, prompt=topic_prompt)

        # Relationship analysis chain
        relationship_prompt = PromptTemplate(
            input_variables=["email1", "email2"],
            template="""
            Compare these two emails and determine their relationship strength and type.
            Rate the relationship from 0-10 and provide the relationship type.

            Email 1: {email1}
            Email 2: {email2}

            Relationship Score (0-10): 
            Relationship Type:
            """
        )
        self.relationship_chain = LLMChain(llm=self.llm, prompt=relationship_prompt)

    def clean_text(self, text: str) -> str:
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
    def load_and_preprocess_data(_self, data: Dict) -> pd.DataFrame:
        """Load and preprocess email data with enhanced features"""
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
        df['text_length'] = df['summary'].str.len()
        df['word_count'] = df['combined_text'].str.split().str.len()

        # Extract entities information
        df['num_people'] = df['entities'].apply(
            lambda x: len(x.get('people', [])) if isinstance(x, dict) else 0
        )
        df['num_organizations'] = df['entities'].apply(
            lambda x: len(x.get('organizations', [])) if isinstance(x, dict) else 0
        )
        df['num_locations'] = df['entities'].apply(
            lambda x: len(x.get('locations', [])) if isinstance(x, dict) else 0
        )

        _self.emails_df = df
        return df

    @st.cache_data
    def generate_traditional_embeddings(_self, df: pd.DataFrame) -> Tuple[np.ndarray, object]:
        """Generate TF-IDF embeddings for emails"""
        vectorizer = TfidfVectorizer(
            max_features=1000,
            min_df=2,
            max_df=0.8,
            ngram_range=(1, 2),
            stop_words='english'
        )

        embeddings = vectorizer.fit_transform(df['combined_text'].fillna(''))
        _self.embeddings = embeddings.toarray()

        return _self.embeddings, vectorizer

    async def generate_semantic_embeddings(self, df: pd.DataFrame) -> np.ndarray:
        """Generate semantic embeddings using Ollama"""
        if not self.use_ollama:
            return None

        texts = df['combined_text'].fillna('').tolist()

        try:
            with st.spinner("Generating semantic embeddings with Ollama..."):
                # Process in batches to avoid memory issues
                batch_size = 10
                all_embeddings = []

                for i in range(0, len(texts), batch_size):
                    batch = texts[i:i + batch_size]
                    batch_embeddings = await asyncio.to_thread(
                        self.embedding_model.embed_documents, batch
                    )
                    all_embeddings.extend(batch_embeddings)

                self.semantic_embeddings = np.array(all_embeddings)
                return self.semantic_embeddings

        except Exception as e:
            st.error(f"Error generating semantic embeddings: {str(e)}")
            return None

    @st.cache_data
    def calculate_similarity_matrix(_self, embeddings: np.ndarray) -> np.ndarray:
        """Calculate cosine similarity matrix"""
        similarity_matrix = cosine_similarity(embeddings)
        _self.similarity_matrix = similarity_matrix
        return similarity_matrix

    def calculate_semantic_similarity_matrix(self, semantic_embeddings: np.ndarray) -> np.ndarray:
        """Calculate semantic similarity matrix from Ollama embeddings"""
        if semantic_embeddings is None:
            return None

        semantic_similarity = cosine_similarity(semantic_embeddings)
        self.semantic_similarity_matrix = semantic_similarity
        return semantic_similarity

    def build_hybrid_graph(self, traditional_sim: np.ndarray, semantic_sim: Optional[np.ndarray] = None,
                           threshold: float = 0.1, semantic_weight: float = 0.3) -> nx.Graph:
        """Build network graph combining traditional and semantic similarities"""
        G = nx.Graph()

        # Add nodes with enhanced metadata
        for i, row in self.emails_df.iterrows():
            G.add_node(
                i,
                title=row['subject'][:50] + '...' if len(str(row['subject'])) > 50 else str(row['subject']),
                sender=row['from'],
                date=row['date'].strftime('%Y-%m-%d %H:%M') if pd.notna(row['date']) else 'Unknown',
                classification=row['classification'],
                tone=row['tone_analysis'],
                size=10 + row['num_recipients'] * 2,
                word_count=row['word_count'],
                num_people=row['num_people'],
                num_organizations=row['num_organizations'],
                num_locations=row['num_locations']
            )

        # Calculate combined similarity
        if semantic_sim is not None:
            # Weighted combination of traditional and semantic similarities
            combined_sim = (1 - semantic_weight) * traditional_sim + semantic_weight * semantic_sim
        else:
            combined_sim = traditional_sim

        # Add edges based on combined similarity
        n = len(combined_sim)
        for i in range(n):
            for j in range(i + 1, n):
                if combined_sim[i][j] > threshold:
                    edge_attrs = {'weight': combined_sim[i][j]}

                    # Add traditional and semantic components if available
                    edge_attrs['traditional_sim'] = traditional_sim[i][j]
                    if semantic_sim is not None:
                        edge_attrs['semantic_sim'] = semantic_sim[i][j]

                    G.add_edge(i, j, **edge_attrs)

        self.graph = G
        return G

    def advanced_community_detection(self, graph: nx.Graph, method: str = 'louvain') -> Dict:
        """Advanced community detection with multiple algorithms"""
        if method == 'louvain':
            partition = community.best_partition(graph, resolution=1.0)
        elif method == 'greedy_modularity':
            communities = community.greedy_modularity_communities(graph)
            partition = {}
            for i, comm in enumerate(communities):
                for node in comm:
                    partition[node] = i
        else:
            # Fallback to louvain
            partition = community.best_partition(graph, resolution=1.0)

        # Add community info to nodes
        for node in graph.nodes():
            graph.nodes[node]['community'] = partition.get(node, 0)

        return partition

    def calculate_advanced_centrality_metrics(self, graph: nx.Graph) -> Dict:
        """Calculate comprehensive centrality metrics"""
        metrics = {}

        if len(graph.nodes()) > 0:
            metrics['degree_centrality'] = nx.degree_centrality(graph)
            metrics['betweenness_centrality'] = nx.betweenness_centrality(graph)
            metrics['closeness_centrality'] = nx.closeness_centrality(graph)

            try:
                metrics['eigenvector_centrality'] = nx.eigenvector_centrality(graph, max_iter=1000)
            except:
                metrics['eigenvector_centrality'] = {node: 0 for node in graph.nodes()}

            # Additional metrics
            metrics['clustering_coefficient'] = nx.clustering(graph)
            metrics['pagerank'] = nx.pagerank(graph)

            # Core number (k-core decomposition)
            metrics['core_number'] = nx.core_number(graph)

        return metrics

    def perform_topic_modeling(self, df: pd.DataFrame, n_topics: int = 5) -> Dict:
        """Perform topic modeling on email content"""
        from sklearn.feature_extraction.text import CountVectorizer
        from sklearn.decomposition import LatentDirichletAllocation

        # Prepare text data
        texts = df['combined_text'].fillna('').tolist()

        # Vectorize
        vectorizer = CountVectorizer(
            max_features=100,
            min_df=2,
            max_df=0.8,
            stop_words='english'
        )

        doc_term_matrix = vectorizer.fit_transform(texts)

        # LDA
        lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
        lda.fit(doc_term_matrix)

        # Extract topics
        feature_names = vectorizer.get_feature_names_out()
        topics = {}

        for topic_idx, topic in enumerate(lda.components_):
            top_words = [feature_names[i] for i in topic.argsort()[-10:]]
            topics[f'Topic {topic_idx + 1}'] = top_words

        # Assign topics to documents
        doc_topic_probs = lda.transform(doc_term_matrix)
        df['dominant_topic'] = doc_topic_probs.argmax(axis=1)
        df['topic_probability'] = doc_topic_probs.max(axis=1)

        return topics, doc_topic_probs

    def detect_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect anomalous emails using various techniques"""
        from sklearn.ensemble import IsolationForest
        from sklearn.preprocessing import StandardScaler

        # Features for anomaly detection
        features = ['text_length', 'word_count', 'num_recipients',
                    'num_people', 'num_organizations', 'num_locations']

        # Handle missing values
        feature_data = df[features].fillna(0)

        # Standardize features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(feature_data)

        # Isolation Forest
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        anomaly_labels = iso_forest.fit_predict(scaled_features)

        df['is_anomaly'] = anomaly_labels == -1
        df['anomaly_score'] = iso_forest.score_samples(scaled_features)

        return df


def create_enhanced_network_visualization(graph: nx.Graph, communities: Dict,
                                          centrality_metrics: Dict) -> go.Figure:
    """Create enhanced network visualization with centrality information"""
    if len(graph.nodes()) == 0:
        return go.Figure()

    # Use spring layout for positioning
    pos = nx.spring_layout(graph, k=1, iterations=50)

    # Prepare node traces by community
    node_traces = []
    community_colors = px.colors.qualitative.Set3

    # Get centrality values for sizing
    degree_centrality = centrality_metrics.get('degree_centrality', {})

    for community_id in set(communities.values()):
        community_nodes = [node for node, comm in communities.items() if comm == community_id]

        if community_nodes:
            node_x = [pos[node][0] for node in community_nodes]
            node_y = [pos[node][1] for node in community_nodes]

            node_info = []
            node_sizes = []
            for node in community_nodes:
                node_data = graph.nodes[node]
                centrality = degree_centrality.get(node, 0)

                info = f"Email ID: {node}<br>"
                info += f"Subject: {node_data.get('title', 'N/A')}<br>"
                info += f"Sender: {node_data.get('sender', 'N/A')}<br>"
                info += f"Date: {node_data.get('date', 'N/A')}<br>"
                info += f"Classification: {node_data.get('classification', 'N/A')}<br>"
                info += f"Tone: {node_data.get('tone', 'N/A')}<br>"
                info += f"Word Count: {node_data.get('word_count', 'N/A')}<br>"
                info += f"People: {node_data.get('num_people', 0)}<br>"
                info += f"Organizations: {node_data.get('num_organizations', 0)}<br>"
                info += f"Locations: {node_data.get('num_locations', 0)}<br>"
                info += f"Centrality: {centrality:.3f}<br>"
                info += f"Community: {community_id}"
                node_info.append(info)

                # Size based on centrality and metadata
                base_size = node_data.get('size', 10)
                centrality_size = centrality * 20
                node_sizes.append(max(base_size + centrality_size, 5))

            node_trace = go.Scatter(
                x=node_x, y=node_y,
                mode='markers',
                hoverinfo='text',
                hovertext=node_info,
                marker=dict(
                    size=node_sizes,
                    color=community_colors[community_id % len(community_colors)],
                    line=dict(width=2, color='white'),
                    opacity=0.8
                ),
                name=f'Community {community_id}',
                showlegend=True
            )
            node_traces.append(node_trace)

    # Create edge traces with varying thickness based on weight
    edge_x = []
    edge_y = []
    edge_weights = []

    for edge in graph.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        weight = edge[2].get('weight', 0.1)

        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_weights.append(weight)

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color='rgba(136,136,136,0.5)'),
        hoverinfo='none',
        mode='lines',
        showlegend=False
    )

    # Create figure
    fig = go.Figure(data=[edge_trace] + node_traces)

    fig.update_layout(
        title='Enhanced Email Network Graph with Centrality Metrics',
        showlegend=True,
        hovermode='closest',
        margin=dict(b=20, l=5, r=5, t=40),
        annotations=[dict(
            text="Node size reflects centrality importance",
            showarrow=False,
            xref="paper", yref="paper",
            x=0.005, y=-0.002,
            xanchor='left', yanchor='bottom',
            font=dict(color="#000", size=12)
        )],
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=700
    )

    return fig


def create_topic_visualization(topics: Dict, doc_topic_probs: np.ndarray) -> go.Figure:
    """Create topic modeling visualization"""
    # Topic distribution
    topic_sizes = doc_topic_probs.sum(axis=0)

    fig = go.Figure(data=[
        go.Bar(
            x=[f"Topic {i + 1}" for i in range(len(topic_sizes))],
            y=topic_sizes,
            text=[f"Top words: {', '.join(topics[f'Topic {i + 1}'][:3])}" for i in range(len(topic_sizes))],
            textposition='outside',
            marker_color='lightblue'
        )
    ])

    fig.update_layout(
        title="Topic Distribution in Email Corpus",
        xaxis_title="Topics",
        yaxis_title="Topic Strength",
        height=500
    )

    return fig


def create_anomaly_visualization(df: pd.DataFrame) -> go.Figure:
    """Create anomaly detection visualization"""
    fig = px.scatter(
        df,
        x='text_length',
        y='word_count',
        color='is_anomaly',
        size='anomaly_score',
        hover_data=['subject', 'from', 'classification'],
        title="Email Anomaly Detection",
        color_discrete_map={True: 'red', False: 'blue'}
    )

    fig.update_layout(height=500)
    return fig

# Main function would be similar to the previous version but with enhanced features
# The main() function would include additional tabs for:
# - Topic Modeling
# - Anomaly Detection
# - Advanced Centrality Analysis
# - Semantic Analysis (if Ollama is available)


def main():
    st.set_page_config(
        page_title="Advanced Email Relationship Analysis",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("🔬 Advanced Email Relationship Analysis Dashboard")
    st.markdown("*Powered by Advanced NLP, Graph Theory, and Semantic Analysis*")
    st.markdown("---")

    # Initialize analyzer
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = None

    # Sidebar for configuration
    with st.sidebar:
        st.header("🔧 Configuration")

        # Ollama integration option
        use_ollama = st.checkbox(
            "Enable Ollama Semantic Analysis",
            value=False,
            help="Requires Ollama with Mistral model installed locally"
        )

        if st.session_state.analyzer is None or st.button("Initialize Analyzer"):
            with st.spinner("Initializing analyzer..."):
                st.session_state.analyzer = AdvancedEmailAnalyzer(use_ollama=use_ollama)

        analyzer = st.session_state.analyzer

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

                st.success(f"✅ Loaded {len(emails_df)} emails")

                # Analysis parameters
                st.subheader("📊 Analysis Parameters")

                similarity_threshold = st.slider(
                    "Similarity Threshold",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.15,
                    step=0.05,
                    help="Minimum similarity score to create connections"
                )

                semantic_weight = st.slider(
                    "Semantic Analysis Weight",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.3,
                    step=0.1,
                    help="Weight for semantic similarity vs traditional TF-IDF",
                    disabled=not analyzer.use_ollama
                )

                community_method = st.selectbox(
                    "Community Detection Method",
                    options=['louvain', 'greedy_modularity'],
                    help="Algorithm for detecting email communities"
                )

                n_topics = st.slider(
                    "Number of Topics",
                    min_value=3,
                    max_value=15,
                    value=5,
                    help="Number of topics for topic modeling"
                )

                # Date range filter
                if not emails_df['date'].isna().all():
                    min_date = emails_df['date'].min().date() if pd.notna(
                        emails_df['date'].min()) else datetime.now().date()
                    max_date = emails_df['date'].max().date() if pd.notna(
                        emails_df['date'].max()) else datetime.now().date()

                    date_range = st.date_input(
                        "Date Range",
                        value=(min_date, max_date),
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

                # Tone filter
                tones = emails_df['tone_analysis'].unique()
                selected_tones = st.multiselect(
                    "Email Tones",
                    options=tones,
                    default=tones,
                    help="Filter by email tone"
                )

            except Exception as e:
                st.error(f"❌ Error loading data: {str(e)}")
                return
        else:
            st.info("📁 Please upload an email dataset to begin analysis")
            return

    # Filter data based on selections
    filtered_df = emails_df.copy()

    if date_range and not emails_df['date'].isna().all():
        if len(date_range) == 2:
            filtered_df = filtered_df[
                (filtered_df['date'].dt.date >= date_range[0]) &
                (filtered_df['date'].dt.date <= date_range[1])
                ]

    if selected_classifications:
        filtered_df = filtered_df[
            filtered_df['classification'].isin(selected_classifications)
        ]

    if selected_tones:
        filtered_df = filtered_df[
            filtered_df['tone_analysis'].isin(selected_tones)
        ]

    # Main content tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "📊 Overview",
        "🕸️ Network Analysis",
        "🔥 Similarity Analysis",
        "📈 Temporal Analysis",
        "👥 Communication Patterns",
        "🎯 Topic Modeling",
        "⚠️ Anomaly Detection",
        "🧠 Semantic Analysis"
    ])

    with tab1:
        st.header("📊 Dataset Overview")

        # Key metrics
        col1, col2, col3, col4, col5 = st.columns(5)

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

        with col5:
            avg_word_count = filtered_df['word_count'].mean()
            st.metric("Avg Word Count", f"{avg_word_count:.0f}")

        # Advanced metrics
        col6, col7, col8, col9 = st.columns(4)

        with col6:
            avg_people = filtered_df['num_people'].mean()
            st.metric("Avg People Mentioned", f"{avg_people:.1f}")

        with col7:
            avg_orgs = filtered_df['num_organizations'].mean()
            st.metric("Avg Organizations", f"{avg_orgs:.1f}")

        with col8:
            avg_locations = filtered_df['num_locations'].mean()
            st.metric("Avg Locations", f"{avg_locations:.1f}")

        with col9:
            complexity_score = (filtered_df['word_count'] +
                                filtered_df['num_people'] * 5 +
                                filtered_df['num_organizations'] * 3).mean()
            st.metric("Complexity Score", f"{complexity_score:.0f}")

        # Visualizations
        col1, col2 = st.columns(2)

        with col1:
            # Classification distribution
            st.subheader("📋 Email Classification Distribution")
            fig_class = create_classification_analysis(filtered_df)
            st.plotly_chart(fig_class, use_container_width=True)

        with col2:
            # Tone analysis distribution
            st.subheader("😊 Tone Analysis Distribution")
            tone_counts = filtered_df['tone_analysis'].value_counts()
            fig_tone = px.pie(
                values=tone_counts.values,
                names=tone_counts.index,
                title="Email Tone Distribution"
            )
            st.plotly_chart(fig_tone, use_container_width=True)

        # Advanced distribution plots
        col3, col4 = st.columns(2)

        with col3:
            st.subheader("📏 Text Length Distribution")
            fig_length = px.histogram(
                filtered_df,
                x='text_length',
                title="Email Text Length Distribution",
                bins=30,
                marginal="box"
            )
            st.plotly_chart(fig_length, use_container_width=True)

        with col4:
            st.subheader("👤 Entity Distribution")
            entity_data = pd.DataFrame({
                'Entity Type': ['People', 'Organizations', 'Locations'],
                'Average Count': [
                    filtered_df['num_people'].mean(),
                    filtered_df['num_organizations'].mean(),
                    filtered_df['num_locations'].mean()
                ]
            })
            fig_entities = px.bar(
                entity_data,
                x='Entity Type',
                y='Average Count',
                title="Average Entity Mentions per Email"
            )
            st.plotly_chart(fig_entities, use_container_width=True)

        # Sample emails with enhanced information
        st.subheader("📝 Sample Emails (Enhanced)")
        sample_columns = ['date', 'from', 'subject', 'classification', 'tone_analysis',
                          'word_count', 'num_people', 'num_organizations', 'num_locations']
        sample_emails = filtered_df[sample_columns].head(5)
        st.dataframe(sample_emails, use_container_width=True)

    with tab2:
        st.header("🕸️ Advanced Network Analysis")

        if len(filtered_df) > 1:
            with st.spinner("🔄 Generating embeddings and building enhanced network..."):
                # Generate traditional embeddings
                traditional_embeddings, vectorizer = analyzer.generate_traditional_embeddings(filtered_df)
                traditional_similarity = analyzer.calculate_similarity_matrix(traditional_embeddings)

                # Generate semantic embeddings if Ollama is available
                semantic_embeddings = None
                semantic_similarity = None

                if analyzer.use_ollama:
                    try:
                        semantic_embeddings = asyncio.run(analyzer.generate_semantic_embeddings(filtered_df))
                        if semantic_embeddings is not None:
                            semantic_similarity = analyzer.calculate_semantic_similarity_matrix(semantic_embeddings)
                    except Exception as e:
                        st.warning(f"⚠️ Semantic analysis failed: {str(e)}")

                # Build hybrid graph
                graph = analyzer.build_hybrid_graph(
                    traditional_similarity,
                    semantic_similarity,
                    similarity_threshold,
                    semantic_weight
                )

                # Advanced community detection
                communities = analyzer.advanced_community_detection(graph, community_method)

                # Calculate comprehensive centrality metrics
                centrality_metrics = analyzer.calculate_advanced_centrality_metrics(graph)

            col1, col2 = st.columns([3, 1])

            with col1:
                st.subheader("🌐 Enhanced Email Network Graph")
                fig_network = create_enhanced_network_visualization(graph, communities, centrality_metrics)
                st.plotly_chart(fig_network, use_container_width=True)

            with col2:
                st.subheader("📈 Network Statistics")
                st.metric("Nodes", graph.number_of_nodes())
                st.metric("Edges", graph.number_of_edges())
                st.metric("Communities", len(set(communities.values())))

                if graph.number_of_nodes() > 0:
                    density = nx.density(graph)
                    st.metric("Graph Density", f"{density:.3f}")

                    # Average clustering coefficient
                    avg_clustering = nx.average_clustering(graph)
                    st.metric("Avg Clustering", f"{avg_clustering:.3f}")

                    # Network diameter (if connected)
                    if nx.is_connected(graph):
                        diameter = nx.diameter(graph)
                        st.metric("Network Diameter", diameter)
                    else:
                        largest_cc = max(nx.connected_components(graph), key=len)
                        subgraph = graph.subgraph(largest_cc)
                        diameter = nx.diameter(subgraph)
                        st.metric("Largest Component Diameter", diameter)

                # Centrality analysis
                st.subheader("🎯 Centrality Rankings")
                centrality_type = st.selectbox(
                    "Select Centrality Metric",
                    options=['degree_centrality', 'betweenness_centrality',
                             'closeness_centrality', 'pagerank']
                )

                if centrality_type in centrality_metrics:
                    top_central = sorted(
                        centrality_metrics[centrality_type].items(),
                        key=lambda x: x[1],
                        reverse=True
                    )[:5]

                    for i, (email_id, centrality) in enumerate(top_central):
                        if email_id < len(filtered_df):
                            email_subject = filtered_df.iloc[email_id]['subject'][:25] + "..."
                            st.write(f"**{i + 1}.** Email {email_id}")
                            st.write(f"Subject: {email_subject}")
                            st.write(f"Score: {centrality:.3f}")
                            st.write("---")
        else:
            st.warning("⚠️ Need at least 2 emails for network analysis")

    with tab3:
        st.header("🔥 Advanced Similarity Analysis")

        if len(filtered_df) > 1 and hasattr(analyzer, 'similarity_matrix') and analyzer.similarity_matrix is not None:
            col1, col2 = st.columns(2)

            with col1:
                # Traditional similarity heatmap
                st.subheader("📊 Traditional Similarity Heatmap")
                fig_traditional = create_similarity_heatmap(analyzer.similarity_matrix, filtered_df)
                st.plotly_chart(fig_traditional, use_container_width=True)

            with col2:
                # Semantic similarity heatmap (if available)
                if analyzer.semantic_similarity_matrix is not None:
                    st.subheader("🧠 Semantic Similarity Heatmap")
                    fig_semantic = create_similarity_heatmap(analyzer.semantic_similarity_matrix, filtered_df)
                    st.plotly_chart(fig_semantic, use_container_width=True)
                else:
                    st.info("🔮 Enable Ollama for semantic similarity analysis")

            # Similarity comparison
            if analyzer.semantic_similarity_matrix is not None:
                st.subheader("⚖️ Similarity Method Comparison")

                # Calculate correlation between methods
                trad_flat = analyzer.similarity_matrix[np.triu_indices_from(analyzer.similarity_matrix, k=1)]
                sem_flat = analyzer.semantic_similarity_matrix[
                    np.triu_indices_from(analyzer.semantic_similarity_matrix, k=1)]
                correlation = np.corrcoef(trad_flat, sem_flat)[0, 1]

                st.metric("Method Correlation", f"{correlation:.3f}")

                # Scatter plot comparison
                fig_comparison = px.scatter(
                    x=trad_flat,
                    y=sem_flat,
                    title="Traditional vs Semantic Similarity",
                    labels={'x': 'Traditional Similarity', 'y': 'Semantic Similarity'},
                    trendline="ols"
                )
                st.plotly_chart(fig_comparison, use_container_width=True)

            # Most similar email pairs
            st.subheader("🔗 Most Similar Email Pairs")
            similarity_matrix = analyzer.similarity_matrix
            n = len(similarity_matrix)
            similarity_pairs = []

            for i in range(n):
                for j in range(i + 1, n):
                    pair_data = {
                        'Email 1 ID': i,
                        'Email 2 ID': j,
                        'Traditional Similarity': similarity_matrix[i][j],
                        'Subject 1': filtered_df.iloc[i]['subject'][:40] + "...",
                        'Subject 2': filtered_df.iloc[j]['subject'][:40] + "...",
                        'Sender 1': filtered_df.iloc[i]['from'],
                        'Sender 2': filtered_df.iloc[j]['from']
                    }

                    if analyzer.semantic_similarity_matrix is not None:
                        pair_data['Semantic Similarity'] = analyzer.semantic_similarity_matrix[i][j]

                    similarity_pairs.append(pair_data)

            similarity_df = pd.DataFrame(similarity_pairs)
            similarity_df = similarity_df.sort_values('Traditional Similarity', ascending=False).head(15)
            st.dataframe(similarity_df, use_container_width=True)
        else:
            st.info("🔄 Run network analysis first to generate similarity matrices")

    with tab4:
        st.header("📈 Temporal Analysis")

        if not filtered_df['date'].isna().all():
            # Enhanced temporal visualizations
            col1, col2 = st.columns(2)

            with col1:
                # Email volume over time
                fig_temporal = create_temporal_analysis(filtered_df)
                st.plotly_chart(fig_temporal, use_container_width=True)

            with col2:
                # Activity heatmap by hour and day
                st.subheader("🕐 Activity Heatmap")
                if not filtered_df['date'].isna().all():
                    filtered_df['hour'] = filtered_df['date'].dt.hour
                    filtered_df['day_name'] = filtered_df['date'].dt.day_name()

                    pivot_data = filtered_df.pivot_table(
                        values='email_id',
                        index='day_name',
                        columns='hour',
                        aggfunc='count',
                        fill_value=0
                    )

                    fig_heatmap = px.imshow(
                        pivot_data,
                        title="Email Activity by Day and Hour",
                        color_continuous_scale="Blues"
                    )
                    st.plotly_chart(fig_heatmap, use_container_width=True)

            # Time series analysis
            col3, col4 = st.columns(2)

            with col3:
                st.subheader("📊 Volume by Classification Over Time")
                time_class_data = filtered_df.groupby([
                    filtered_df['date'].dt.date, 'classification'
                ]).size().reset_index(name='count')

                fig_time_class = px.line(
                    time_class_data,
                    x='date',
                    y='count',
                    color='classification',
                    title="Email Volume by Classification Over Time"
                )
                st.plotly_chart(fig_time_class, use_container_width=True)

            with col4:
                st.subheader("😊 Tone Evolution")
                time_tone_data = filtered_df.groupby([
                    filtered_df['date'].dt.date, 'tone_analysis'
                ]).size().reset_index(name='count')

                fig_time_tone = px.area(
                    time_tone_data,
                    x='date',
                    y='count',
                    color='tone_analysis',
                    title="Email Tone Distribution Over Time"
                )
                st.plotly_chart(fig_time_tone, use_container_width=True)
        else:
            st.warning("⚠️ No valid date information available for temporal analysis")

    with tab5:
        st.header("👥 Communication Patterns")

        # Enhanced communication analysis
        col1, col2 = st.columns(2)

        with col1:
            # Sender-recipient network
            st.subheader("🌐 Communication Network")
            fig_sender_network = create_sender_network(filtered_df)
            st.plotly_chart(fig_sender_network, use_container_width=True)

        with col2:
            # Communication frequency matrix
            st.subheader("📊 Communication Frequency")
            comm_matrix = pd.crosstab(filtered_df['from'], filtered_df['classification'])
            fig_comm_freq = px.imshow(
                comm_matrix,
                title="Sender vs Classification Frequency",
                color_continuous_scale="Viridis"
            )
            st.plotly_chart(fig_comm_freq, use_container_width=True)

        # Advanced communication metrics
        col3, col4 = st.columns(2)

        with col3:
            st.subheader("🏆 Top Communicators")
            sender_stats = filtered_df.groupby('from').agg({
                'email_id': 'count',
                'num_recipients': 'mean',
                'word_count': 'mean',
                'num_people': 'sum',
                'num_organizations': 'sum'
            }).round(2)

            sender_stats.columns = ['Emails Sent', 'Avg Recipients', 'Avg Words',
                                    'Total People', 'Total Orgs']
            sender_stats = sender_stats.sort_values('Emails Sent', ascending=False).head(10)
            st.dataframe(sender_stats, use_container_width=True)

        with col4:
            st.subheader("🎯 Communication Efficiency")
            # Calculate efficiency metrics
            efficiency_data = filtered_df.groupby('from').agg({
                'word_count': 'mean',
                'num_recipients': 'mean',
                'num_people': 'mean',
                'num_organizations': 'mean'
            }).round(2)

            # Create efficiency score
            efficiency_data['efficiency_score'] = (
                                                          efficiency_data['num_people'] + efficiency_data[
                                                      'num_organizations']
                                                  ) / efficiency_data['word_count'] * 1000

            top_efficient = efficiency_data.sort_values('efficiency_score', ascending=False).head(10)

            fig_efficiency = px.bar(
                x=top_efficient.index,
                y=top_efficient['efficiency_score'],
                title="Communication Efficiency Score"
            )
            fig_efficiency.update_xaxes(tickangle=45)
            st.plotly_chart(fig_efficiency, use_container_width=True)

    with tab6:
        st.header("🎯 Topic Modeling Analysis")

        if len(filtered_df) > 5:  # Need minimum emails for topic modeling
            with st.spinner("🔄 Performing topic modeling..."):
                topics, doc_topic_probs = analyzer.perform_topic_modeling(filtered_df, n_topics)

            col1, col2 = st.columns([2, 1])

            with col1:
                # Topic visualization
                fig_topics = create_topic_visualization(topics, doc_topic_probs)
                st.plotly_chart(fig_topics, use_container_width=True)

            with col2:
                st.subheader("📋 Discovered Topics")
                for topic_name, words in topics.items():
                    st.write(f"**{topic_name}:**")
                    st.write(", ".join(words[:5]))
                    st.write("---")

            # Topic distribution over time
            if not filtered_df['date'].isna().all():
                st.subheader("📈 Topic Evolution Over Time")
                topic_time_data = filtered_df.groupby([
                    filtered_df['date'].dt.date, 'dominant_topic'
                ]).size().reset_index(name='count')

                fig_topic_time = px.line(
                    topic_time_data,
                    x='date',
                    y='count',
                    color='dominant_topic',
                    title="Topic Popularity Over Time"
                )
                st.plotly_chart(fig_topic_time, use_container_width=True)

            # Topic-sender relationship
            st.subheader("👤 Topic-Sender Relationships")
            topic_sender = pd.crosstab(filtered_df['from'], filtered_df['dominant_topic'])
            fig_topic_sender = px.imshow(
                topic_sender,
                title="Sender vs Topic Frequency",
                color_continuous_scale="Blues"
            )
            st.plotly_chart(fig_topic_sender, use_container_width=True)
        else:
            st.warning("⚠️ Need at least 6 emails for meaningful topic modeling")

    with tab7:
        st.header("⚠️ Anomaly Detection")

        with st.spinner("🔍 Detecting anomalies..."):
            anomaly_df = analyzer.detect_anomalies(filtered_df)

        # Anomaly statistics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            anomaly_count = anomaly_df['is_anomaly'].sum()
            st.metric("Anomalous Emails", anomaly_count)

        with col2:
            anomaly_percentage = (anomaly_count / len(anomaly_df)) * 100
            st.metric("Anomaly Rate", f"{anomaly_percentage:.1f}%")

        with col3:
            avg_anomaly_score = anomaly_df[anomaly_df['is_anomaly']]['anomaly_score'].mean()
            st.metric("Avg Anomaly Score", f"{avg_anomaly_score:.3f}")

        with col4:
            min_anomaly_score = anomaly_df['anomaly_score'].min()
            st.metric("Most Anomalous Score", f"{min_anomaly_score:.3f}")

        # Anomaly visualizations
        col1, col2 = st.columns(2)

        with col1:
            # Anomaly scatter plot
            fig_anomaly = create_anomaly_visualization(anomaly_df)
            st.plotly_chart(fig_anomaly, use_container_width=True)

        with col2:
            # Anomaly score distribution
            fig_anomaly_dist = px.histogram(
                anomaly_df,
                x='anomaly_score',
                color='is_anomaly',
                title="Anomaly Score Distribution",
                bins=30
            )
            st.plotly_chart(fig_anomaly_dist, use_container_width=True)

        # Anomalous emails details
        st.subheader("🔍 Detected Anomalous Emails")
        anomalous_emails = anomaly_df[anomaly_df['is_anomaly']].sort_values('anomaly_score')

        if len(anomalous_emails) > 0:
            display_cols = ['date', 'from', 'subject', 'classification', 'tone_analysis',
                            'text_length', 'word_count', 'num_recipients', 'anomaly_score']
            st.dataframe(anomalous_emails[display_cols], use_container_width=True)
        else:
            st.info("🎉 No anomalous emails detected with current parameters")

    with tab8:
        st.header("🧠 Semantic Analysis")

        if analyzer.use_ollama and analyzer.semantic_embeddings is not None:
            st.success("✅ Semantic analysis powered by Ollama Mistral")

            # Semantic clustering
            st.subheader("🔬 Semantic Clustering")

            with st.spinner("🔄 Performing semantic clustering..."):
                # Use TSNE for dimensionality reduction
                tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(filtered_df) - 1))
                semantic_2d = tsne.fit_transform(analyzer.semantic_embeddings)

                # K-means clustering on semantic embeddings
                n_clusters = min(5, len(filtered_df) // 2)
                if n_clusters > 1:
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                    semantic_clusters = kmeans.fit_predict(analyzer.semantic_embeddings)

                    # Create visualization
                    cluster_df = pd.DataFrame({
                        'x': semantic_2d[:, 0],
                        'y': semantic_2d[:, 1],
                        'cluster': semantic_clusters,
                        'subject': filtered_df['subject'].values,
                        'classification': filtered_df['classification'].values,
                        'tone': filtered_df['tone_analysis'].values
                    })

                    fig_semantic_cluster = px.scatter(
                        cluster_df,
                        x='x',
                        y='y',
                        color='cluster',
                        hover_data=['subject', 'classification', 'tone'],
                        title="Semantic Clustering of Emails (t-SNE Projection)",
                        color_continuous_scale="viridis"
                    )
                    st.plotly_chart(fig_semantic_cluster, use_container_width=True)

                    # Cluster analysis
                    st.subheader("📊 Semantic Cluster Analysis")
                    cluster_analysis = pd.DataFrame({
                        'Cluster': range(n_clusters),
                        'Size': [np.sum(semantic_clusters == i) for i in range(n_clusters)],
                        'Avg Word Count': [filtered_df[semantic_clusters == i]['word_count'].mean() for i in
                                           range(n_clusters)],
                        'Dominant Classification': [
                            filtered_df[semantic_clusters == i]['classification'].mode().iloc[0] if len(
                                filtered_df[semantic_clusters == i]) > 0 else 'N/A' for i in range(n_clusters)]
                    })
                    st.dataframe(cluster_analysis, use_container_width=True)

            # Semantic similarity insights
            st.subheader("🔍 Semantic Similarity Insights")

            if analyzer.semantic_similarity_matrix is not None:
                # Find semantically similar but textually different emails
                traditional_flat = analyzer.similarity_matrix[np.triu_indices_from(analyzer.similarity_matrix, k=1)]
                semantic_flat = analyzer.semantic_similarity_matrix[
                    np.triu_indices_from(analyzer.semantic_similarity_matrix, k=1)]

                # Find pairs with high semantic similarity but low traditional similarity
                diff_threshold = 0.3
                semantic_high = semantic_flat > 0.5
                traditional_low = traditional_flat < 0.2
                interesting_pairs = semantic_high & traditional_low

                if np.any(interesting_pairs):
                    st.success(
                        f"🎯 Found {np.sum(interesting_pairs)} semantically similar but textually different email pairs")

                    # Show some examples
                    indices = np.where(interesting_pairs)[0][:5]  # Show top 5
                    triu_indices = np.triu_indices_from(analyzer.similarity_matrix, k=1)

                    for idx in indices:
                        i, j = triu_indices[0][idx], triu_indices[1][idx]
                        st.write(f"**Email {i} & {j}:**")
                        st.write(f"- Semantic Similarity: {semantic_flat[idx]:.3f}")
                        st.write(f"- Traditional Similarity: {traditional_flat[idx]:.3f}")
                        st.write(f"- Subject 1: {filtered_df.iloc[i]['subject'][:50]}...")
                        st.write(f"- Subject 2: {filtered_df.iloc[j]['subject'][:50]}...")
                        st.write("---")
                else:
                    st.info("📊 No significantly different semantic vs traditional similarities found")
        else:
            st.info("🔮 Enable Ollama integration to access semantic analysis features")
            st.markdown("""
            **To enable semantic analysis:**
            1. Install Ollama: https://ollama.ai/
            2. Pull Mistral model: `ollama pull mistral`
            3. Install LangChain: `pip install langchain`
            4. Enable the option in the sidebar
            """)

    # Export functionality
    st.markdown("---")
    st.subheader("💾 Export Results")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("📊 Export Filtered Data"):
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name=f"email_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

    with col2:
        if hasattr(analyzer, 'similarity_matrix') and analyzer.similarity_matrix is not None:
            if st.button("🔥 Export Similarity Data"):
                similarity_df = pd.DataFrame(analyzer.similarity_matrix)
                csv = similarity_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Similarity Matrix",
                    data=csv,
                    file_name=f"similarity_matrix_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )

    with col3:
        if hasattr(analyzer, 'graph') and analyzer.graph is not None:
            if st.button("🕸️ Export Network Data"):
                # Export as edge list with attributes
                edge_list = []
                for edge in analyzer.graph.edges(data=True):
                    edge_data = {
                        'source': edge[0],
                        'target': edge[1],
                        'weight': edge[2].get('weight', 1),
                        'traditional_sim': edge[2].get('traditional_sim', 0),
                        'semantic_sim': edge[2].get('semantic_sim', 0)
                    }
                    edge_list.append(edge_data)

                edge_df = pd.DataFrame(edge_list)
                csv = edge_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Network Edges",
                    data=csv,
                    file_name=f"network_edges_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )

    with col4:
        if st.button("📋 Export Analysis Report"):
            # Generate comprehensive analysis report
            report_data = {
                'analysis_timestamp': datetime.now().isoformat(),
                'dataset_summary': {
                    'total_emails': len(filtered_df),
                    'unique_senders': filtered_df['from'].nunique(),
                    'date_range': {
                        'start': filtered_df['date'].min().isoformat() if not filtered_df[
                            'date'].isna().all() else None,
                        'end': filtered_df['date'].max().isoformat() if not filtered_df['date'].isna().all() else None
                    },
                    'avg_recipients': filtered_df['num_recipients'].mean(),
                    'avg_word_count': filtered_df['word_count'].mean()
                },
                'network_metrics': {},
                'anomaly_summary': {},
                'parameters_used': {
                    'similarity_threshold': similarity_threshold,
                    'semantic_weight': semantic_weight,
                    'community_method': community_method,
                    'n_topics': n_topics,
                    'ollama_enabled': analyzer.use_ollama
                }
            }

            # Add network metrics if available
            if hasattr(analyzer, 'graph') and analyzer.graph is not None:
                report_data['network_metrics'] = {
                    'nodes': analyzer.graph.number_of_nodes(),
                    'edges': analyzer.graph.number_of_edges(),
                    'density': nx.density(analyzer.graph),
                    'communities': len(set(communities.values())) if 'communities' in locals() else 0
                }

            # Add anomaly summary if available
            if 'anomaly_df' in locals():
                report_data['anomaly_summary'] = {
                    'total_anomalies': int(anomaly_df['is_anomaly'].sum()),
                    'anomaly_rate': float((anomaly_df['is_anomaly'].sum() / len(anomaly_df)) * 100),
                    'avg_anomaly_score': float(anomaly_df[anomaly_df['is_anomaly']]['anomaly_score'].mean()) if
                    anomaly_df['is_anomaly'].any() else 0
                }

            report_json = json.dumps(report_data, indent=2, default=str)
            st.download_button(
                label="📥 Download Analysis Report",
                data=report_json,
                file_name=f"analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )


if __name__ == "__main__":
    main()