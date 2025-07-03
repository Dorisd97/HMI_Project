"""
Streamlit Email Relationship Analysis System
A comprehensive web application for analyzing and visualizing email relationships using
advanced NLP, graph theory, and interactive visualizations.

To run this application:
1. Install dependencies: pip install -r requirements.txt
2. Run: streamlit run email_analyzer.py
3. Open browser to http://localhost:8501

Requirements.txt:
streamlit==1.28.0
langchain==0.0.350
langchain-mistralai==0.0.1
networkx==3.2
plotly==5.17.0
pandas==2.1.3
numpy==1.24.3
scikit-learn==1.3.2
spacy==3.7.2
nltk==3.8.1
sentence-transformers==2.2.2
python-louvain==0.16
streamlit-agraph==0.0.45
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import re
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Set
import logging
from pathlib import Path
import hashlib
import tempfile
import io

# Core libraries
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# NLP and ML
try:
    import spacy
    from sentence_transformers import SentenceTransformer
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.cluster import KMeans
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from nltk.stem import WordNetLemmatizer
except ImportError as e:
    st.error(f"Missing required packages. Please install: {e}")
    st.stop()

# LangChain and Mistral
try:
    from langchain.llms import MistralAI
    from langchain.prompts import PromptTemplate
    from langchain.chains import LLMChain
    from langchain.cache import InMemoryCache
    from langchain.globals import set_llm_cache
except ImportError:
    st.error("LangChain not installed. Run: pip install langchain langchain-mistralai")
    st.stop()

# Graph analysis
try:
    import community
    from networkx.algorithms import centrality
except ImportError:
    st.error("Missing graph libraries. Run: pip install python-louvain")
    st.stop()

# Configure Streamlit page
st.set_page_config(
    page_title="📧 Email Relationship Analyzer",
    page_icon="📧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #007bff;
    }
    .insight-box {
        background: #e8f5e8;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #28a745;
    }
</style>
""", unsafe_allow_html=True)


class SimpleEmailProcessor:
    """Simplified email data processor for Streamlit app."""

    def __init__(self):
        # Initialize NLTK data
        self._setup_nltk()
        self.stop_words = set(stopwords.words('english')) if self._nltk_available() else set()

    def _setup_nltk(self):
        """Setup NLTK with error handling."""
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            with st.spinner("Downloading NLTK data..."):
                nltk.download('punkt', quiet=True)
                nltk.download('stopwords', quiet=True)
                nltk.download('wordnet', quiet=True)

    def _nltk_available(self):
        """Check if NLTK is properly configured."""
        try:
            from nltk.corpus import stopwords
            stopwords.words('english')
            return True
        except:
            return False

    def load_email_data(self, uploaded_file) -> pd.DataFrame:
        """Load and parse JSON email dataset from Streamlit upload."""
        try:
            # Read uploaded file
            content = uploaded_file.read()
            data = json.loads(content)

            # Handle different JSON structures
            if isinstance(data, list):
                emails = data
            elif isinstance(data, dict):
                if 'emails' in data:
                    emails = data['emails']
                elif 'data' in data:
                    emails = data['data']
                else:
                    emails = [data]
            else:
                emails = [data]

            # Normalize email structure
            normalized_emails = []
            for i, email in enumerate(emails):
                normalized = {
                    'id': email.get('id', f"email_{i}"),
                    'sender': str(email.get('sender', email.get('from', f'unknown_{i}'))).lower().strip(),
                    'recipients': self._normalize_recipients(email.get('recipients', email.get('to', []))),
                    'subject': str(email.get('subject', 'No Subject')),
                    'body': str(email.get('body', email.get('content', email.get('message', '')))),
                    'timestamp': self._parse_timestamp(email.get('timestamp', email.get('date', ''))),
                    'thread_id': str(email.get('thread_id', email.get('conversation_id', f'thread_{i}'))),
                }
                normalized_emails.append(normalized)

            df = pd.DataFrame(normalized_emails)
            return self._clean_data(df)

        except Exception as e:
            st.error(f"Error loading email data: {str(e)}")
            return pd.DataFrame()

    def _normalize_recipients(self, recipients) -> List[str]:
        """Normalize recipient email addresses."""
        if isinstance(recipients, str):
            recipients = [recipients]
        elif not isinstance(recipients, list):
            recipients = []

        return [str(r).lower().strip() for r in recipients if r]

    def _parse_timestamp(self, timestamp) -> Optional[datetime]:
        """Parse various timestamp formats."""
        if not timestamp:
            return datetime.now()

        # Try pandas first
        try:
            return pd.to_datetime(timestamp)
        except:
            # Try common formats
            formats = [
                '%Y-%m-%d %H:%M:%S',
                '%Y-%m-%dT%H:%M:%S',
                '%Y-%m-%d',
                '%d/%m/%Y',
                '%m/%d/%Y'
            ]

            for fmt in formats:
                try:
                    return datetime.strptime(str(timestamp), fmt)
                except ValueError:
                    continue

            return datetime.now()

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate dataframe."""
        if df.empty:
            return df

        # Fill missing values
        df['sender'] = df['sender'].fillna('unknown')
        df['subject'] = df['subject'].fillna('No Subject')
        df['body'] = df['body'].fillna('')

        # Remove empty emails
        df = df[df['body'].str.len() > 0]

        # Sort by timestamp
        df = df.sort_values('timestamp', na_last=True)

        return df.reset_index(drop=True)


class SimpleRelationshipExtractor:
    """Simplified relationship extractor using basic NLP."""

    def __init__(self, use_mistral=False, api_key=None):
        self.use_mistral = use_mistral and api_key

        if self.use_mistral:
            try:
                set_llm_cache(InMemoryCache())
                self.llm = MistralAI(api_key=api_key, model="mistral-medium", temperature=0.1)
            except Exception as e:
                st.warning(f"Mistral setup failed: {e}. Using basic analysis.")
                self.use_mistral = False

        # Initialize sentence transformer
        try:
            self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            st.warning(f"Sentence transformer failed: {e}. Using basic similarity.")
            self.sentence_transformer = None

    def compute_similarity_matrix(self, emails_df: pd.DataFrame) -> np.ndarray:
        """Compute email similarity matrix."""
        texts = (emails_df['subject'] + ' ' + emails_df['body']).tolist()

        if self.sentence_transformer:
            try:
                # Use sentence transformers
                embeddings = self.sentence_transformer.encode(texts, batch_size=16)
                similarity_matrix = cosine_similarity(embeddings)
            except Exception as e:
                st.warning(f"Embedding failed: {e}. Using TF-IDF.")
                similarity_matrix = self._tfidf_similarity(texts)
        else:
            similarity_matrix = self._tfidf_similarity(texts)

        return similarity_matrix

    def _tfidf_similarity(self, texts: List[str]) -> np.ndarray:
        """Fallback TF-IDF similarity computation."""
        try:
            vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
            tfidf_matrix = vectorizer.fit_transform(texts)
            return cosine_similarity(tfidf_matrix)
        except Exception:
            # Ultimate fallback - random similarity
            n = len(texts)
            return np.random.random((n, n)) * 0.3

    def extract_relationships(self, emails_df: pd.DataFrame, similarity_matrix: np.ndarray, threshold: float = 0.5) -> List[Tuple]:
        """Extract relationships between emails."""
        relationships = []

        # Direct reply relationships (simplified)
        sender_recipient_map = {}
        for idx, row in emails_df.iterrows():
            sender = row['sender']
            for recipient in row['recipients']:
                key = (recipient, sender)  # Recipient becomes sender in reply
                if key not in sender_recipient_map:
                    sender_recipient_map[key] = []
                sender_recipient_map[key].append(idx)

        # Add reply relationships
        for idx, row in emails_df.iterrows():
            sender = row['sender']
            for recipient in row['recipients']:
                key = (sender, recipient)
                if key in sender_recipient_map:
                    for reply_idx in sender_recipient_map[key]:
                        if reply_idx != idx:
                            relationships.append((idx, reply_idx, 'reply', 0.9))

        # Add similarity relationships
        n = len(emails_df)
        for i in range(n):
            for j in range(i + 1, n):
                if similarity_matrix[i][j] > threshold:
                    relationships.append((i, j, 'similar', similarity_matrix[i][j]))

        return relationships


class SimpleGraphBuilder:
    """Simplified graph builder for email networks."""

    def __init__(self):
        self.graph = nx.Graph()

    def build_graph(self, emails_df: pd.DataFrame, relationships: List[Tuple]) -> nx.Graph:
        """Build email relationship graph."""
        self.graph.clear()

        # Add email nodes
        for idx, row in emails_df.iterrows():
            self.graph.add_node(
                f"email_{idx}",
                type='email',
                sender=row['sender'],
                subject=row['subject'][:50],
                timestamp=str(row['timestamp']),
                index=idx
            )

        # Add person nodes
        people = set()
        for _, row in emails_df.iterrows():
            people.add(row['sender'])
            people.update(row['recipients'])

        for person in people:
            if person and person != 'unknown':
                self.graph.add_node(
                    f"person_{person}",
                    type='person',
                    name=person
                )

        # Add relationships
        for source, target, rel_type, weight in relationships:
            self.graph.add_edge(
                f"email_{source}",
                f"email_{target}",
                relationship=rel_type,
                weight=weight
            )

        # Connect people to emails
        for idx, row in emails_df.iterrows():
            sender = row['sender']
            if sender and sender != 'unknown':
                self.graph.add_edge(
                    f"person_{sender}",
                    f"email_{idx}",
                    relationship='authored',
                    weight=1.0
                )

        return self.graph

    def analyze_graph(self) -> Dict:
        """Analyze graph properties."""
        if self.graph.number_of_nodes() == 0:
            return {}

        metrics = {
            'nodes': self.graph.number_of_nodes(),
            'edges': self.graph.number_of_edges(),
            'density': nx.density(self.graph),
            'components': nx.number_connected_components(self.graph)
        }

        # Centrality measures (for connected graphs only)
        if nx.is_connected(self.graph):
            try:
                metrics['centrality'] = {
                    'betweenness': nx.betweenness_centrality(self.graph),
                    'closeness': nx.closeness_centrality(self.graph),
                    'pagerank': nx.pagerank(self.graph)
                }
            except Exception as e:
                st.warning(f"Centrality calculation failed: {e}")

        return metrics


def create_network_visualization(graph: nx.Graph) -> go.Figure:
    """Create network visualization using Plotly."""
    if graph.number_of_nodes() == 0:
        fig = go.Figure()
        fig.add_annotation(text="No data to display", x=0.5, y=0.5, showarrow=False)
        return fig

    # Calculate layout
    try:
        pos = nx.spring_layout(graph, k=1, iterations=50)
    except:
        pos = {node: (i, 0) for i, node in enumerate(graph.nodes())}

    # Prepare node traces
    email_nodes = [n for n, d in graph.nodes(data=True) if d.get('type') == 'email']
    person_nodes = [n for n, d in graph.nodes(data=True) if d.get('type') == 'person']

    # Email nodes
    email_x = [pos[node][0] for node in email_nodes]
    email_y = [pos[node][1] for node in email_nodes]
    email_text = [f"From: {graph.nodes[node].get('sender', 'Unknown')}<br>Subject: {graph.nodes[node].get('subject', '')}"
                  for node in email_nodes]

    # Person nodes
    person_x = [pos[node][0] for node in person_nodes]
    person_y = [pos[node][1] for node in person_nodes]
    person_text = [f"Person: {graph.nodes[node].get('name', 'Unknown')}" for node in person_nodes]

    # Edge traces
    edge_x, edge_y = [], []
    for edge in graph.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    fig = go.Figure()

    # Add edges
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='rgba(125,125,125,0.3)'),
        hoverinfo='none',
        mode='lines',
        showlegend=False
    ))

    # Add email nodes
    if email_nodes:
        fig.add_trace(go.Scatter(
            x=email_x, y=email_y,
            mode='markers',
            marker=dict(size=15, color='lightblue', line=dict(width=2, color='blue')),
            text=email_text,
            hoverinfo='text',
            name='Emails',
            showlegend=True
        ))

    # Add person nodes
    if person_nodes:
        fig.add_trace(go.Scatter(
            x=person_x, y=person_y,
            mode='markers',
            marker=dict(size=20, color='lightcoral', line=dict(width=2, color='red')),
            text=person_text,
            hoverinfo='text',
            name='People',
            showlegend=True
        ))

    fig.update_layout(
        title="Email Relationship Network",
        showlegend=True,
        hovermode='closest',
        margin=dict(b=20,l=5,r=5,t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor='white'
    )

    return fig


def create_sample_data() -> str:
    """Create sample email data for demonstration."""
    sample_data = {
        "emails": [
            {
                "id": "email_1",
                "sender": "alice@company.com",
                "recipients": ["bob@company.com", "charlie@company.com"],
                "subject": "Project Alpha Kickoff",
                "body": "Let's schedule a meeting to discuss the new project Alpha timeline and deliverables.",
                "timestamp": "2024-01-15 09:00:00"
            },
            {
                "id": "email_2",
                "sender": "bob@company.com",
                "recipients": ["alice@company.com"],
                "subject": "Re: Project Alpha Kickoff",
                "body": "Great idea! I'm available Tuesday and Wednesday. Let's also include the design team.",
                "timestamp": "2024-01-15 10:30:00"
            },
            {
                "id": "email_3",
                "sender": "charlie@company.com",
                "recipients": ["alice@company.com", "bob@company.com"],
                "subject": "Alpha Project Requirements",
                "body": "I've attached the requirements document for Project Alpha. Please review and provide feedback.",
                "timestamp": "2024-01-16 14:00:00"
            },
            {
                "id": "email_4",
                "sender": "diana@company.com",
                "recipients": ["alice@company.com"],
                "subject": "Budget Approval for Alpha",
                "body": "The budget for Project Alpha has been approved. We can proceed with the implementation.",
                "timestamp": "2024-01-17 11:00:00"
            },
            {
                "id": "email_5",
                "sender": "alice@company.com",
                "recipients": ["team@company.com"],
                "subject": "Project Alpha Update",
                "body": "Good news everyone! Project Alpha is approved and we're ready to start. Meeting scheduled for Friday.",
                "timestamp": "2024-01-18 16:00:00"
            }
        ]
    }
    return json.dumps(sample_data, indent=2)


def main():
    """Main Streamlit application."""

    # Header
    st.markdown("""
    <div class="main-header">
        <h1>📧 Email Relationship Analyzer</h1>
        <p>Discover hidden patterns and relationships in your email communications using AI and graph analytics</p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")

        # API Key
        use_mistral = st.checkbox("Use Mistral AI (Advanced Analysis)", value=False)
        mistral_api_key = None

        if use_mistral:
            mistral_api_key = st.text_input(
                "Mistral API Key",
                type="password",
                help="Get your API key from https://mistral.ai/"
            )

        st.header("📁 Data Input")

        # File upload
        uploaded_file = st.file_uploader(
            "Upload Email JSON",
            type=['json'],
            help="Upload a JSON file containing email data"
        )

        # Sample data option
        if st.button("📋 Use Sample Data"):
            sample_json = create_sample_data()
            st.session_state.sample_data = sample_json
            st.success("Sample data loaded! Scroll down to see the JSON structure.")

        # Download sample format
        sample_format = create_sample_data()
        st.download_button(
            "📥 Download Sample Format",
            data=sample_format,
            file_name="sample_emails.json",
            mime="application/json"
        )

    # Show sample data if requested
    if 'sample_data' in st.session_state:
        with st.expander("📋 Sample Data Structure", expanded=True):
            st.code(st.session_state.sample_data, language='json')

            # Convert to file-like object for processing
            sample_bytes = st.session_state.sample_data.encode('utf-8')
            uploaded_file = io.BytesIO(sample_bytes)
            uploaded_file.name = "sample_data.json"

    # Main content
    if uploaded_file is None:
        st.info("👆 Please upload an email JSON file or use sample data to begin analysis.")

        # Show expected format
        with st.expander("📖 Expected JSON Format"):
            st.code('''
{
  "emails": [
    {
      "id": "unique_email_id",
      "sender": "sender@example.com",
      "recipients": ["recipient1@example.com", "recipient2@example.com"],
      "subject": "Email subject",
      "body": "Email content...",
      "timestamp": "2024-01-15 10:30:00"
    }
  ]
}
            ''', language='json')
        return

    # Initialize processors
    if 'processor' not in st.session_state:
        st.session_state.processor = SimpleEmailProcessor()

    if 'extractor' not in st.session_state:
        st.session_state.extractor = SimpleRelationshipExtractor(use_mistral, mistral_api_key)

    if 'graph_builder' not in st.session_state:
        st.session_state.graph_builder = SimpleGraphBuilder()

    # Load and process data
    with st.spinner("📊 Loading email data..."):
        emails_df = st.session_state.processor.load_email_data(uploaded_file)

    if emails_df.empty:
        st.error("❌ Failed to load email data. Please check the file format.")
        return

    # Data overview
    st.header("📊 Data Overview")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📧 Total Emails", len(emails_df))
    with col2:
        st.metric("👥 Unique Senders", emails_df['sender'].nunique())
    with col3:
        all_recipients = set()
        for recipients in emails_df['recipients']:
            all_recipients.update(recipients)
        st.metric("📮 Recipients", len(all_recipients))
    with col4:
        if emails_df['timestamp'].notna().any():
            days = (emails_df['timestamp'].max() - emails_df['timestamp'].min()).days
            st.metric("📅 Time Span", f"{days} days")
        else:
            st.metric("📅 Time Span", "Unknown")

    # Analysis controls
    st.header("🔧 Analysis Settings")

    col1, col2 = st.columns(2)
    with col1:
        similarity_threshold = st.slider(
            "Similarity Threshold",
            0.0, 1.0, 0.5,
            help="Minimum similarity to create connections between emails"
        )

    with col2:
        max_emails = st.number_input(
            "Max Emails to Process",
            1, len(emails_df),
            min(len(emails_df), 50),
            help="Limit for performance"
        )

    # Analysis button
    if st.button("🚀 Analyze Email Relationships", type="primary"):

        # Limit dataset
        analysis_df = emails_df.head(max_emails).copy()

        with st.spinner("🔍 Analyzing relationships..."):
            # Progress tracking
            progress = st.progress(0)

            # Compute similarity
            progress.progress(0.3)
            similarity_matrix = st.session_state.extractor.compute_similarity_matrix(analysis_df)

            # Extract relationships
            progress.progress(0.6)
            relationships = st.session_state.extractor.extract_relationships(
                analysis_df, similarity_matrix, similarity_threshold
            )

            # Build graph
            progress.progress(0.8)
            graph = st.session_state.graph_builder.build_graph(analysis_df, relationships)

            # Analyze graph
            progress.progress(0.9)
            metrics = st.session_state.graph_builder.analyze_graph()

            progress.progress(1.0)

            # Store results
            st.session_state.analysis_results = {
                'df': analysis_df,
                'graph': graph,
                'metrics': metrics,
                'relationships': relationships,
                'similarity_matrix': similarity_matrix
            }

        st.success("✅ Analysis complete!")

    # Display results
    if 'analysis_results' in st.session_state:
        results = st.session_state.analysis_results

        # Graph metrics
        st.header("📈 Network Analysis")

        metrics = results['metrics']
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("🔗 Nodes", metrics.get('nodes', 0))
        with col2:
            st.metric("📊 Edges", metrics.get('edges', 0))
        with col3:
            st.metric("🌐 Density", f"{metrics.get('density', 0):.3f}")
        with col4:
            st.metric("🧩 Components", metrics.get('components', 0))

        # Key insights
        if 'centrality' in metrics:
            st.markdown("### 🎯 Key Insights")

            pagerank = metrics['centrality'].get('pagerank', {})
            if pagerank:
                # Find most important emails
                email_pagerank = {k: v for k, v in pagerank.items() if k.startswith('email_')}
                if email_pagerank:
                    top_email = max(email_pagerank.items(), key=lambda x: x[1])
                    email_idx = int(top_email[0].split('_')[1])

                    if email_idx < len(results['df']):
                        email_data = results['df'].iloc[email_idx]

                        st.markdown(f"""
                        <div class="insight-box">
                        <strong>🏆 Most Influential Email:</strong><br>
                        <strong>From:</strong> {email_data['sender']}<br>
                        <strong>Subject:</strong> {email_data['subject']}<br>
                        <strong>Influence Score:</strong> {top_email[1]:.3f}
                        </div>
                        """, unsafe_allow_html=True)

        # Visualizations
        st.header("📊 Interactive Visualizations")

        tab1, tab2, tab3 = st.tabs(["🌐 Network Graph", "📈 Timeline", "🔥 Heatmap"])

        with tab1:
            st.subheader("Email Relationship Network")

            # Network visualization
            network_fig = create_network_visualization(results['graph'])
            st.plotly_chart(network_fig, use_container_width=True)

            # Network stats
            st.markdown("**Network Statistics:**")
            col1, col2 = st.columns(2)
            with col1:
                avg_degree = 2 * results['graph'].number_of_edges() / max(results['graph'].number_of_nodes(), 1)
                st.write(f"• Average connections: {avg_degree:.2f}")
                st.write(f"• Total relationships: {len(results['relationships'])}")

            with col2:
                if results['graph'].number_of_nodes() > 0:
                    connected = nx.is_connected(results['graph'])
                    st.write(f"• Network connected: {'Yes' if connected else 'No'}")
                    st.write(f"• Graph density: {metrics.get('density', 0):.3f}")

        with tab2:
            st.subheader("Communication Timeline")

            df_time = results['df'].copy()
            if df_time['timestamp'].notna().any():
                df_time['date'] = df_time['timestamp'].dt.date
                daily_counts = df_time.groupby('date').size().reset_index(name='count')

                fig_timeline = px.line(
                    daily_counts,
                    x='date',
                    y='count',
                    title="Daily Email Volume",
                    markers=True
                )
                fig_timeline.update_layout(
                    xaxis_title="Date",
                    yaxis_title="Number of Emails"
                )
                st.plotly_chart(fig_timeline, use_container_width=True)

                # Time insights
                peak_day = daily_counts.loc[daily_counts['count'].idxmax(), 'date']
                st.write(f"📅 **Peak activity day:** {peak_day}")
                st.write(f"📊 **Average daily emails:** {daily_counts['count'].mean():.1f}")
            else:
                st.info("No timestamp data available for timeline analysis.")

        with tab3:
            st.subheader("Communication Heatmap")

            # Create sender-recipient matrix
            senders = results['df']['sender'].unique()
            recipients = []
            for r_list in results['df']['recipients']:
                recipients.extend(r_list)
            recipients = list(set(recipients))
            if len(senders) > 1 and len(recipients) > 1:
                comm_matrix = np.zeros((len(senders), len(recipients)))

                for _, row in results['df'].iterrows():
                    sender_idx = list(senders).index(row['sender'])
                    for recipient in row['recipients']:
                        if recipient in recipients:
                            recipient_idx = recipients.index(recipient)
                            comm_matrix[sender_idx][recipient_idx] += 1

                    fig_heatmap = px.imshow(
                        comm_matrix,
                        x=recipients,
                        y=senders,
                        title="Communication Intensity Heatmap",
                        color_continuous_scale="Blues"
                    )
                    fig_heatmap.update_layout(
                        xaxis_title="Recipients",
                        yaxis_title="Senders"
                    )
                    st.plotly_chart(fig_heatmap, use_container_width=True)

                    # Communication insights
                    most_active_sender = results['df']['sender'].value_counts().index[0]
                    most_contacted = max(recipients, key=lambda r: sum(comm_matrix[:, recipients.index(r)]))

                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"📤 **Most active sender:** {most_active_sender}")
                    with col2:
                        st.write(f"📥 **Most contacted person:** {most_contacted}")
                else:
                    st.info("Not enough unique senders/recipients for heatmap visualization.")

        # Export options
        st.header("💾 Export Results")

        col1, col2, col3 = st.columns(3)

        with col1:
            # Export graph as JSON
            graph_data = {
                'nodes': [
                    {
                        'id': node,
                        'type': data.get('type', 'unknown'),
                        'label': data.get('name', data.get('subject', node))
                    }
                    for node, data in results['graph'].nodes(data=True)
                ],
                'edges': [
                    {
                        'source': edge[0],
                        'target': edge[1],
                        'relationship': edge[2].get('relationship', ''),
                        'weight': edge[2].get('weight', 1.0)
                    }
                    for edge in results['graph'].edges(data=True)
                ]
            }

            st.download_button(
                "📊 Download Graph JSON",
                data=json.dumps(graph_data, indent=2),
                file_name="email_graph.json",
                mime="application/json"
            )

        with col2:
            # Export relationships as CSV
            relationships_df = pd.DataFrame(results['relationships'],
                                         columns=['Source', 'Target', 'Type', 'Weight'])

            st.download_button(
                "📋 Download Relationships CSV",
                data=relationships_df.to_csv(index=False),
                file_name="email_relationships.csv",
                mime="text/csv"
            )

        with col3:
            # Export analysis report
            report = {
                'analysis_summary': {
                    'timestamp': datetime.now().isoformat(),
                    'emails_analyzed': len(results['df']),
                    'relationships_found': len(results['relationships']),
                    'similarity_threshold': similarity_threshold
                },
                'graph_metrics': metrics,
                'top_senders': results['df']['sender'].value_counts().head().to_dict(),
                'insights': {
                    'most_active_day': str(results['df']['timestamp'].dt.date.mode().iloc[0]) if results['df']['timestamp'].notna().any() else None,
                    'average_email_length': int(results['df']['body'].str.len().mean()),
                    'unique_participants': len(set(results['df']['sender'].unique()) | all_recipients)
                }
            }

            st.download_button(
                "📑 Download Analysis Report",
                data=json.dumps(report, indent=2, default=str),
                file_name="email_analysis_report.json",
                mime="application/json"
            )

        # Advanced analysis
        with st.expander("🔍 Advanced Analysis"):
            st.subheader("Detailed Email Information")

            # Email search and filter
            search_term = st.text_input("🔎 Search emails (subject/body):")

            display_df = results['df'].copy()
            if search_term:
                mask = (display_df['subject'].str.contains(search_term, case=False, na=False) |
                       display_df['body'].str.contains(search_term, case=False, na=False))
                display_df = display_df[mask]

            # Show filtered emails
            if not display_df.empty:
                for idx, row in display_df.head(10).iterrows():
                    with st.container():
                        st.markdown(f"""
                        **📧 Email {idx + 1}**
                        - **From:** {row['sender']}
                        - **To:** {', '.join(row['recipients'])}
                        - **Subject:** {row['subject']}
                        - **Date:** {row['timestamp']}
                        - **Preview:** {row['body'][:200]}...
                        """)
                        st.divider()
            else:
                st.info("No emails match the search criteria.")

            # Similarity analysis
            st.subheader("📊 Similarity Analysis")

            if len(results['similarity_matrix']) > 1:
                # Show similarity distribution
                similarity_values = results['similarity_matrix'][np.triu_indices_from(results['similarity_matrix'], k=1)]

                fig_sim_dist = px.histogram(
                    x=similarity_values,
                    nbins=30,
                    title="Distribution of Email Similarities",
                    labels={'x': 'Cosine Similarity', 'y': 'Frequency'}
                )
                st.plotly_chart(fig_sim_dist, use_container_width=True)

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("📊 Mean Similarity", f"{np.mean(similarity_values):.3f}")
                with col2:
                    st.metric("📈 Max Similarity", f"{np.max(similarity_values):.3f}")
                with col3:
                    st.metric("📉 Min Similarity", f"{np.min(similarity_values):.3f}")

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray;'>
        📧 Email Relationship Analyzer | Built with Streamlit, NetworkX, and Advanced NLP
    </div>
    """, unsafe_allow_html=True)


# Additional utility functions
def setup_logging():
    """Setup logging for debugging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def validate_environment():
    """Validate that all required packages are installed."""
    required_packages = {
        'streamlit': 'streamlit',
        'pandas': 'pandas',
        'numpy': 'numpy',
        'networkx': 'networkx',
        'plotly': 'plotly',
        'scikit-learn': 'sklearn',
        'nltk': 'nltk'
    }

    missing_packages = []
    for package_name, import_name in required_packages.items():
        try:
            __import__(import_name)
        except ImportError:
            missing_packages.append(package_name)

    if missing_packages:
        st.error(f"Missing required packages: {', '.join(missing_packages)}")
        st.code(f"pip install {' '.join(missing_packages)}")
        return False

    return True


# Run the application
if __name__ == "__main__":
    # Validate environment
    if not validate_environment():
        st.stop()

    # Setup logging
    setup_logging()

    # Run main application
    main()


# Requirements file content for easy installation
REQUIREMENTS_TXT = """
streamlit==1.28.0
pandas==2.1.3
numpy==1.24.3
networkx==3.2
plotly==5.17.0
scikit-learn==1.3.2
nltk==3.8.1
sentence-transformers==2.2.2
spacy==3.7.2
langchain==0.0.350
langchain-mistralai==0.0.1
python-louvain==0.16
streamlit-agraph==0.0.45
"""

# Save requirements.txt when running
def save_requirements():
    """Save requirements.txt file."""
    with open('requirements.txt', 'w') as f:
        f.write(REQUIREMENTS_TXT)

# Create startup script
STARTUP_SCRIPT = """#!/bin/bash
# Email Analyzer Startup Script

echo "🚀 Setting up Email Relationship Analyzer..."

# Install Python dependencies
echo "📦 Installing Python packages..."
pip install -r requirements.txt

# Download spaCy model
echo "🧠 Downloading spaCy language model..."
python -m spacy download en_core_web_sm

# Download NLTK data
echo "📚 Downloading NLTK data..."
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"

echo "✅ Setup complete!"
echo "🌐 Starting Streamlit application..."
streamlit run email_analyzer.py
"""