# Email Relationship Analysis System
# A comprehensive solution for analyzing email relationships and generating interactive visualizations

import json
import re
import logging
import hashlib
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import asyncio

import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt

from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain.schema import BaseOutputParser

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =====================================================================
# Data Models and Configuration
# =====================================================================

@dataclass
class EmailNode:
    """Represents an email as a node in the relationship graph."""
    id: str
    sender: str
    recipients: List[str]
    subject: str
    content: str
    timestamp: datetime
    entities: Dict[str, List[str]]
    sentiment: float
    topic_cluster: int
    importance_score: float


@dataclass
class RelationshipEdge:
    """Represents a relationship between two emails."""
    source_id: str
    target_id: str
    weight: float
    relationship_type: str
    semantic_similarity: float
    temporal_proximity: float
    participant_overlap: float
    shared_entities: List[str]
    narrative_connection: str


@dataclass
class AnalysisConfig:
    """Configuration for analysis parameters."""
    semantic_weight: float = 0.4
    temporal_weight: float = 0.3
    participant_weight: float = 0.2
    entity_weight: float = 0.1
    min_relationship_threshold: float = 0.3
    max_temporal_days: int = 30
    cluster_count: int = 10


# =====================================================================
# Data Processing Pipeline
# =====================================================================

class EmailDataProcessor:
    """Handles preprocessing and cleaning of email data."""

    def __init__(self):
        self.name_patterns = self._compile_name_patterns()
        self.signature_patterns = self._compile_signature_patterns()

    def _compile_name_patterns(self) -> List[re.Pattern]:
        """Compile regex patterns for name normalization."""
        return [
            re.compile(r'([a-zA-Z]+)\.([a-zA-Z]+)@([a-zA-Z]+\.[a-zA-Z]+)'),
            re.compile(r'([a-zA-Z]+)@([a-zA-Z]+\.[a-zA-Z]+)'),
        ]

    def _compile_signature_patterns(self) -> List[re.Pattern]:
        """Compile regex patterns for signature removal."""
        return [
            re.compile(r'--\s*\n.*', re.DOTALL | re.MULTILINE),
            re.compile(r'Best regards,.*', re.DOTALL | re.IGNORECASE),
            re.compile(r'Sincerely,.*', re.DOTALL | re.IGNORECASE),
            re.compile(r'Thanks,.*', re.DOTALL | re.IGNORECASE),
        ]

    def normalize_email_address(self, email: str) -> str:
        """Normalize email addresses for consistent comparison."""
        return email.lower().strip()

    def extract_name_from_email(self, email: str) -> str:
        """Extract readable name from email address."""
        email = self.normalize_email_address(email)
        if '@' in email:
            local_part = email.split('@')[0]
            # Handle common patterns like first.last
            if '.' in local_part:
                parts = local_part.split('.')
                return ' '.join(part.title() for part in parts)
            return local_part.title()
        return email

    def clean_email_content(self, content: str) -> str:
        """Clean email content by removing signatures and formatting."""
        if not content:
            return ""

        # Remove signatures
        for pattern in self.signature_patterns:
            content = pattern.sub('', content)

        # Remove excessive whitespace
        content = re.sub(r'\n\s*\n', '\n\n', content)
        content = re.sub(r'\s+', ' ', content)

        return content.strip()

    def parse_timestamp(self, timestamp_str: str) -> datetime:
        """Parse timestamp from various formats."""
        formats = [
            "%d.%m.%Y %H:%M:%S",
            "%Y-%m-%d %H:%M:%S",
            "%m/%d/%Y %H:%M:%S",
        ]

        for fmt in formats:
            try:
                return datetime.strptime(timestamp_str, fmt)
            except ValueError:
                continue

        logger.warning(f"Could not parse timestamp: {timestamp_str}")
        return datetime.now()

    def process_email_data(self, email_data: Dict) -> EmailNode:
        """Process raw email data into structured EmailNode."""
        email_id = str(email_data.get('email_id', ''))

        # Extract and normalize participants
        sender = self.normalize_email_address(email_data.get('from', ''))
        recipients_raw = email_data.get('to', '')
        recipients = [
            self.normalize_email_address(r.strip())
            for r in recipients_raw.split(',') if r.strip()
        ]

        # Process content
        subject = email_data.get('subject', '')
        content = self.clean_email_content(email_data.get('summary', ''))

        # Parse timestamp
        timestamp = self.parse_timestamp(email_data.get('date', ''))

        # Extract entities (if available)
        entities = email_data.get('entities', {})

        return EmailNode(
            id=email_id,
            sender=sender,
            recipients=recipients,
            subject=subject,
            content=content,
            timestamp=timestamp,
            entities=entities,
            sentiment=0.0,  # Will be calculated later
            topic_cluster=0,  # Will be calculated later
            importance_score=0.0  # Will be calculated later
        )


# =====================================================================
# Relationship Extraction Engine
# =====================================================================

class RelationshipExtractor:
    """Extracts and scores relationships between emails using LLM and ML techniques."""

    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.llm = OllamaLLM(model="mistral")
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.email_vectors = None

    def _calculate_semantic_similarity(self, emails: List[EmailNode]) -> np.ndarray:
        """Calculate semantic similarity matrix using TF-IDF and cosine similarity."""
        # Combine subject and content for better semantic representation
        texts = [f"{email.subject} {email.content}" for email in emails]

        # Fit vectorizer and transform texts
        try:
            self.email_vectors = self.vectorizer.fit_transform(texts)
            similarity_matrix = cosine_similarity(self.email_vectors)
            return similarity_matrix
        except Exception as e:
            logger.error(f"Error calculating semantic similarity: {e}")
            return np.zeros((len(emails), len(emails)))

    def _calculate_temporal_proximity(self, emails: List[EmailNode]) -> np.ndarray:
        """Calculate temporal proximity matrix."""
        n = len(emails)
        proximity_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                if i != j:
                    time_diff = abs((emails[i].timestamp - emails[j].timestamp).days)
                    # Use exponential decay for temporal proximity
                    proximity = np.exp(-time_diff / self.config.max_temporal_days)
                    proximity_matrix[i][j] = proximity

        return proximity_matrix

    def _calculate_participant_overlap(self, emails: List[EmailNode]) -> np.ndarray:
        """Calculate participant overlap matrix."""
        n = len(emails)
        overlap_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                if i != j:
                    # Get all participants for each email
                    participants_i = set([emails[i].sender] + emails[i].recipients)
                    participants_j = set([emails[j].sender] + emails[j].recipients)

                    # Calculate Jaccard similarity
                    intersection = len(participants_i & participants_j)
                    union = len(participants_i | participants_j)

                    if union > 0:
                        overlap_matrix[i][j] = intersection / union

        return overlap_matrix

    def _calculate_entity_overlap(self, emails: List[EmailNode]) -> np.ndarray:
        """Calculate entity overlap matrix."""
        n = len(emails)
        overlap_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                if i != j:
                    # Extract all entities from both emails
                    entities_i = set()
                    entities_j = set()

                    for entity_type, entity_list in emails[i].entities.items():
                        entities_i.update(entity_list)

                    for entity_type, entity_list in emails[j].entities.items():
                        entities_j.update(entity_list)

                    # Calculate Jaccard similarity
                    if entities_i or entities_j:
                        intersection = len(entities_i & entities_j)
                        union = len(entities_i | entities_j)
                        if union > 0:
                            overlap_matrix[i][j] = intersection / union

        return overlap_matrix

    async def _generate_narrative_connection(self, email1: EmailNode, email2: EmailNode,
                                             similarity_score: float) -> str:
        """Generate narrative description of relationship using LLM."""
        if similarity_score < self.config.min_relationship_threshold:
            return "Weak connection"

        prompt = PromptTemplate(
            input_variables=["email1_subject", "email1_content", "email2_subject", "email2_content"],
            template="""
            Analyze the relationship between these two emails and provide a brief narrative description:

            Email 1:
            Subject: {email1_subject}
            Content: {email1_content}

            Email 2:
            Subject: {email2_subject}
            Content: {email2_content}

            Provide a concise description (1-2 sentences) of how these emails are related:
            """
        )

        try:
            formatted_prompt = prompt.format(
                email1_subject=email1.subject,
                email1_content=email1.content[:500],  # Limit content length
                email2_subject=email2.subject,
                email2_content=email2.content[:500]
            )

            result = await asyncio.to_thread(self.llm.invoke, formatted_prompt)
            return result.strip()
        except Exception as e:
            logger.error(f"Error generating narrative connection: {e}")
            return "Connection analysis unavailable"

    def extract_relationships(self, emails: List[EmailNode]) -> List[RelationshipEdge]:
        """Extract all relationships between emails."""
        if len(emails) < 2:
            return []

        # Calculate similarity matrices
        semantic_sim = self._calculate_semantic_similarity(emails)
        temporal_prox = self._calculate_temporal_proximity(emails)
        participant_overlap = self._calculate_participant_overlap(emails)
        entity_overlap = self._calculate_entity_overlap(emails)

        relationships = []
        n = len(emails)

        for i in range(n):
            for j in range(i + 1, n):  # Only consider unique pairs
                # Calculate composite relationship score
                semantic_score = semantic_sim[i][j]
                temporal_score = temporal_prox[i][j]
                participant_score = participant_overlap[i][j]
                entity_score = entity_overlap[i][j]

                composite_score = (
                        self.config.semantic_weight * semantic_score +
                        self.config.temporal_weight * temporal_score +
                        self.config.participant_weight * participant_score +
                        self.config.entity_weight * entity_score
                )

                # Only create relationships above threshold
                if composite_score >= self.config.min_relationship_threshold:
                    # Determine relationship type
                    relationship_type = self._classify_relationship_type(
                        semantic_score, temporal_score, participant_score
                    )

                    # Extract shared entities
                    shared_entities = self._extract_shared_entities(emails[i], emails[j])

                    relationship = RelationshipEdge(
                        source_id=emails[i].id,
                        target_id=emails[j].id,
                        weight=composite_score,
                        relationship_type=relationship_type,
                        semantic_similarity=semantic_score,
                        temporal_proximity=temporal_score,
                        participant_overlap=participant_score,
                        shared_entities=shared_entities,
                        narrative_connection=""  # Will be filled asynchronously if needed
                    )

                    relationships.append(relationship)

        return relationships

    def _classify_relationship_type(self, semantic: float, temporal: float,
                                    participant: float) -> str:
        """Classify the type of relationship based on similarity scores."""
        if participant > 0.7:
            return "Direct Conversation"
        elif semantic > 0.6:
            return "Thematic Connection"
        elif temporal > 0.8:
            return "Temporal Sequence"
        else:
            return "General Relationship"

    def _extract_shared_entities(self, email1: EmailNode, email2: EmailNode) -> List[str]:
        """Extract entities that appear in both emails."""
        entities1 = set()
        entities2 = set()

        for entity_list in email1.entities.values():
            entities1.update(entity_list)

        for entity_list in email2.entities.values():
            entities2.update(entity_list)

        return list(entities1 & entities2)


# =====================================================================
# Analysis Engine
# =====================================================================

class EmailAnalysisEngine:
    """Main engine for comprehensive email analysis."""

    def __init__(self, config: AnalysisConfig = None):
        self.config = config or AnalysisConfig()
        self.processor = EmailDataProcessor()
        self.extractor = RelationshipExtractor(self.config)
        self.emails: List[EmailNode] = []
        self.relationships: List[RelationshipEdge] = []
        self.network: nx.Graph = None

    def load_data(self, json_data: Dict) -> None:
        """Load and process email data from JSON."""
        try:
            emails_data = json_data.get('emails', [])
            self.emails = []

            for email_data in emails_data:
                processed_email = self.processor.process_email_data(email_data)
                self.emails.append(processed_email)

            logger.info(f"Loaded and processed {len(self.emails)} emails")
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise

    def analyze_relationships(self) -> None:
        """Perform comprehensive relationship analysis."""
        if not self.emails:
            raise ValueError("No emails loaded. Call load_data() first.")

        try:
            # Extract relationships
            self.relationships = self.extractor.extract_relationships(self.emails)

            # Build network graph
            self._build_network_graph()

            # Calculate additional metrics
            self._calculate_importance_scores()
            self._cluster_emails()

            logger.info(f"Analysis complete. Found {len(self.relationships)} relationships")
        except Exception as e:
            logger.error(f"Error during analysis: {e}")
            raise

    def _build_network_graph(self) -> None:
        """Build NetworkX graph from relationships."""
        self.network = nx.Graph()

        # Add nodes
        for email in self.emails:
            self.network.add_node(
                email.id,
                sender=email.sender,
                subject=email.subject,
                timestamp=email.timestamp,
                importance=email.importance_score
            )

        # Add edges
        for rel in self.relationships:
            self.network.add_edge(
                rel.source_id,
                rel.target_id,
                weight=rel.weight,
                relationship_type=rel.relationship_type
            )

    def _calculate_importance_scores(self) -> None:
        """Calculate importance scores using network centrality measures."""
        if not self.network:
            return

        try:
            # Calculate various centrality measures
            betweenness = nx.betweenness_centrality(self.network, weight='weight')
            closeness = nx.closeness_centrality(self.network, distance='weight')
            eigenvector = nx.eigenvector_centrality(self.network, weight='weight', max_iter=1000)

            # Combine centrality measures
            for email in self.emails:
                if email.id in self.network:
                    importance = (
                            0.4 * betweenness.get(email.id, 0) +
                            0.3 * closeness.get(email.id, 0) +
                            0.3 * eigenvector.get(email.id, 0)
                    )
                    email.importance_score = importance
        except Exception as e:
            logger.warning(f"Error calculating importance scores: {e}")

    def _cluster_emails(self) -> None:
        """Cluster emails by topic using content similarity."""
        if not self.emails or not self.extractor.email_vectors:
            return

        try:
            # Use K-means clustering on TF-IDF vectors
            kmeans = KMeans(n_clusters=min(self.config.cluster_count, len(self.emails)))
            clusters = kmeans.fit_predict(self.extractor.email_vectors)

            for i, email in enumerate(self.emails):
                email.topic_cluster = int(clusters[i])
        except Exception as e:
            logger.warning(f"Error clustering emails: {e}")

    def export_results(self) -> Dict:
        """Export analysis results as structured JSON."""
        return {
            "metadata": {
                "total_emails": len(self.emails),
                "total_relationships": len(self.relationships),
                "analysis_config": asdict(self.config),
                "timestamp": datetime.now().isoformat()
            },
            "nodes": [
                {
                    "id": email.id,
                    "sender": email.sender,
                    "recipients": email.recipients,
                    "subject": email.subject,
                    "content_preview": email.content[:200] + "..." if len(email.content) > 200 else email.content,
                    "timestamp": email.timestamp.isoformat(),
                    "importance_score": email.importance_score,
                    "topic_cluster": email.topic_cluster,
                    "entities": email.entities
                }
                for email in self.emails
            ],
            "edges": [
                {
                    "source": rel.source_id,
                    "target": rel.target_id,
                    "weight": rel.weight,
                    "type": rel.relationship_type,
                    "semantic_similarity": rel.semantic_similarity,
                    "temporal_proximity": rel.temporal_proximity,
                    "participant_overlap": rel.participant_overlap,
                    "shared_entities": rel.shared_entities,
                    "narrative_connection": rel.narrative_connection
                }
                for rel in self.relationships
            ],
            "network_metrics": self._calculate_network_metrics() if self.network else {}
        }

    def _calculate_network_metrics(self) -> Dict:
        """Calculate overall network metrics."""
        if not self.network:
            return {}

        try:
            return {
                "density": nx.density(self.network),
                "average_clustering": nx.average_clustering(self.network),
                "number_of_components": nx.number_connected_components(self.network),
                "average_path_length": nx.average_shortest_path_length(self.network) if nx.is_connected(
                    self.network) else None,
                "diameter": nx.diameter(self.network) if nx.is_connected(self.network) else None
            }
        except Exception as e:
            logger.warning(f"Error calculating network metrics: {e}")
            return {}


# =====================================================================
# Streamlit Visualization Application
# =====================================================================

class EmailVisualizationApp:
    """Streamlit application for interactive email relationship visualization."""

    def __init__(self):
        self.engine = None
        self.results = None

    def run(self):
        """Run the Streamlit application."""
        st.set_page_config(
            page_title="Email Relationship Analyzer",
            page_icon="📧",
            layout="wide",
            initial_sidebar_state="expanded"
        )

        st.title("📧 Email Relationship Analysis System")
        st.markdown("Analyze communication patterns and relationships in email data")

        # Sidebar configuration
        self._render_sidebar()

        # Main content
        uploaded_file = st.file_uploader(
            "Upload Email JSON Data",
            type=['json'],
            help="Upload a JSON file containing email data"
        )

        if uploaded_file is not None:
            self._process_uploaded_file(uploaded_file)

        if self.results:
            self._render_visualizations()

    def _render_sidebar(self):
        """Render the sidebar with configuration options."""
        st.sidebar.header("Analysis Configuration")

        # Analysis parameters
        semantic_weight = st.sidebar.slider("Semantic Weight", 0.0, 1.0, 0.4, 0.1)
        temporal_weight = st.sidebar.slider("Temporal Weight", 0.0, 1.0, 0.3, 0.1)
        participant_weight = st.sidebar.slider("Participant Weight", 0.0, 1.0, 0.2, 0.1)
        entity_weight = st.sidebar.slider("Entity Weight", 0.0, 1.0, 0.1, 0.1)

        min_threshold = st.sidebar.slider("Min Relationship Threshold", 0.0, 1.0, 0.3, 0.05)
        max_temporal_days = st.sidebar.slider("Max Temporal Days", 1, 365, 30, 1)
        cluster_count = st.sidebar.slider("Number of Clusters", 2, 20, 10, 1)

        # Store configuration in session state
        st.session_state.analysis_config = AnalysisConfig(
            semantic_weight=semantic_weight,
            temporal_weight=temporal_weight,
            participant_weight=participant_weight,
            entity_weight=entity_weight,
            min_relationship_threshold=min_threshold,
            max_temporal_days=max_temporal_days,
            cluster_count=cluster_count
        )

    def _process_uploaded_file(self, uploaded_file):
        """Process the uploaded JSON file."""
        try:
            # Load JSON data
            json_data = json.load(uploaded_file)

            # Initialize engine with configuration
            config = getattr(st.session_state, 'analysis_config', AnalysisConfig())
            self.engine = EmailAnalysisEngine(config)

            # Process data
            with st.spinner("Processing email data..."):
                self.engine.load_data(json_data)

            # Perform analysis
            with st.spinner("Analyzing relationships..."):
                self.engine.analyze_relationships()

            # Export results
            self.results = self.engine.export_results()

            st.success(
                f"Analysis complete! Processed {len(self.engine.emails)} emails and found {len(self.engine.relationships)} relationships.")

        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

    def _render_visualizations(self):
        """Render the main visualization components."""
        if not self.results:
            return

        # Create tabs for different visualizations
        tab1, tab2, tab3, tab4 = st.tabs(["Network Graph", "Heatmap", "Timeline", "Analytics"])

        with tab1:
            self._render_network_graph()

        with tab2:
            self._render_relationship_heatmap()

        with tab3:
            self._render_timeline_view()

        with tab4:
            self._render_analytics_dashboard()

    def _render_network_graph(self):
        """Render interactive network graph visualization."""
        st.subheader("Email Relationship Network")

        # Filter controls
        col1, col2, col3 = st.columns(3)

        with col1:
            min_weight = st.slider("Minimum Relationship Weight", 0.0, 1.0, 0.3, 0.05)

        with col2:
            relationship_types = list(set(edge['type'] for edge in self.results['edges']))
            selected_types = st.multiselect("Relationship Types", relationship_types, relationship_types)

        with col3:
            node_size_metric = st.selectbox("Node Size Based On", ["importance_score", "topic_cluster", "uniform"])

        # Filter data
        filtered_edges = [
            edge for edge in self.results['edges']
            if edge['weight'] >= min_weight and edge['type'] in selected_types
        ]

        # Create network visualization
        if filtered_edges:
            fig = self._create_network_figure(filtered_edges, node_size_metric)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No relationships match the current filters.")

    def _create_network_figure(self, edges, node_size_metric):
        """Create interactive network graph using Plotly."""
        # Build NetworkX graph from filtered edges
        G = nx.Graph()

        # Add nodes
        node_dict = {node['id']: node for node in self.results['nodes']}
        for edge in edges:
            if edge['source'] not in G:
                G.add_node(edge['source'], **node_dict[edge['source']])
            if edge['target'] not in G:
                G.add_node(edge['target'], **node_dict[edge['target']])
            G.add_edge(edge['source'], edge['target'], **edge)

        # Calculate layout
        pos = nx.spring_layout(G, k=3, iterations=50)

        # Create edge traces
        edge_trace = []
        for edge in G.edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]

            edge_trace.append(
                go.Scatter(
                    x=[x0, x1, None],
                    y=[y0, y1, None],
                    mode='lines',
                    line=dict(width=edge[2]['weight'] * 3, color='rgba(125,125,125,0.5)'),
                    hoverinfo='none',
                    showlegend=False
                )
            )

        # Create node trace
        node_x = []
        node_y = []
        node_text = []
        node_size = []
        node_color = []

        for node_id in G.nodes():
            x, y = pos[node_id]
            node_x.append(x)
            node_y.append(y)

            node_data = G.nodes[node_id]
            node_text.append(f"Subject: {node_data['subject']}<br>Sender: {node_data['sender']}")

            # Set node size based on selected metric
            if node_size_metric == "importance_score":
                node_size.append(max(10, node_data['importance_score'] * 50))
            elif node_size_metric == "topic_cluster":
                node_size.append(15)
            else:
                node_size.append(15)

            node_color.append(node_data.get('topic_cluster', 0))

        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode='markers',
            hoverinfo='text',
            text=node_text,
            marker=dict(
                size=node_size,
                color=node_color,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Topic Cluster")
            ),
            showlegend=False
        )

        # Create figure
        fig = go.Figure(data=[*edge_trace, node_trace])
        fig.update_layout(
            title="Email Relationship Network",
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            annotations=[
                dict(
                    text="Node size represents importance, color represents topic cluster",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.002,
                    xanchor='left', yanchor='bottom',
                    font=dict(size=12)
                )
            ],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )

        return fig

    def _render_relationship_heatmap(self):
        """Render relationship strength heatmap."""
        st.subheader("Relationship Strength Heatmap")

        # Create similarity matrix
        email_ids = [node['id'] for node in self.results['nodes']]
        n = len(email_ids)
        similarity_matrix = np.zeros((n, n))

        id_to_index = {email_id: i for i, email_id in enumerate(email_ids)}

        for edge in self.results['edges']:
            i = id_to_index[edge['source']]
            j = id_to_index[edge['target']]
            similarity_matrix[i][j] = edge['weight']
            similarity_matrix[j][i] = edge['weight']  # Make symmetric

        # Create heatmap
        fig = px.imshow(
            similarity_matrix,
            labels=dict(x="Email ID", y="Email ID", color="Relationship Strength"),
            x=[f"Email {i + 1}" for i in range(n)],
            y=[f"Email {i + 1}" for i in range(n)],
            title="Email Relationship Strength Matrix"
        )

        fig.update_layout(
            width=800,
            height=600,
            title_x=0.5
        )

        st.plotly_chart(fig, use_container_width=True)

        # Show relationship statistics
        st.subheader("Relationship Statistics")
        col1, col2, col3 = st.columns(3)

        with col1:
            avg_strength = np.mean(similarity_matrix[similarity_matrix > 0])
            st.metric("Average Relationship Strength", f"{avg_strength:.3f}")

        with col2:
            max_strength = np.max(similarity_matrix)
            st.metric("Strongest Relationship", f"{max_strength:.3f}")

        with col3:
            total_relationships = np.sum(similarity_matrix > 0) // 2  # Divide by 2 for symmetric matrix
            st.metric("Total Relationships", total_relationships)

    def _render_timeline_view(self):
        """Render timeline visualization of email communications."""
        st.subheader("Email Communication Timeline")

        # Prepare timeline data
        timeline_data = []
        for node in self.results['nodes']:
            timeline_data.append({
                'email_id': node['id'],
                'timestamp': pd.to_datetime(node['timestamp']),
                'sender': node['sender'],
                'subject': node['subject'],
                'importance': node['importance_score'],
                'cluster': node['topic_cluster']
            })

        df = pd.DataFrame(timeline_data)
        df = df.sort_values('timestamp')

        # Time range selector
        min_date = df['timestamp'].min().date()
        max_date = df['timestamp'].max().date()

        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", min_date, min_value=min_date, max_value=max_date)
        with col2:
            end_date = st.date_input("End Date", max_date, min_value=min_date, max_value=max_date)

        # Filter data by date range
        mask = (df['timestamp'].dt.date >= start_date) & (df['timestamp'].dt.date <= end_date)
        filtered_df = df[mask]

        if not filtered_df.empty:
            # Create timeline scatter plot
            fig = px.scatter(
                filtered_df,
                x='timestamp',
                y='sender',
                size='importance',
                color='cluster',
                hover_data=['subject'],
                title="Email Timeline by Sender",
                labels={'timestamp': 'Date/Time', 'sender': 'Sender'}
            )

            fig.update_layout(
                height=600,
                showlegend=True
            )

            st.plotly_chart(fig, use_container_width=True)

            # Email volume over time
            st.subheader("Email Volume Over Time")

            # Group by date
            daily_counts = filtered_df.groupby(filtered_df['timestamp'].dt.date).size().reset_index()
            daily_counts.columns = ['date', 'count']

            fig_volume = px.bar(
                daily_counts,
                x='date',
                y='count',
                title="Daily Email Volume",
                labels={'date': 'Date', 'count': 'Number of Emails'}
            )

            st.plotly_chart(fig_volume, use_container_width=True)
        else:
            st.warning("No emails found in the selected date range.")

    def _render_analytics_dashboard(self):
        """Render comprehensive analytics dashboard."""
        st.subheader("Email Analysis Dashboard")

        # Network metrics
        if 'network_metrics' in self.results and self.results['network_metrics']:
            st.subheader("Network Metrics")

            metrics = self.results['network_metrics']
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                density = metrics.get('density', 0)
                st.metric("Network Density", f"{density:.3f}")

            with col2:
                clustering = metrics.get('average_clustering', 0)
                st.metric("Average Clustering", f"{clustering:.3f}")

            with col3:
                components = metrics.get('number_of_components', 0)
                st.metric("Connected Components", components)

            with col4:
                avg_path = metrics.get('average_path_length')
                if avg_path:
                    st.metric("Avg Path Length", f"{avg_path:.2f}")
                else:
                    st.metric("Avg Path Length", "N/A")

        # Participant analysis
        st.subheader("Participant Analysis")

        # Extract all participants
        participants = {}
        for node in self.results['nodes']:
            sender = node['sender']
            if sender not in participants:
                participants[sender] = {'sent': 0, 'received': 0, 'importance': 0}
            participants[sender]['sent'] += 1
            participants[sender]['importance'] += node['importance_score']

            for recipient in node['recipients']:
                if recipient not in participants:
                    participants[recipient] = {'sent': 0, 'received': 0, 'importance': 0}
                participants[recipient]['received'] += 1

        # Create participant dataframe
        participant_data = []
        for email, stats in participants.items():
            participant_data.append({
                'participant': email,
                'emails_sent': stats['sent'],
                'emails_received': stats['received'],
                'total_emails': stats['sent'] + stats['received'],
                'avg_importance': stats['importance'] / max(stats['sent'], 1)
            })

        participant_df = pd.DataFrame(participant_data)
        participant_df = participant_df.sort_values('total_emails', ascending=False)

        # Top participants chart
        top_participants = participant_df.head(10)

        fig_participants = px.bar(
            top_participants,
            x='participant',
            y=['emails_sent', 'emails_received'],
            title="Top 10 Most Active Participants",
            labels={'value': 'Number of Emails', 'participant': 'Participant'}
        )

        fig_participants.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_participants, use_container_width=True)

        # Topic cluster analysis
        st.subheader("Topic Cluster Analysis")

        cluster_data = {}
        for node in self.results['nodes']:
            cluster = node['topic_cluster']
            if cluster not in cluster_data:
                cluster_data[cluster] = {'count': 0, 'subjects': []}
            cluster_data[cluster]['count'] += 1
            cluster_data[cluster]['subjects'].append(node['subject'])

        # Cluster distribution
        cluster_df = pd.DataFrame([
            {'cluster': k, 'count': v['count']}
            for k, v in cluster_data.items()
        ])

        fig_clusters = px.pie(
            cluster_df,
            values='count',
            names='cluster',
            title="Email Distribution by Topic Cluster"
        )

        st.plotly_chart(fig_clusters, use_container_width=True)

        # Relationship type analysis
        st.subheader("Relationship Type Distribution")

        relationship_types = {}
        for edge in self.results['edges']:
            rel_type = edge['type']
            if rel_type not in relationship_types:
                relationship_types[rel_type] = 0
            relationship_types[rel_type] += 1

        rel_type_df = pd.DataFrame([
            {'type': k, 'count': v}
            for k, v in relationship_types.items()
        ])

        fig_rel_types = px.bar(
            rel_type_df,
            x='type',
            y='count',
            title="Distribution of Relationship Types"
        )

        st.plotly_chart(fig_rel_types, use_container_width=True)

        # Export functionality
        st.subheader("Export Analysis Results")

        if st.button("Generate Downloadable Report"):
            # Create downloadable JSON report
            report_json = json.dumps(self.results, indent=2, default=str)
            st.download_button(
                label="Download Analysis Report (JSON)",
                data=report_json,
                file_name=f"email_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )

        # Raw data tables
        with st.expander("View Raw Data Tables"):
            st.subheader("Email Nodes")
            nodes_df = pd.DataFrame(self.results['nodes'])
            st.dataframe(nodes_df)

            st.subheader("Relationships")
            edges_df = pd.DataFrame(self.results['edges'])
            st.dataframe(edges_df)


# =====================================================================
# Command Line Interface
# =====================================================================

def main():
    """Main function for command-line usage."""
    import argparse

    parser = argparse.ArgumentParser(description="Email Relationship Analysis System")
    parser.add_argument("--input", "-i", required=True, help="Input JSON file path")
    parser.add_argument("--output", "-o", help="Output JSON file path")
    parser.add_argument("--web", "-w", action="store_true", help="Launch web interface")

    args = parser.parse_args()

    if args.web:
        # Launch Streamlit app
        app = EmailVisualizationApp()
        app.run()
    else:
        # Command line processing
        try:
            # Load data
            with open(args.input, 'r') as f:
                json_data = json.load(f)

            # Initialize and run analysis
            engine = EmailAnalysisEngine()
            engine.load_data(json_data)
            engine.analyze_relationships()

            # Export results
            results = engine.export_results()

            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(results, f, indent=2, default=str)
                print(f"Analysis results saved to {args.output}")
            else:
                print(json.dumps(results, indent=2, default=str))

        except Exception as e:
            logger.error(f"Error during analysis: {e}")
            return 1

    return 0


# =====================================================================
# Entry Point
# =====================================================================

if __name__ == "__main__":
    # Check if running in Streamlit
    try:
        import streamlit.runtime.scriptrunner.script_run_context as script_run_context

        if script_run_context.get_script_run_ctx() is not None:
            # Running in Streamlit
            app = EmailVisualizationApp()
            app.run()
        else:
            # Running from command line
            exit(main())
    except ImportError:
        # Streamlit not available, use command line
        exit(main())


# =====================================================================
# Additional Utility Functions
# =====================================================================

def validate_json_structure(json_data: Dict) -> bool:
    """Validate that the input JSON has the expected structure."""
    required_keys = ['emails']

    if not all(key in json_data for key in required_keys):
        return False

    emails = json_data.get('emails', [])
    if not isinstance(emails, list):
        return False

    # Check that each email has required fields
    required_email_keys = ['email_id', 'from', 'to', 'date', 'subject']
    for email in emails[:5]:  # Check first 5 emails
        if not all(key in email for key in required_email_keys):
            return False

    return True


def preprocess_enron_data(json_data: Dict) -> Dict:
    """Preprocess Enron-specific data format."""
    # Handle the specific structure of your Enron data
    if 'emails' in json_data:
        # Data is already in the expected format
        return json_data

    # If data needs restructuring, handle it here
    # This is a placeholder for any specific preprocessing needed
    return json_data


# =====================================================================
# Configuration and Constants
# =====================================================================

# Default configuration
DEFAULT_CONFIG = AnalysisConfig()

# Color schemes for visualizations
COLOR_SCHEMES = {
    'default': px.colors.qualitative.Set3,
    'professional': px.colors.qualitative.Safe,
    'vibrant': px.colors.qualitative.Vivid
}

# Logging configuration
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        },
    },
    'handlers': {
        'default': {
            'level': 'INFO',
            'formatter': 'standard',
            'class': 'logging.StreamHandler',
        },
    },
    'loggers': {
        '': {
            'handlers': ['default'],
            'level': 'INFO',
            'propagate': False
        }
    }
}