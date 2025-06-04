import streamlit as st
import json
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from collections import defaultdict, Counter
import numpy as np
import os
import sys

# Add the parent directory to the path to import config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.config.config import REFINED_JSON_PATH, CLEANED_JSON_PATH, EXTRACTED_ENTITIES_JSON_PATH

# Configure Streamlit page
st.set_page_config(
    page_title="Email Network Analysis",
    page_icon="📧",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📧 Email Network Analysis Dashboard")
st.markdown("---")


@st.cache_data
def load_json_data(file_path):
    """Load and parse JSON data from file path"""
    try:
        if not os.path.exists(file_path):
            st.error(f"File not found: {file_path}")
            return None

        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        st.error(f"Error loading JSON file: {str(e)}")
        return None


def create_email_communication_graph(data):
    """Create network graph based on email communications (From -> To)"""
    G = nx.DiGraph()  # Directed graph for email flow

    for email in data:
        from_email = email.get('From', '')
        to_emails = email.get('To', '').split(',')  # Handle multiple recipients

        if from_email:
            # Add sender node
            G.add_node(from_email, node_type='email', color='#DC143C')

            # Add recipient nodes and edges
            for to_email in to_emails:
                to_email = to_email.strip()
                if to_email and to_email != from_email:
                    G.add_node(to_email, node_type='email', color='#DC143C')

                    # Add edge (or increase weight if exists)
                    if G.has_edge(from_email, to_email):
                        G[from_email][to_email]['weight'] += 1
                    else:
                        G.add_edge(from_email, to_email, weight=1)

    return G


def create_entity_network_graph(data, entity_types=None):
    """Create network graph based on entity co-occurrences"""
    G = nx.Graph()

    # Color mapping for different entity types
    entity_colors = {
        'PERSON': '#FF6B6B',
        'ORG': '#4ECDC4',
        'GPE': '#45B7D1',  # Geopolitical entity
        'DATE': '#96CEB4',
        'TIME': '#FFEAA7',
        'LAW': '#DDA0DD',
        'QUANTITY': '#F4A460',
        'PERCENT': '#FFB347',
        'CARDINAL': '#98D8C8'
    }

    if entity_types is None:
        entity_types = list(entity_colors.keys())

    for email in data:
        entities = email.get('entities', [])

        # Filter entities by selected types
        filtered_entities = [e for e in entities if e['label'] in entity_types]

        # Add nodes for entities
        for entity in filtered_entities:
            value = entity['value']
            label = entity['label']
            color = entity_colors.get(label, '#888888')

            G.add_node(value,
                       node_type='entity',
                       entity_label=label,
                       color=color,
                       size=10)

        # Create edges between entities that appear in the same email
        for i, entity1 in enumerate(filtered_entities):
            for entity2 in filtered_entities[i + 1:]:
                val1, val2 = entity1['value'], entity2['value']
                if val1 != val2:
                    if G.has_edge(val1, val2):
                        G[val1][val2]['weight'] += 1
                    else:
                        G.add_edge(val1, val2, weight=1)

    return G


def create_plotly_network(G, title="Network Graph"):
    """Create interactive Plotly network visualization"""

    if len(G.nodes()) == 0:
        return go.Figure().add_annotation(
            text="No data to display",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )

    # Calculate layout
    pos = nx.spring_layout(G, k=1, iterations=50, seed=42)

    # Prepare edge traces
    edge_x, edge_y = [], []
    edge_info = []

    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

        weight = G[edge[0]][edge[1]].get('weight', 1)
        edge_info.append(f"{edge[0]} → {edge[1]}<br>Weight: {weight}")

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color='#888888'),
        hoverinfo='none',
        mode='lines',
        showlegend=False
    )

    # Prepare node traces
    node_x, node_y = [], []
    node_text, node_info = [], []
    node_colors, node_sizes = [], []

    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

        # Node info
        node_data = G.nodes[node]
        node_type = node_data.get('node_type', 'unknown')

        if node_type == 'email':
            node_text.append(node.split('@')[0])  # Show username
            connections = len(list(G.neighbors(node)))
            node_info.append(f"Email: {node}<br>Connections: {connections}")
            node_colors.append('#DC143C')
            node_sizes.append(15)
        else:  # entity
            entity_label = node_data.get('entity_label', 'Unknown')
            node_text.append(f"{node[:20]}..." if len(node) > 20 else node)
            connections = len(list(G.neighbors(node)))
            node_info.append(f"Entity: {node}<br>Type: {entity_label}<br>Connections: {connections}")
            node_colors.append(node_data.get('color', '#888888'))
            node_sizes.append(12)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hovertemplate='%{hovertext}<extra></extra>',
        hovertext=node_info,
        text=node_text,
        textposition="middle center",
        textfont=dict(size=8, color='white'),
        marker=dict(
            size=node_sizes,
            color=node_colors,
            symbol='square',
            line=dict(width=1, color='white')
        ),
        showlegend=False
    )

    # Create figure
    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        title=title,
        titlefont_size=16,
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20, l=5, r=5, t=40),
        annotations=[
            dict(
                text="Hover over nodes for details",
                showarrow=False,
                xref="paper", yref="paper",
                x=0.005, y=-0.002
            )
        ],
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor='white',
        height=600
    )

    return fig


def analyze_network_stats(G, network_type="Network"):
    """Calculate and display network statistics"""
    if len(G.nodes()) == 0:
        return

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Nodes", len(G.nodes()))

    with col2:
        st.metric("Edges", len(G.edges()))

    with col3:
        if G.is_directed():
            avg_degree = np.mean([d for n, d in G.in_degree()] + [d for n, d in G.out_degree()])
        else:
            avg_degree = np.mean([d for n, d in G.degree()])
        st.metric("Avg Degree", f"{avg_degree:.2f}")

    with col4:
        try:
            density = nx.density(G)
            st.metric("Density", f"{density:.3f}")
        except:
            st.metric("Density", "N/A")


def main():
    # Sidebar for data source and options
    with st.sidebar:
        st.header("⚙️ Configuration")

        # Data source selection
        st.subheader("📁 Data Source")
        data_source = st.selectbox(
            "Choose data file:",
            [
                ("Extracted Entities", EXTRACTED_ENTITIES_JSON_PATH),
                ("Refined Enron", REFINED_JSON_PATH),
                ("Cleaned Enron", CLEANED_JSON_PATH)
            ],
            format_func=lambda x: x[0],
            help="Select which JSON file to load for analysis"
        )

        # Load selected data
        selected_file = data_source[1]
        st.info(f"Loading: {os.path.basename(selected_file)}")

        data = load_json_data(selected_file)

        if data:
            st.success(f"✅ Loaded {len(data)} records")

            # Network type selection
            st.subheader("🌐 Network Type")
            network_type = st.radio(
                "Choose visualization:",
                ["Email Communications", "Entity Networks", "Both"]
            )

            # Entity type selection for entity networks
            if network_type in ["Entity Networks", "Both"]:
                st.subheader("🏷️ Entity Types")

                # Get all entity types from data
                all_entity_types = set()
                for record in data:
                    for entity in record.get('entities', []):
                        all_entity_types.add(entity['label'])

                all_entity_types = sorted(list(all_entity_types))

                if all_entity_types:
                    selected_entity_types = st.multiselect(
                        "Select entity types:",
                        all_entity_types,
                        default=all_entity_types[:5] if len(all_entity_types) > 5 else all_entity_types,
                        help="Choose which entity types to include in the network"
                    )
                else:
                    st.warning("No entities found in the data")
                    selected_entity_types = []

            # Analysis options
            st.subheader("📊 Analysis Options")
            show_stats = st.checkbox("Show Network Statistics", value=True)
            show_top_nodes = st.checkbox("Show Top Connected Nodes", value=True)

            # Data file info
            st.subheader("📋 File Info")
            file_size = os.path.getsize(selected_file) / (1024 * 1024)  # MB
            st.write(f"File size: {file_size:.2f} MB")
            st.write(f"Records: {len(data):,}")
        else:
            st.error("❌ Failed to load data")
            return

    # Main content area
    if data:

        # Email Communications Network
        if network_type in ["Email Communications", "Both"]:
            st.header("📨 Email Communication Network")

            email_graph = create_email_communication_graph(data)

            if len(email_graph.nodes()) == 0:
                st.warning("No email communication data found in the selected file.")
            else:
                if show_stats:
                    analyze_network_stats(email_graph, "Email Communication")

                fig_email = create_plotly_network(email_graph, "Email Communication Network")
                st.plotly_chart(fig_email, use_container_width=True)

                if show_top_nodes and len(email_graph.nodes()) > 0:
                    st.subheader("📊 Top Connected Email Addresses")

                    # Calculate centrality metrics
                    if email_graph.is_directed():
                        in_degree = dict(email_graph.in_degree())
                        out_degree = dict(email_graph.out_degree())

                        col1, col2 = st.columns(2)

                        with col1:
                            st.write("**Most emails received:**")
                            top_receivers = sorted(in_degree.items(), key=lambda x: x[1], reverse=True)[:5]
                            for email, count in top_receivers:
                                st.write(f"• {email}: {count} emails")

                        with col2:
                            st.write("**Most emails sent:**")
                            top_senders = sorted(out_degree.items(), key=lambda x: x[1], reverse=True)[:5]
                            for email, count in top_senders:
                                st.write(f"• {email}: {count} emails")

                    else:
                        degree = dict(email_graph.degree())
                        top_connected = sorted(degree.items(), key=lambda x: x[1], reverse=True)[:10]
                        for email, connections in top_connected:
                            st.write(f"• {email}: {connections} connections")

        # Entity Networks
        if network_type in ["Entity Networks", "Both"]:
            if network_type == "Both":
                st.markdown("---")

            st.header("🏷️ Entity Co-occurrence Network")

            entity_graph = create_entity_network_graph(data, selected_entity_types)

            if len(entity_graph.nodes()) == 0:
                st.warning("No entity data found for the selected types.")
            else:
                if show_stats:
                    analyze_network_stats(entity_graph, "Entity")

                fig_entity = create_plotly_network(entity_graph, "Entity Co-occurrence Network")
                st.plotly_chart(fig_entity, use_container_width=True)

                if show_top_nodes and len(entity_graph.nodes()) > 0:
                    st.subheader("📊 Most Connected Entities")

                    degree = dict(entity_graph.degree())
                    top_entities = sorted(degree.items(), key=lambda x: x[1], reverse=True)[:10]

                    for entity, connections in top_entities:
                        # Get entity type
                        entity_data = entity_graph.nodes[entity]
                        entity_type = entity_data.get('entity_label', 'Unknown')
                        st.write(f"• **{entity}** ({entity_type}): {connections} connections")

        # Data summary
        st.markdown("---")
        st.header("📋 Data Summary")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Email Statistics")
            total_records = len(data)
            unique_senders = len(set(record.get('From', '') for record in data if record.get('From')))
            unique_recipients = len(set(
                recipient.strip()
                for record in data
                for recipient in record.get('To', '').split(',')
                if recipient.strip()
            ))

            st.write(f"• Total records: {total_records:,}")
            st.write(f"• Unique senders: {unique_senders:,}")
            st.write(f"• Unique recipients: {unique_recipients:,}")

        with col2:
            st.subheader("Entity Statistics")

            entity_counts = Counter()
            for record in data:
                for entity in record.get('entities', []):
                    entity_counts[entity['label']] += 1

            total_entities = sum(entity_counts.values())
            st.write(f"• Total entities: {total_entities:,}")

            for entity_type, count in entity_counts.most_common():
                st.write(f"• {entity_type}: {count:,}")

        # Dataset source information
        st.markdown("---")
        st.info(f"📂 Currently analyzing: **{os.path.basename(selected_file)}**")


if __name__ == "__main__":
    main()
