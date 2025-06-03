import sys
import pandas as pd
import streamlit as st
import json
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict, Counter
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Import config for file paths
try:
    from src.config.config import EXTRACTED_ENTITIES_JSON_PATH, BASE_DIR
except ImportError:
    st.error("Config file not found. Please ensure config.py is in your project directory.")
    st.stop()


def load_entities_from_json(file_path):
    """Load entities from JSON file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        st.error(f"File {file_path} not found. Please check if the entities extraction has been completed.")
        return None
    except json.JSONDecodeError:
        st.error("Invalid JSON format. Please check your entities JSON file.")
        return None


def create_network_from_entities(entities_data):
    """
    Create a network graph from entities data.
    Handles the specific format: {"entities": [{"value": "entity_name", "label": "entity_type"}, ...]}
    """
    G = nx.Graph()

    # Handle the specific format from the Enron entities file
    if isinstance(entities_data, dict) and 'entities' in entities_data:
        entities_list = entities_data['entities']

        # Group entities by type for better visualization
        entities_by_type = defaultdict(list)

        # First pass: Add all nodes
        for entity in entities_list:
            if isinstance(entity, dict):
                entity_value = entity.get('value', '')
                entity_label = entity.get('label', 'UNKNOWN')

                if entity_value:
                    G.add_node(entity_value, type=entity_label)
                    entities_by_type[entity_label].append(entity_value)

        # Second pass: Create meaningful connections
        # Strategy 1: Connect entities that might be related (same type clustering)
        for entity_type, entity_list in entities_by_type.items():
            if entity_type in ['PERSON', 'ORG']:  # Connect people and organizations
                # Create a hub-like structure for important entity types
                if len(entity_list) > 1:
                    # Connect each entity to the first one (creating a star pattern)
                    hub = entity_list[0]
                    for entity in entity_list[1:]:
                        G.add_edge(hub, entity, relation_type='same_type')

        # Strategy 2: Create cross-type connections for likely relationships
        people = entities_by_type.get('PERSON', [])
        organizations = entities_by_type.get('ORG', [])
        places = entities_by_type.get('GPE', [])

        # Connect people to organizations (likely work relationships)
        for person in people[:10]:  # Limit to avoid too many connections
            for org in organizations[:5]:
                if 'Enron' in org or 'Energy' in org or 'FERC' in org or 'CPUC' in org:
                    G.add_edge(person, org, relation_type='works_with')
                    break  # Only connect to one main org per person

        # Connect organizations to places (geographical relationships)
        for org in organizations:
            for place in places:
                if place in ['California', 'El Paso'] and len(list(G.neighbors(org))) < 3:
                    G.add_edge(org, place, relation_type='located_in')
                    break

        # Strategy 3: If still no connections, create a minimal connected graph
        if G.number_of_edges() == 0 and G.number_of_nodes() > 1:
            # Create connections between different entity types to form a connected network
            all_nodes = list(G.nodes())
            for i in range(len(all_nodes) - 1):
                if len(list(G.neighbors(all_nodes[i]))) == 0:
                    G.add_edge(all_nodes[i], all_nodes[i + 1], relation_type='co_occurrence')

    else:
        # Fallback for other formats
        st.warning(
            "Warning: Entities format not recognized. Expected format: {'entities': [{'value': '...', 'label': '...'}, ...]}")

    return G


def create_plotly_network(G, layout_type='spring'):
    """Create an interactive Plotly network visualization"""
    if len(G.nodes()) == 0:
        return None

    # Calculate positions
    if layout_type == 'spring':
        pos = nx.spring_layout(G, k=3, iterations=50)
    elif layout_type == 'circular':
        pos = nx.circular_layout(G)
    elif layout_type == 'kamada_kawai':
        pos = nx.kamada_kawai_layout(G)
    else:
        pos = nx.fruchterman_reingold_layout(G, k=2)

    # Get node information
    node_types = [G.nodes[node].get('type', 'UNKNOWN') for node in G.nodes()]
    unique_types = list(set(node_types))

    # Create color mapping
    colors = px.colors.qualitative.Set3[:len(unique_types)]
    if len(unique_types) > len(colors):
        colors = px.colors.qualitative.Plotly[:len(unique_types)]
    type_to_color = dict(zip(unique_types, colors))

    # Prepare edge traces
    edge_x = []
    edge_y = []
    edge_info = []

    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

        relation = G.edges[edge].get('relation_type', 'connected')
        edge_info.append(f"{edge[0]} - {edge[1]} ({relation})")

    edge_trace = go.Scatter(x=edge_x, y=edge_y,
                            line=dict(width=1, color='#888'),
                            hoverinfo='none',
                            mode='lines')

    # Prepare node traces by type
    traces = [edge_trace]

    for entity_type in unique_types:
        # Filter nodes by type
        nodes_of_type = [node for node in G.nodes() if G.nodes[node].get('type') == entity_type]

        node_x = [pos[node][0] for node in nodes_of_type]
        node_y = [pos[node][1] for node in nodes_of_type]

        # Calculate node sizes based on degree
        degrees = dict(G.degree())
        node_sizes = [max(10, degrees[node] * 5 + 15) for node in nodes_of_type]

        # Node hover info
        node_info = []
        for node in nodes_of_type:
            connections = list(G.neighbors(node))
            hover_text = f"<b>{node}</b><br>"
            hover_text += f"Type: {entity_type}<br>"
            hover_text += f"Connections: {len(connections)}<br>"
            if connections:
                hover_text += f"Connected to: {', '.join(connections[:3])}"
                if len(connections) > 3:
                    hover_text += f" and {len(connections) - 3} more..."
            node_info.append(hover_text)

        node_trace = go.Scatter(x=node_x, y=node_y,
                                mode='markers+text',
                                hoverinfo='text',
                                hovertext=node_info,
                                text=nodes_of_type,
                                textposition="middle center",
                                textfont=dict(size=8, color="white"),
                                name=entity_type,
                                marker=dict(size=node_sizes,
                                            color=type_to_color[entity_type],
                                            line=dict(width=2, color='white')))
        traces.append(node_trace)

    return traces


def display_network_analysis(G):
    """Display network analysis in Streamlit"""
    st.subheader("📊 Network Analysis")

    if G.number_of_nodes() == 0:
        st.warning("No nodes found in the network.")
        return

    # Basic stats in columns
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Nodes", G.number_of_nodes())
    with col2:
        st.metric("Total Edges", G.number_of_edges())
    with col3:
        st.metric("Density", f"{nx.density(G):.3f}")
    with col4:
        components = list(nx.connected_components(G))
        st.metric("Components", len(components))

    # Entity type distribution
    st.subheader("Entity Type Distribution")
    node_types = [G.nodes[node].get('type', 'UNKNOWN') for node in G.nodes()]
    type_counts = Counter(node_types)

    # Create a bar chart
    fig_bar = px.bar(x=list(type_counts.keys()),
                     y=list(type_counts.values()),
                     title="Number of Entities by Type",
                     labels={'x': 'Entity Type', 'y': 'Count'})
    st.plotly_chart(fig_bar, use_container_width=True)

    # Most connected entities
    st.subheader("🔗 Most Connected Entities")
    degrees = dict(G.degree())
    top_entities = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:10]

    top_df_data = []
    for entity, degree in top_entities:
        entity_type = G.nodes[entity].get('type', 'UNKNOWN')
        top_df_data.append({
            'Entity': entity,
            'Type': entity_type,
            'Connections': degree
        })

    if top_df_data:
        import pandas as pd
        top_df = pd.DataFrame(top_df_data)
        st.dataframe(top_df, use_container_width=True)

    # Relationship types
    edge_relations = [G.edges[edge].get('relation_type', 'unspecified') for edge in G.edges()]
    if edge_relations:
        st.subheader("Relationship Types")
        relation_counts = Counter(edge_relations)

        fig_pie = px.pie(values=list(relation_counts.values()),
                         names=list(relation_counts.keys()),
                         title="Distribution of Relationship Types")
        st.plotly_chart(fig_pie, use_container_width=True)


def save_network_analysis(G):
    """Save network analysis and graph"""
    output_path = os.path.join(os.path.dirname(EXTRACTED_ENTITIES_JSON_PATH), "enron_entity_network.gexf")
    nx.write_gexf(G, output_path)
    return output_path


def main():
    st.set_page_config(page_title="Entity Network Graph",
                       page_icon="🕸️",
                       layout="wide")

    st.title("🕸️ Entity Network Graph Analyzer")
    st.markdown("### Visualizing Entity Relationships from Enron Email Data")

    # Sidebar for controls
    st.sidebar.title("⚙️ Controls")

    # Layout selection
    layout_type = st.sidebar.selectbox(
        "Choose Layout Algorithm:",
        options=['spring', 'fruchterman_reingold', 'circular', 'kamada_kawai'],
        index=0,
        help="Different algorithms arrange nodes differently"
    )

    # Load data
    with st.spinner("Loading entity data..."):
        entities_data = load_entities_from_json(EXTRACTED_ENTITIES_JSON_PATH)

    if entities_data is None:
        st.stop()

    # Create network
    with st.spinner("Building network graph..."):
        G = create_network_from_entities(entities_data)

    if G.number_of_nodes() == 0:
        st.error("No entities found in the data. Please check your data format.")
        st.stop()

    # Display analysis
    display_network_analysis(G)

    # Network visualization
    st.subheader("🔍 Interactive Network Visualization")

    with st.spinner("Creating interactive visualization..."):
        traces = create_plotly_network(G, layout_type)

    if traces:
        fig = go.Figure(data=traces,
                        layout=go.Layout(
                            title="Entity Network Graph",
                            titlefont_size=16,
                            showlegend=True,
                            hovermode='closest',
                            margin=dict(b=20, l=5, r=5, t=40),
                            annotations=[dict(
                                text="Hover over nodes for details. Click and drag to explore.",
                                showarrow=False,
                                xref="paper", yref="paper",
                                x=0.005, y=-0.002,
                                xanchor='left', yanchor='bottom',
                                font=dict(color="#888", size=12)
                            )],
                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            plot_bgcolor='white'
                        ))

        fig.update_layout(height=700)
        st.plotly_chart(fig, use_container_width=True)

    # Download section
    st.subheader("💾 Export Network")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Save Network as GEXF"):
            with st.spinner("Saving network..."):
                output_path = save_network_analysis(G)
                st.success(f"Network saved as: {output_path}")
                st.info("You can open this file in Gephi or other network analysis tools.")

    with col2:
        # Create a summary report
        if st.button("Generate Summary Report"):
            with st.spinner("Generating report..."):
                st.subheader("📋 Network Summary Report")

                report = f"""
                **Entity Network Analysis Report**

                **Dataset:** Enron Email Entities
                **Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

                **Network Statistics:**
                - Total Entities: {G.number_of_nodes()}
                - Total Relationships: {G.number_of_edges()}
                - Network Density: {nx.density(G):.3f}
                - Connected Components: {len(list(nx.connected_components(G)))}

                **Entity Type Breakdown:**
                """

                node_types = [G.nodes[node].get('type', 'UNKNOWN') for node in G.nodes()]
                type_counts = Counter(node_types)

                for entity_type, count in type_counts.most_common():
                    report += f"- {entity_type}: {count} entities\n"

                st.markdown(report)


if __name__ == "__main__":
    main()
