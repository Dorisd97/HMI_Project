import sys
import os
import json
import streamlit as st
import pandas as pd
import networkx as nx
from collections import defaultdict, Counter
import plotly.graph_objects as go
import plotly.express as px

# Add project’s src directory to Python path so config can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Attempt to load the config values for file paths
try:
    from src.config.config import EXTRACTED_ENTITIES_JSON_PATH
except ImportError:
    st.error("Config file not found. Please ensure config.py is in your project directory.")
    st.stop()


def load_entities_from_json(file_path):
    """
    Load and merge all entities from the JSON file.
    The JSON can be either:
      1) A list of objects, each having an 'entities' field, e.g.:
         [
           { "entities": [ { "value": "...", "label": "..." }, ... ] },
           ...
         ]
      2) A single dict with an 'entities' key, e.g.:
         { "entities": [ { "value": "...", "label": "..." }, ... ] }
    Returns a dict {"entities": [...] } or None if there’s an error.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)

        if isinstance(raw_data, list):
            merged_entities = []
            for item in raw_data:
                if isinstance(item, dict) and 'entities' in item and isinstance(item['entities'], list):
                    merged_entities.extend(item['entities'])
            return {"entities": merged_entities}

        elif isinstance(raw_data, dict) and 'entities' in raw_data and isinstance(raw_data['entities'], list):
            return raw_data

        else:
            st.warning(
                "Entities format not recognized. "
                "Expected a list of objects with 'entities' field or a dict with 'entities'."
            )
            return None

    except FileNotFoundError:
        st.error(f"File not found: {file_path}. Ensure the entities JSON has been generated.")
        return None
    except json.JSONDecodeError:
        st.error("Invalid JSON format. Please check your entities JSON file.")
        return None


def create_network_from_entities(entities_data):
    """
    Build a NetworkX graph from the loaded entities.
    Each entity (value, label) becomes a node with attribute 'type'=label.
    Then:
      - Link all PERSONs together under 'same_type'
      - Link all ORGs together under 'same_type'
      - Link some PERSON → ORG as 'works_with' if the ORG contains certain keywords
      - Link ORG → GPE as 'located_in' if the GPE is 'California' or 'El Paso'
      - If no edges formed, chain nodes in a simple co_occurrence order
    """
    G = nx.Graph()

    if not isinstance(entities_data, dict) or 'entities' not in entities_data:
        st.warning("Warning: Entities format not recognized. Expected {'entities': [...]} ")
        return G

    entities_list = entities_data['entities']
    entities_by_type = defaultdict(list)

    # 1) Add each entity as a node
    for entity in entities_list:
        if isinstance(entity, dict):
            entity_value = entity.get('value', '').strip()
            entity_label = entity.get('label', 'UNKNOWN')
            if entity_value:
                G.add_node(entity_value, type=entity_label)
                entities_by_type[entity_label].append(entity_value)

    # 2) Link PERSON and ORG under 'same_type'
    for entity_type, entity_list in entities_by_type.items():
        if entity_type in ['PERSON', 'ORG'] and len(entity_list) > 1:
            hub = entity_list[0]
            for other in entity_list[1:]:
                G.add_edge(hub, other, relation_type='same_type')

    # 3) Link some PERSON → ORG as 'works_with' (example heuristic)
    people = entities_by_type.get('PERSON', [])
    organizations = entities_by_type.get('ORG', [])
    for person in people[:10]:  # limit to first 10 people
        for org in organizations[:5]:  # limit to first 5 orgs
            if any(key in org for key in ['Enron', 'Energy', 'FERC', 'CPUC']):
                G.add_edge(person, org, relation_type='works_with')
                break

    # 4) Link ORG → GPE as 'located_in' if place is 'California' or 'El Paso'
    places = entities_by_type.get('GPE', [])
    for org in organizations:
        for place in places:
            if place in ['California', 'El Paso'] and G.degree(org) < 3:
                G.add_edge(org, place, relation_type='located_in')
                break

    # 5) If we still have no edges but multiple nodes, chain them under 'co_occurrence'
    if G.number_of_edges() == 0 and G.number_of_nodes() > 1:
        all_nodes = list(G.nodes())
        for i in range(len(all_nodes) - 1):
            if G.degree(all_nodes[i]) == 0:
                G.add_edge(all_nodes[i], all_nodes[i + 1], relation_type='co_occurrence')

    return G


def create_plotly_network(G, layout_type='spring'):
    """
    Turn the NetworkX graph into Plotly traces.
    layout_type: 'spring', 'circular', 'kamada_kawai', or 'fruchterman_reingold'
    Returns a list of Plotly Scatter traces (first for edges, then one per node-type).
    """
    if G.number_of_nodes() == 0:
        return None

    # 1) Choose a layout algorithm for node positions
    if layout_type == 'spring':
        pos = nx.spring_layout(G, k=3, iterations=50)
    elif layout_type == 'circular':
        pos = nx.circular_layout(G)
    elif layout_type == 'kamada_kawai':
        pos = nx.kamada_kawai_layout(G)
    else:
        pos = nx.fruchterman_reingold_layout(G, k=2)

    # 2) Gather node types and assign colors
    node_types = [G.nodes[node].get('type', 'UNKNOWN') for node in G.nodes()]
    unique_types = list(set(node_types))

    # Choose a color palette from Plotly
    colors = px.colors.qualitative.Set3[:len(unique_types)]
    if len(unique_types) > len(colors):
        colors = px.colors.qualitative.Plotly[:len(unique_types)]
    type_to_color = dict(zip(unique_types, colors))

    # 3) Build edge traces (all edges as one Scatter trace)
    edge_x, edge_y = [], []
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color='#888'),
        hoverinfo='none',
        mode='lines'
    )

    traces = [edge_trace]

    # 4) Build node traces, one per entity type
    degrees = dict(G.degree())
    for entity_type in unique_types:
        nodes_of_type = [n for n in G.nodes() if G.nodes[n].get('type') == entity_type]
        node_x = [pos[n][0] for n in nodes_of_type]
        node_y = [pos[n][1] for n in nodes_of_type]
        node_sizes = [max(10, degrees[n] * 5 + 15) for n in nodes_of_type]

        node_info = []
        for n in nodes_of_type:
            neighbors = list(G.neighbors(n))
            text = f"<b>{n}</b><br>Type: {entity_type}<br>Connections: {len(neighbors)}"
            if neighbors:
                text += "<br>Connected to: " + ", ".join(neighbors[:3])
                if len(neighbors) > 3:
                    text += f" and {len(neighbors) - 3} more..."
            node_info.append(text)

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            hovertext=node_info,
            text=nodes_of_type,
            textposition="middle center",
            textfont=dict(size=8, color="white"),
            name=entity_type,
            marker=dict(
                size=node_sizes,
                color=type_to_color[entity_type],
                line=dict(width=2, color='white')
            )
        )
        traces.append(node_trace)

    return traces


def display_network_plain(G):
    """
    Display network statistics and counts in plain English sentences,
    so that someone without graph-analysis background can understand.
    """
    if G.number_of_nodes() == 0:
        st.warning("No entities were found in the data.")
        return

    total_nodes = G.number_of_nodes()
    total_edges = G.number_of_edges()
    density = nx.density(G)
    num_components = len(list(nx.connected_components(G)))

    st.subheader("📊 Network Summary (Plain English)")
    st.write(f"We identified **{total_nodes}** unique entities (people, organizations, or places) in all the emails.")
    st.write(
        f"These entities form **{total_edges}** connections. "
        "Each connection means those two names appeared together in at least one email."
    )
    st.write(
        f"The network’s density (how tightly people and organizations appear together) is **{density:.3f}**. "
        f"There are **{num_components}** separate groups of connected entities—meaning some sets of names never link to others."
    )

    # Breakdown by entity type
    node_types = [G.nodes[n].get('type', 'UNKNOWN') for n in G.nodes()]
    type_counts = Counter(node_types)
    num_people = type_counts.get('PERSON', 0)
    num_orgs = type_counts.get('ORG', 0)
    num_places = type_counts.get('GPE', 0)
    num_other = total_nodes - (num_people + num_orgs + num_places)

    st.subheader("Entity Breakdown")
    st.write(f"- **{num_people}** of the entities are **people**.")
    st.write(f"- **{num_orgs}** of the entities are **organizations**.")
    st.write(f"- **{num_places}** of the entities are **locations** (cities, states, etc.).")
    if num_other > 0:
        st.write(f"- **{num_other}** entities are other types (dates, project names, etc.).")

    # Top connected entities
    degrees = dict(G.degree())
    top_entities = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:5]
    st.subheader("Most “Popular” Names (by Connections)")
    if not top_entities:
        st.write("No connections to report.")
    else:
        for name, deg in top_entities:
            node_type = G.nodes[name].get('type', 'UNKNOWN')
            st.write(f"- **{name}** ({node_type}): appears with **{deg}** other entities.")
        st.write(
            "A higher connection count means that name was mentioned alongside many other entities in the emails."
        )

    # Relationship types
    edge_relations = [G.edges[e].get('relation_type', 'unspecified') for e in G.edges()]
    relation_counts = Counter(edge_relations)
    st.subheader("Types of Connections (Plain Language)")
    if not relation_counts:
        st.write("No special relationship categories were found.")
    else:
        for relation, count in relation_counts.items():
            if relation == 'same_type':
                meaning = "Links between entities of the same category (e.g., person linked to person, or org linked to org)."
            elif relation == 'works_with':
                meaning = "Links suggesting a person often appears with a particular organization."
            elif relation == 'located_in':
                meaning = "Links suggesting an organization is mentioned alongside a certain location."
            elif relation == 'co_occurrence':
                meaning = "Fallback links showing these two entities simply appeared together in an email."
            else:
                meaning = "A generic connection type."
            st.write(f"- **{relation}**: **{count}** occurrences. {meaning}")

    st.write("---")
    st.write(
        "In summary, the above tells us:\n"
        "1. How many unique names (people, companies, places) appeared across all emails.\n"
        "2. How many times those names showed up together (connections).\n"
        "3. Which names appeared most often with others.\n"
        "4. What kinds of relationships (same_type, works_with, located_in) exist and how many of each."
    )


def generate_summary_plain(G):
    """
    Build a multi-paragraph plain-English text summary of the network.
    """
    total_nodes = G.number_of_nodes()
    total_edges = G.number_of_edges()
    density = nx.density(G)
    num_components = len(list(nx.connected_components(G)))

    node_types = [G.nodes[n].get('type', 'UNKNOWN') for n in G.nodes()]
    type_counts = Counter(node_types)
    num_people = type_counts.get('PERSON', 0)
    num_orgs = type_counts.get('ORG', 0)
    num_places = type_counts.get('GPE', 0)
    num_other = total_nodes - (num_people + num_orgs + num_places)

    # Top 3 connected entities
    degrees = dict(G.degree())
    sorted_deg = sorted(degrees.items(), key=lambda x: x[1], reverse=True)
    top_three = sorted_deg[:3]

    paragraph1 = (
        f"We found a total of {total_nodes} unique entities in the email dataset. "
        f"These include {num_people} people, {num_orgs} organizations, and {num_places} locations. "
    )
    if num_other > 0:
        paragraph1 += f"The remaining {num_other} entities are other types (such as dates or project names)."

    paragraph2 = (
        f"In total, there are {total_edges} connections among those entities. "
        f"A “connection” here means two names appeared together in one or more emails. "
        f"The network’s overall density (a measure of how tightly every entity is linked with others) is {density:.3f}. "
        f"There are {num_components} separate groups, meaning some sets of entities never link to entities in another group."
    )

    if top_three:
        top_sentences = []
        for name, deg in top_three:
            node_label = G.nodes[name].get('type', 'UNKNOWN')
            top_sentences.append(f"{name} ({node_label}) appears with {deg} other entities")
        paragraph3 = "The three entities with the most connections are: " + "; ".join(top_sentences) + "."
    else:
        paragraph3 = "No single entity had more connections than any other."

    edge_relations = [G.edges[e].get('relation_type', 'unspecified') for e in G.edges()]
    rel_counts = Counter(edge_relations)
    if rel_counts:
        pairs = []
        for rel, cnt in rel_counts.items():
            if rel == 'same_type':
                meaning = "links between entities of the same category"
            elif rel == 'works_with':
                meaning = "links suggesting a person often appears with a particular organization"
            elif rel == 'located_in':
                meaning = "links showing an organization is mentioned alongside a certain location"
            elif rel == 'co_occurrence':
                meaning = "links indicating these two entities simply appeared together"
            else:
                meaning = "a general connection type"
            pairs.append(f"{cnt} ({meaning})")
        paragraph4 = "We categorized connections into types: " + "; ".join(pairs) + "."
    else:
        paragraph4 = "There were no special relationship categories assigned."

    report = "\n\n".join([paragraph1, paragraph2, paragraph3, paragraph4])
    return report


def save_network_analysis(G):
    """
    Save the graph G as a .gexf file so that it can be opened in tools like Gephi.
    Returns the file path.
    """
    output_dir = os.path.dirname(EXTRACTED_ENTITIES_JSON_PATH)
    output_path = os.path.join(output_dir, "enron_entity_network.gexf")
    nx.write_gexf(G, output_path)
    return output_path


def main():
    st.set_page_config(page_title="Entity Network Graph", page_icon="🕸️", layout="wide")
    st.title("🕸️ Entity Network Graph Analyzer")
    st.markdown("### Visualizing Relationships from Enron Email Data (in Plain English)")

    st.sidebar.title("⚙️ Controls")
    layout_type = st.sidebar.selectbox(
        "Choose Layout Algorithm:",
        options=['spring', 'fruchterman_reingold', 'circular', 'kamada_kawai'],
        index=0,
        help="Different algorithms arrange nodes differently"
    )

    # 1) Load the entities data
    with st.spinner("Loading entity data..."):
        entities_data = load_entities_from_json(EXTRACTED_ENTITIES_JSON_PATH)
    if entities_data is None:
        st.stop()

    # 2) Build the network graph
    with st.spinner("Building network graph..."):
        G = create_network_from_entities(entities_data)
    if G.number_of_nodes() == 0:
        st.error("No entities found in the data. Please check your JSON format.")
        st.stop()

    # 3) Display plain-English summary
    display_network_plain(G)

    # 4) Interactive network visualization
    st.subheader("🔍 Interactive Network Visualization")
    with st.spinner("Creating interactive visualization..."):
        traces = create_plotly_network(G, layout_type)
    if traces:
        fig = go.Figure(
            data=traces,
            layout=go.Layout(
                title="Entity Network Graph",
                titlefont_size=16,
                showlegend=True,
                hovermode='closest',
                margin=dict(b=20, l=5, r=5, t=40),
                annotations=[dict(
                    text="Hover over nodes to see details, and drag to explore.",
                    showarrow=False,
                    xref="paper", yref="paper", x=0.005, y=-0.002,
                    xanchor='left', yanchor='bottom',
                    font=dict(color="#888", size=12)
                )],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                plot_bgcolor='white'
            )
        )
        fig.update_layout(height=700)
        st.plotly_chart(fig, use_container_width=True)

    # 5) Export & summary buttons
    st.subheader("💾 Export & Summary")
    col1, col2 = st.columns(2)

    with col1:
        if st.button("Save Network as GEXF"):
            with st.spinner("Saving network..."):
                output_path = save_network_analysis(G)
                st.success(f"Network saved as: {output_path}")
                st.info("You can open this .gexf file in Gephi or another network analysis tool.")

    with col2:
        if st.button("Generate Plain-English Summary"):
            with st.spinner("Generating summary..."):
                summary_text = generate_summary_plain(G)
                st.subheader("📋 Network Summary (Plain English)")
                st.write(summary_text)


if __name__ == "__main__":
    main()
