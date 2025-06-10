import streamlit as st
import json
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import plotly.graph_objects as go
from collections import Counter
import warnings

from src.config.config import PROCESSED_JSON_OUTPUT

warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(page_title="Email Network Analysis", layout="wide", initial_sidebar_state="collapsed")

# Define the file path
  # Change this to your actual file path


@st.cache_data
def load_and_process_data():
    """Load JSON data and perform all processing"""
    # Read the JSON file
    with open(PROCESSED_JSON_OUTPUT, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Convert to DataFrame
    df = pd.DataFrame(data)

    # Create combined text for clustering
    combined_texts = []
    for idx, row in df.iterrows():
        text_parts = []
        if 'subject' in row and row['subject']:
            text_parts.append(row['subject'])
        if 'summary' in row and row['summary']:
            text_parts.append(row['summary'])
        topics = row.get('entities', {}).get('topics', [])
        if topics:
            text_parts.extend(topics)
        if 'classification' in row and row['classification']:
            text_parts.append(row['classification'])
        combined_text = ' '.join(text_parts)
        combined_texts.append(combined_text)

    # Perform TF-IDF and clustering
    tfidf = TfidfVectorizer(max_features=100, stop_words='english')
    tfidf_matrix = tfidf.fit_transform(combined_texts)

    # Determine optimal number of clusters (between 3 and 7)
    n_clusters = min(max(3, len(df) // 10), 7)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(tfidf_matrix)

    # Calculate similarity matrix
    similarity_matrix = cosine_similarity(tfidf_matrix)

    return df, clusters, similarity_matrix, n_clusters


def create_network_graph(df, clusters, similarity_matrix, threshold=0.3):
    """Create network graph from clustering results"""
    G = nx.Graph()

    # Add nodes with attributes
    for idx, row in df.iterrows():
        G.add_node(idx,
                   email_id=row.get('email_id', idx),
                   subject=row.get('subject', 'No Subject'),
                   cluster=clusters[idx],
                   classification=row.get('classification', 'Unknown'),
                   from_email=row.get('from', 'Unknown'),
                   to_email=row.get('to', 'Unknown'),
                   date=row.get('date', 'Unknown'),
                   topics=row.get('entities', {}).get('topics', []))

    # Add edges based on similarity
    for i in range(len(df)):
        for j in range(i + 1, len(df)):
            if similarity_matrix[i, j] > threshold:
                G.add_edge(i, j, weight=similarity_matrix[i, j])

    return G


def create_interactive_plot(G, df, clusters):
    """Create interactive Plotly network visualization"""
    # Use Kamada-Kawai layout for better structure
    pos = nx.kamada_kawai_layout(G)

    # Create edge traces
    edge_traces = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        weight = G[edge[0]][edge[1]]['weight']

        edge_trace = go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            line=dict(width=weight * 5, color='rgba(125,125,125,0.3)'),
            hoverinfo='none',
            mode='lines',
            showlegend=False)
        edge_traces.append(edge_trace)

    # Define cluster colors
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#FFD93D', '#6C5CE7']

    # Create node traces for each cluster
    node_traces = []
    for cluster_id in range(max(clusters) + 1):
        cluster_nodes = [node for node in G.nodes() if G.nodes[node]['cluster'] == cluster_id]

        if not cluster_nodes:
            continue

        node_x = []
        node_y = []
        node_text = []
        hover_text = []

        for node in cluster_nodes:
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)

            # Create hover text
            email_data = G.nodes[node]
            hover = f"<b>Email ID:</b> {email_data['email_id']}<br>"
            hover += f"<b>Subject:</b> {email_data['subject'][:50]}...<br>"
            hover += f"<b>From:</b> {email_data['from_email']}<br>"
            hover += f"<b>Classification:</b> {email_data['classification']}<br>"
            if email_data['topics']:
                hover += f"<b>Topics:</b> {', '.join(email_data['topics'][:5])}"
            hover_text.append(hover)
            node_text.append(str(email_data['email_id']))

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            name=f'Cluster {cluster_id}',
            text=node_text,
            textposition="top center",
            textfont=dict(size=8),
            hovertext=hover_text,
            hoverinfo='text',
            marker=dict(
                size=20,
                color=colors[cluster_id % len(colors)],
                line=dict(color='white', width=2),
                opacity=0.9
            )
        )
        node_traces.append(node_trace)

    # Create figure
    fig = go.Figure(data=edge_traces + node_traces,
                    layout=go.Layout(
                        showlegend=True,
                        hovermode='closest',
                        margin=dict(b=0, l=0, r=0, t=0),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        plot_bgcolor='white',
                        height=600,
                        legend=dict(
                            yanchor="top",
                            y=0.99,
                            xanchor="left",
                            x=0.01,
                            bgcolor="rgba(255,255,255,0.8)"
                        )
                    ))

    return fig


def generate_network_summary(G, df, clusters):
    """Generate a comprehensive summary of the network"""
    summary = {}

    # Basic network metrics
    summary['total_emails'] = len(df)
    summary['total_connections'] = G.number_of_edges()
    summary['num_clusters'] = len(set(clusters))

    # Calculate density
    possible_connections = len(df) * (len(df) - 1) / 2
    summary['network_density'] = G.number_of_edges() / possible_connections if possible_connections > 0 else 0

    # Find most connected emails
    degrees = dict(G.degree())
    sorted_degrees = sorted(degrees.items(), key=lambda x: x[1], reverse=True)
    summary['most_connected'] = [(G.nodes[node]['email_id'], degree) for node, degree in sorted_degrees[:3]]

    # Cluster analysis
    cluster_info = {}
    for cluster_id in range(max(clusters) + 1):
        cluster_emails = df[clusters == cluster_id]

        # Get top topics for this cluster
        all_topics = []
        for idx, row in cluster_emails.iterrows():
            topics = row.get('entities', {}).get('topics', [])
            all_topics.extend(topics)

        topic_counts = Counter(all_topics)
        top_topics = [topic for topic, _ in topic_counts.most_common(5)]

        # Get classifications
        classifications = cluster_emails['classification'].value_counts().to_dict()

        cluster_info[cluster_id] = {
            'size': len(cluster_emails),
            'percentage': (len(cluster_emails) / len(df)) * 100,
            'top_topics': top_topics,
            'main_classification': classifications
        }

    summary['clusters'] = cluster_info

    return summary


def display_summary(summary):
    """Display network summary in a user-friendly format"""
    # Create three columns for the summary
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Emails", summary['total_emails'])
        st.metric("Network Connections", summary['total_connections'])

    with col2:
        st.metric("Number of Clusters", summary['num_clusters'])
        st.metric("Network Density", f"{summary['network_density']:.2%}")

    with col3:
        st.subheader("Most Connected Emails")
        for email_id, connections in summary['most_connected']:
            st.write(f"Email {email_id}: {connections} connections")

    # Cluster details
    st.subheader("📊 Cluster Analysis")

    for cluster_id, info in summary['clusters'].items():
        with st.expander(f"**Cluster {cluster_id}** - {info['size']} emails ({info['percentage']:.1f}%)"):
            col1, col2 = st.columns(2)

            with col1:
                st.write("**Main Topics:**")
                for topic in info['top_topics']:
                    st.write(f"• {topic}")

            with col2:
                st.write("**Classifications:**")
                for classification, count in list(info['main_classification'].items())[:3]:
                    st.write(f"• {classification}: {count}")


def main():
    # Title and description
    st.title("📧 Email Network Analysis")
    st.markdown("Interactive visualization of email communication patterns and topic clusters")
    st.markdown("---")

    try:
        # Load and process data (cached)
        df, clusters, similarity_matrix, n_clusters = load_and_process_data()

        # Create network graph
        G = create_network_graph(df, clusters, similarity_matrix, threshold=0.3)

        # Generate summary
        summary = generate_network_summary(G, df, clusters)

        # Display summary metrics
        display_summary(summary)

        # Display interactive network graph
        st.subheader("🔗 Email Network Graph")
        st.markdown(
            "*Hover over nodes to see email details. Nodes are colored by cluster, and connections show content similarity.*")

        print("Running", clusters)
        fig = create_interactive_plot(G, df, clusters)
        st.plotly_chart(fig, use_container_width=True)

        # Network insights
        st.subheader("🔍 Key Insights")

        # Calculate insights
        avg_connections = sum(dict(G.degree()).values()) / len(G.nodes()) if len(G.nodes()) > 0 else 0
        isolated_nodes = sum(1 for node, degree in G.degree() if degree == 0)

        col1, col2 = st.columns(2)

        with col1:
            st.info(f"**Average Connections per Email:** {avg_connections:.1f}")
            st.info(f"**Isolated Emails:** {isolated_nodes}")

        with col2:
            largest_cluster = max(summary['clusters'].items(), key=lambda x: x[1]['size'])
            st.info(f"**Largest Cluster:** Cluster {largest_cluster[0]} with {largest_cluster[1]['size']} emails")

            if summary['network_density'] > 0.5:
                st.success("**High Network Density:** Emails show strong interconnections")
            elif summary['network_density'] > 0.2:
                st.warning("**Moderate Network Density:** Some email groups are well-connected")
            else:
                st.error("**Low Network Density:** Emails show limited connections")

        # Summary interpretation
        st.subheader("📝 Network Summary")

        summary_text = f"""
        This email network consists of **{summary['total_emails']} emails** organized into **{summary['num_clusters']} distinct clusters** 
        based on content similarity. The network has **{summary['total_connections']} connections** between emails, 
        representing a density of **{summary['network_density']:.1%}**.

        The clustering analysis reveals different communication patterns and topics within the email dataset. 
        Each cluster represents emails with similar content, subjects, or communication patterns. The most connected 
        emails serve as central nodes in the communication network, potentially indicating key conversations or 
        frequently referenced topics.
        """

        st.markdown(summary_text)

    except FileNotFoundError:
        st.error(
            f"Error: Could not find file '{PROCESSED_JSON_OUTPUT}'. Please ensure the file exists in the correct location.")
    except json.JSONDecodeError:
        st.error("Error: The file contains invalid JSON. Please check the file format.")
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")


if __name__ == "__main__":
    main()