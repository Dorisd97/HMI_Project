"""
Enron Email Explorer with Topic Clusters

Performance-optimized Streamlit app for exploring email communication patterns.
Handles large datasets with WebSocket stability improvements.

If you encounter WebSocket errors, try:
1. Refresh the browser page
2. Reduce the number of displayed nodes
3. Close and reopen the browser tab
4. Clear browser cache

Author: Assistant
"""

import os
os.environ['TRANSFORMERS_CACHE'] = "E:/.cache"
os.environ['HF_HOME'] = "E:/.cache"

from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer
import hdbscan
import streamlit as st
import json
import pandas as pd
import numpy as np
import networkx as nx
from collections import defaultdict, Counter
import time
import logging

# Configure logging to reduce noise
logging.getLogger('tornado.access').setLevel(logging.WARNING)
logging.getLogger('streamlit').setLevel(logging.WARNING)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

from streamlit_agraph import agraph, Node, Edge, Config

from src.config.config import PROCESSED_JSON_OUTPUT


# All other configurations and constants like st.set_page_config, TOPIC_COLORS, etc. remain the same
# We'll update your show_topic_clustering() and fit_topic_model() replacement here

@st.cache_data(hash_funcs={pd.DataFrame: lambda _: None})
def fit_transformer_topic_model(corpus):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(corpus, show_progress_bar=True)
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=15, metric='euclidean', prediction_data=True)
    cluster_labels = clusterer.fit_predict(embeddings_scaled)
    return cluster_labels, embeddings, clusterer

@st.cache_data
def extract_keywords_by_cluster(df, n_top=8):
    cluster_texts = df.groupby('topic_id')['classification'].apply(lambda texts: ' '.join(texts))
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    X = vectorizer.fit_transform(cluster_texts)
    keywords = []
    for row in X.toarray():
        top_idxs = row.argsort()[-n_top:][::-1]
        keywords.append([vectorizer.get_feature_names_out()[i] for i in top_idxs])
    return keywords

def show_topic_clustering(df):
    st.header("1️⃣ Transformer-Based Topic Clustering")
    df = process_to_list(df)

    with st.spinner("Encoding and clustering topics..."):
        corpus = (df['classification'].fillna('') + ' ' +
                  df['subject'].fillna('') + ' ' +
                  df['summary'].fillna(''))
        cluster_labels, embeddings, clusterer = fit_transformer_topic_model(corpus)
        df['topic_id'] = cluster_labels
        df = df[df['topic_id'] != -1]  # Filter noise
        df['topic_weight'] = 1.0
        topics = extract_keywords_by_cluster(df)

    n_topics = len(topics)
    st.subheader("Topic Keywords")
    cols = st.columns(min(n_topics, 6))
    for t, col in enumerate(cols * (n_topics // len(cols) + 1)):
        if t >= n_topics:
            break
        color = TOPIC_COLORS[t % len(TOPIC_COLORS)]
        col.markdown(f"<div style='background-color: {color}; padding: 10px; border-radius: 5px;'>"
                     f"<strong style='color:white;'>Topic {t}</strong><br>"
                     f"<span style='color:white;'>{', '.join(topics[t])}</span></div>", unsafe_allow_html=True)

    st.subheader("Emails per Topic")
    st.bar_chart(df['topic_id'].value_counts().sort_index())
    return df, topics

# Keep rest of your code: show_comm_network, main(), etc. unchanged
# Only replace the old NMF-based fit_topic_model and show_topic_clustering

st.set_page_config(
    page_title="Enron Email Explorer",
    layout="wide",
    initial_sidebar_state="expanded",
    # Add memory management settings
    menu_items={
        'About': "Enron Email Network Analysis - Optimized for large datasets"
    }
)

# Define colors for topics/clusters
TOPIC_COLORS = [
    "#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7",
    "#DDA0DD", "#F4A460", "#87CEEB", "#98D8C8", "#FFB6C1"
]

# Alternative hash function to avoid caching issues
def hash_dataframe(df):
    """Custom hash function that excludes problematic columns"""
    # Only hash basic columns, exclude lists and complex objects
    hashable_cols = [col for col in df.columns if col not in ['to_list']]
    return hash(str(df[hashable_cols].values.tobytes()))

@st.cache_data(hash_funcs={pd.DataFrame: hash_dataframe})
def load_data():
    """Load and preprocess data with progress indication"""
    with st.spinner("Loading email data..."):
        with open(PROCESSED_JSON_OUTPUT, 'r', encoding='utf-8') as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        # Pre-filter out null values
        df = df.dropna(subset=['from'])
        # Convert to string for caching, we'll process to_list later
        df['to_cleaned'] = df['to'].fillna('')
    return df

def process_to_list(df):
    """Process the 'to' field into lists after loading from cache"""
    if 'to_list' not in df.columns:
        df['to_list'] = df['to_cleaned'].str.split(r'\s*,\s*')
        df['to_list'] = df['to_list'].apply(lambda x: [addr.strip() for addr in x if addr.strip()] if x else [])
    return df

@st.cache_data
def fit_topic_model(corpus, n_topics):
    """Optimized topic modeling with progress indication"""
    with st.spinner("Training topic model..."):
        tfidf = TfidfVectorizer(max_features=500, stop_words='english', max_df=0.8, min_df=2)
        X = tfidf.fit_transform(corpus)
        nmf = NMF(n_components=n_topics, random_state=42, max_iter=100).fit(X)
        W = nmf.transform(X)
        H = nmf.components_
    return tfidf, nmf, W, H

@st.cache_data
def get_user_topic_affiliation_optimized(_df, topic_ids, topic_weights):
    """Highly optimized topic affiliation calculation"""
    with st.spinner("Calculating user topic affiliations..."):
        # Ensure to_list is processed
        df = process_to_list(_df.copy())

        user_topics = defaultdict(lambda: defaultdict(float))

        # Vectorized approach
        for idx, (from_addr, to_list, topic_id, topic_weight) in enumerate(zip(
            df['from'], df['to_list'], topic_ids, topic_weights
        )):
            # Add to sender
            user_topics[from_addr][topic_id] += topic_weight

            # Add to receivers (batch process)
            if to_list:
                for recipient in to_list:
                    user_topics[recipient][topic_id] += topic_weight

        # Determine primary topic for each user (vectorized)
        user_primary_topics = {
            user: max(topics.items(), key=lambda x: x[1])[0]
            for user, topics in user_topics.items()
        }

    return user_primary_topics

@st.cache_data
def build_comm_graph_optimized(_df):
    """Optimized graph building with progress indication"""
    with st.spinner("Building communication graph..."):
        # Ensure to_list is processed
        df = process_to_list(_df.copy())

        # Vectorized approach for counting communications
        pairs_data = []
        for from_addr, to_list in zip(df['from'], df['to_list']):
            if to_list:
                for to_addr in to_list:
                    pairs_data.append((from_addr, to_addr))

        # Use Counter for efficient counting
        edge_counts = Counter(pairs_data)

        # Build graph efficiently
        G = nx.DiGraph()

        # Batch add edges
        edges_to_add = [(from_addr, to_addr, {'weight': count})
                       for (from_addr, to_addr), count in edge_counts.items()]
        G.add_edges_from(edges_to_add)

        # Calculate node attributes efficiently
        node_attrs = {}
        for node in G.nodes():
            sent = sum(data['weight'] for _, _, data in G.out_edges(node, data=True))
            recv = sum(data['weight'] for _, _, data in G.in_edges(node, data=True))
            node_attrs[node] = {'sent': sent, 'recv': recv}

        nx.set_node_attributes(G, node_attrs)

    return G

# def show_topic_clustering(df):
#     st.header("1️⃣ Topic Clustering")
#
#     # Ensure to_list is processed
#     df = process_to_list(df)
#
#     # Sidebar controls
#     n_topics = st.sidebar.slider("Number of Topics", 3, 10, 5)
#
#     with st.spinner("Processing topic clustering..."):
#         # Build corpus efficiently
#         corpus = (df['classification'].fillna('') + ' ' +
#                  df['subject'].fillna('') + ' ' +
#                  df['summary'].fillna(''))
#
#         # Fit topic model
#         tfidf, nmf, W, H = fit_topic_model(corpus, n_topics)
#
#         # Extract topics
#         feature_names = tfidf.get_feature_names_out()
#         top_n = 8
#         topics = []
#         for t in range(n_topics):
#             idxs = H[t].argsort()[::-1][:top_n]
#             topics.append([feature_names[i] for i in idxs])
#
#         # Assign topics to emails
#         df['topic_id'] = W.argmax(axis=1)
#         df['topic_weight'] = W.max(axis=1)
#
#     # Show topic keywords with colors
#     st.subheader("Topic Keywords")
#     cols = st.columns(n_topics)
#     for t, col in enumerate(cols):
#         color = TOPIC_COLORS[t % len(TOPIC_COLORS)]
#         col.markdown(f"<div style='background-color: {color}; padding: 10px; border-radius: 5px; margin: 5px;'>"
#                     f"<strong style='color: white;'>Topic {t}</strong><br>"
#                     f"<span style='color: white; font-size: 0.9em;'>{', '.join(topics[t])}</span>"
#                     f"</div>", unsafe_allow_html=True)
#
#     # Topic distribution
#     counts = df['topic_id'].value_counts().sort_index()
#     st.subheader("Emails per Topic")
#     st.bar_chart(counts)
#
#     return df, topics

def create_clustered_positions(nodes_by_topic, n_topics, G):
    """Create well-spaced clustered positions for nodes based on their topics"""
    import math

    positions = {}

    # Much larger spacing for better visibility
    cluster_radius = 500  # Distance between cluster centers
    base_node_spacing = 80  # Minimum distance between nodes

    for topic_id, node_list in nodes_by_topic.items():
        n_nodes = len(node_list)

        # Calculate cluster center position
        angle = (2 * math.pi * topic_id) / n_topics
        cluster_x = cluster_radius * math.cos(angle)
        cluster_y = cluster_radius * math.sin(angle)

        if n_nodes == 1:
            # Single node at cluster center
            positions[node_list[0]] = {'x': cluster_x, 'y': cluster_y}
        elif n_nodes <= 6:
            # Small clusters - circular arrangement
            node_cluster_radius = max(60, n_nodes * 25)
            for i, node in enumerate(node_list):
                node_angle = (2 * math.pi * i) / n_nodes
                node_x = cluster_x + node_cluster_radius * math.cos(node_angle)
                node_y = cluster_y + node_cluster_radius * math.sin(node_angle)
                positions[node] = {'x': node_x, 'y': node_y}
        else:
            # Larger clusters - spiral arrangement for better spacing
            nodes_sorted = sorted(node_list, key=lambda x: G.nodes[x]['sent'] + G.nodes[x]['recv'], reverse=True)

            for i, node in enumerate(nodes_sorted):
                if i == 0:
                    # Most active node at center
                    positions[node] = {'x': cluster_x, 'y': cluster_y}
                else:
                    # Spiral outward
                    spiral_radius = 40 + (i * 30)
                    spiral_angle = i * 0.8  # Golden angle for even distribution
                    node_x = cluster_x + spiral_radius * math.cos(spiral_angle)
                    node_y = cluster_y + spiral_radius * math.sin(spiral_angle)
                    positions[node] = {'x': node_x, 'y': node_y}

    return positions

def show_comm_network(df, topics):
    st.header("2️⃣ Communication Network by Topic Clusters")

    # Performance warning for large datasets
    if len(df) > 50000:
        st.warning("⚠️ Large dataset detected. Consider reducing node count for better performance.")

    # Enhanced controls with performance optimizations
    col1, col2, col3 = st.columns(3)
    with col1:
        min_activity = st.slider("Minimum total emails (sent + received)", 1, 100, 20)
    with col2:
        # Reduce default max nodes for better stability
        max_nodes = st.slider("Maximum nodes to display", 50, 500, 150,
                             help="Lower values improve performance and reduce WebSocket errors")
    with col3:
        show_full_emails = st.checkbox("Show full email addresses", value=False,
                                      help="Show complete email addresses in node labels")

    # Get user topic affiliations (optimized)
    user_topics = get_user_topic_affiliation_optimized(
        df, df['topic_id'].values, df['topic_weight'].values
    )

    # Build communication graph (optimized)
    G = build_comm_graph_optimized(df)

    # Ensure df has to_list processed for edge analysis
    df = process_to_list(df)

    with st.spinner("Filtering and preparing network visualization..."):
        # Filter nodes by activity
        active_nodes = [node for node, data in G.nodes(data=True)
                       if (data['sent'] + data['recv']) >= min_activity]

        # Limit nodes for performance
        if len(active_nodes) > max_nodes:
            # Sort by activity and take top N
            node_activity = [(node, G.nodes[node]['sent'] + G.nodes[node]['recv'])
                           for node in active_nodes]
            node_activity.sort(key=lambda x: x[1], reverse=True)
            active_nodes = [node for node, _ in node_activity[:max_nodes]]

        G_filtered = G.subgraph(active_nodes).copy()

    if G_filtered.number_of_nodes() == 0:
        st.warning("No nodes meet the criteria. Try lowering the thresholds.")
        return

    # Performance warning
    if G_filtered.number_of_nodes() > 500:
        st.warning(f"⚠️ Displaying {G_filtered.number_of_nodes()} nodes. Consider reducing the maximum nodes for better performance.")

    with st.spinner("Rendering network visualization..."):
        # Group nodes by topic for clustering
        nodes_by_topic = defaultdict(list)
        topic_counts = defaultdict(int)

        # First pass: group nodes by topic
        for node, data in G_filtered.nodes(data=True):
            primary_topic = user_topics.get(node, 0)
            nodes_by_topic[primary_topic].append(node)
            topic_counts[primary_topic] += 1

        # Create clustered positions with better spacing
        n_topics = len(topics)
        positions = create_clustered_positions(nodes_by_topic, n_topics, G_filtered)

        # Build enhanced node information with communication context
        node_communication_info = defaultdict(lambda: {'sent_to': set(), 'received_from': set(), 'topics': set()})

        # Analyze communication patterns per node
        for _, row in df.iterrows():
            from_addr = row['from']
            to_list = row.get('to_list', [])
            topic_id = row.get('topic_id', 0)

            if isinstance(to_list, list) and to_list:
                for to_addr in to_list:
                    if to_addr:  # Ensure to_addr is not empty
                        # Track who this person sends to
                        node_communication_info[from_addr]['sent_to'].add(to_addr)
                        node_communication_info[from_addr]['topics'].add(topic_id)

                        # Track who this person receives from
                        node_communication_info[to_addr]['received_from'].add(from_addr)
                        node_communication_info[to_addr]['topics'].add(topic_id)

        # Build agraph elements with enhanced communication context
        nodes, edges = [], []

        # Create enhanced node labels and information
        for node, data in G_filtered.nodes(data=True):
            primary_topic = user_topics.get(node, 0)
            comm_info = node_communication_info[node]

            color = TOPIC_COLORS[primary_topic % len(TOPIC_COLORS)]
            total_activity = data['sent'] + data['recv']

            # Better size scaling - more distinct sizes
            if total_activity > 100:
                size = 35
            elif total_activity > 50:
                size = 28
            elif total_activity > 20:
                size = 22
            elif total_activity > 10:
                size = 18
            else:
                size = 15

            # Flexible label based on user preference
            if show_full_emails:
                label = node  # Show complete email address

                # If extremely long, show smart truncation
                if len(label) > 25:
                    if '@' in label:
                        username, domain = label.split('@', 1)
                        if len(username) > 18:
                            label = f"{username[:18]}...@{domain}"
                        else:
                            label = f"{username}@{domain}"
                    else:
                        label = label[:25] + "..."
            else:
                # Show just username part (original behavior)
                base_label = node.split('@')[0] if '@' in node else node
                if len(base_label) > 12:
                    base_label = base_label[:12] + "…"
                label = base_label

            # Get clustered position
            pos = positions.get(node, {'x': 0, 'y': 0})

            # Create comprehensive communication summary (clean text)
            sent_to_list = list(comm_info['sent_to'])
            received_from_list = list(comm_info['received_from'])

            sent_to_summary = ", ".join([addr.split('@')[0] for addr in sent_to_list[:5]])
            if len(sent_to_list) > 5:
                sent_to_summary += f" +{len(sent_to_list) - 5} more"

            received_from_summary = ", ".join([addr.split('@')[0] for addr in received_from_list[:5]])
            if len(received_from_list) > 5:
                received_from_summary += f" +{len(received_from_list) - 5} more"

            # Topic names for this user (simplified)
            user_topics_list = [f"Topic {t}" for t in comm_info['topics'] if t < len(topics)]

            # Create tooltip exactly in the format requested by user
            # "To: email, From: email, Topic: cluster X"
            to_list_short = list(comm_info['sent_to'])[:3]  # Top 3 emails this person sends to
            from_list_short = list(comm_info['received_from'])[:3]  # Top 3 emails this person receives from

            # Show full email addresses in tooltip as requested
            to_display = ", ".join(to_list_short) if to_list_short else "No recipients"
            from_display = ", ".join(from_list_short) if from_list_short else "No senders"

            # Simple format exactly as requested
            tooltip_text = f"""To: {to_display}
From: {from_display}
Topic: Cluster {primary_topic}

Activity: Sent {data['sent']}, Received {data['recv']}"""

            # Adjust font size based on label preference
            font_size = 8 if show_full_emails else 10

            nodes.append(Node(
                id=node,
                label=label,
                size=size,
                color=color,
                title=tooltip_text,
                x=pos['x'],
                y=pos['y'],
                # Enable full movement and interactivity like Game of Thrones
                physics=True,   # Enable physics so nodes can be moved
                fixed=False,    # Allow nodes to be dragged and repositioned
                # Enhanced visual styling
                borderWidth=3,
                borderWidthSelected=5,
                labelHighlightBold=True,
                font={
                    'size': font_size,
                    'color': '#ffffff',
                    'face': 'Arial, sans-serif',
                    'background': 'rgba(0,0,0,0.9)',  # Dark background for better readability
                    'strokeWidth': 2,
                    'strokeColor': '#000000',
                    'align': 'center',
                    'multi': False  # Single line for cleaner look
                }
            ))

        # Enhanced edge styling with communication context
        for u, v, data in G_filtered.edges(data=True):
            # Better width scaling
            email_count = data['weight']
            if email_count > 20:
                width = 4
            elif email_count > 10:
                width = 3
            elif email_count > 5:
                width = 2
            else:
                width = 1

            source_topic = user_topics.get(u, 0)
            target_topic = user_topics.get(v, 0)

            # Enhanced edge information with sample email data
            sample_emails = df[(df['from'] == u) & (df['to_list'].apply(lambda x: v in x if isinstance(x, list) and x else False))]

            # Get sample subjects and topics
            if not sample_emails.empty:
                sample_subjects = sample_emails['subject'].head(3).tolist()
                sample_summaries = sample_emails['summary'].head(2).tolist()
                common_topics = sample_emails['topic_id'].mode().tolist()
            else:
                sample_subjects = ["No subject data available"]
                sample_summaries = ["No summary available"]
                common_topics = [source_topic]

            # Better edge coloring with topic context
            if source_topic == target_topic:
                edge_color = TOPIC_COLORS[source_topic % len(TOPIC_COLORS)]
                opacity = "BB"  # More visible for intra-cluster
                use_dash = False
                connection_type = "Same Topic"
            else:
                edge_color = "#888888"
                opacity = "88"  # Less prominent for inter-cluster
                use_dash = [5, 5]  # Dashed for cross-cluster
                connection_type = "Cross Topic"

            # Create simple, clean edge tooltip
            edge_tooltip = f"""{u} → {v}
Emails: {email_count}
From: Cluster {source_topic}
To: Cluster {target_topic}"""

            edges.append(Edge(
                source=u,
                target=v,
                color=edge_color + opacity,
                width=width,
                title=edge_tooltip,
                smooth={
                    'enabled': True,
                    'type': 'continuous',
                    'roundness': 0.3
                },
                arrows={
                    'to': {
                        'enabled': True,
                        'scaleFactor': 1.0
                    }
                },
                dashes=use_dash if isinstance(use_dash, list) else False
            ))

    # Display enhanced topic legend with cluster info
    st.subheader("🎨 Topic Clusters Layout")

    # Create a visual representation of the cluster arrangement
    legend_text = "**Clusters are arranged in a circle by topic:**\n\n"

    legend_cols = st.columns(min(n_topics, 5))
    for t in range(n_topics):
        col_idx = t % len(legend_cols)
        with legend_cols[col_idx]:
            color = TOPIC_COLORS[t % len(TOPIC_COLORS)]
            count = topic_counts[t]
            keywords = ', '.join(topics[t][:3])

            # Enhanced legend with cluster info
            legend_cols[col_idx].markdown(
                f"<div style='background: linear-gradient(135deg, {color} 0%, {color}AA 100%); "
                f"padding: 12px; border-radius: 8px; margin: 4px; border: 2px solid {color};'>"
                f"<div style='color: white; font-weight: bold; font-size: 0.9em; margin-bottom: 4px;'>"
                f"🔸 Topic {t} Cluster"
                f"</div>"
                f"<div style='color: white; font-size: 0.8em; margin-bottom: 6px;'>"
                f"👥 {count} users"
                f"</div>"
                f"<div style='color: white; font-size: 0.75em; opacity: 0.9;'>"
                f"📝 {keywords}"
                f"</div>"
                f"</div>",
                unsafe_allow_html=True
            )

    # Add interaction guide
    st.markdown("---")

    # Simplified interaction guide focused on dragging
    st.markdown("---")
    guide_col1, guide_col2 = st.columns(2)

    with guide_col1:
        st.markdown("**🖱️ Optimized Navigation:**")
        st.markdown("• **🎯 DRAG NODES**: Click and drag any node to move it freely")
        st.markdown("• **🖱️ Pan View**: Drag empty space to move the view")
        st.markdown("• **🔍 Zoom**: Mouse wheel or pinch to zoom in/out")
        st.markdown("• **💡 Hover**: See detailed communication information")
        st.markdown("• **⚖️ Optimized**: Settings tuned for WebSocket stability")
        st.markdown("• **📊 Progress**: Visual feedback for large datasets")

    with guide_col2:
        st.markdown("**📊 Visual Guide:**")
        st.markdown("• **Node Colors**: Topic clusters (legend above)")
        st.markdown("• **Node Labels**: Full email addresses (toggle above)")
        st.markdown("• **Node Sizes**: Email activity level")
        st.markdown("• **Clean Tooltips**: 'To: email, From: email, Topic: Cluster X'")
        st.markdown("• **Solid Edges**: Same-topic communication")
        st.markdown("• **Dashed Edges**: Cross-topic communication")
        st.markdown("• **Edge Thickness**: Volume of emails exchanged")

    # Communication insights section
    st.markdown("### 💡 Communication Insights")

    insight_col1, insight_col2 = st.columns(2)

    with insight_col1:
        st.markdown("**🔍 What You Can Discover:**")
        st.markdown("• **Topic-based Communities**: Who talks about what")
        st.markdown("• **Communication Patterns**: Frequency and direction")
        st.markdown("• **Cross-topic Bridges**: Users connecting different topics")
        st.markdown("• **Information Hubs**: Most active communicators")
        st.markdown("• **Email Content**: Sample subjects and summaries")

    with insight_col2:
        st.markdown("**📈 Analysis Tips:**")
        st.markdown("• **Large Nodes**: Key players in email network")
        st.markdown("• **Solid Edge Clusters**: Focused topic discussions")
        st.markdown("• **Dashed Edges**: Information flow between topics")
        st.markdown("• **Isolated Nodes**: Limited communication scope")
        st.markdown("• **Dense Clusters**: Highly collaborative teams")

    # Optimized graph configuration for WebSocket stability
    config = Config(
        width=1200,  # Slightly smaller for better performance
        height=800,
        directed=True,
        # Optimized physics for stability and performance
        physics={
            'enabled': True,
            'stabilization': {
                'enabled': True,
                'iterations': 50,  # Reduced for faster rendering
                'updateInterval': 100,  # Less frequent updates
                'onlyDynamicEdges': False,
                'fit': True
            },
            'barnesHut': {
                'gravitationalConstant': -800,   # Reduced for less CPU load
                'centralGravity': 0.05,          # Less central pull
                'springLength': 120,             # Shorter springs
                'springConstant': 0.02,          # Weaker springs
                'damping': 0.2,                  # More damping for stability
                'avoidOverlap': 1.0              # Maximum overlap avoidance
            },
            'minVelocity': 0.1,
            'maxVelocity': 15,  # Reduced max velocity
            'timestep': 1.0,    # Larger timestep for stability
            'solver': 'barnesHut',
            'adaptiveTimestep': True
        },
        # Simple layout configuration
        layout={
            'randomSeed': 42,
            'improvedLayout': False,  # Disable for better performance
            'hierarchical': {
                'enabled': False
            }
        },
        # Optimized interaction for stability
        interaction={
            'dragNodes': True,
            'dragView': True,
            'zoomView': True,
            'selectConnectedEdges': True,
            'hover': True,
            'hoverConnectedEdges': False,  # Disable for performance
            'navigationButtons': True,
            'keyboard': {
                'enabled': False  # Disable keyboard for stability
            },
            'tooltipDelay': 500,  # Longer delay to reduce events
            'hideEdgesOnDrag': True,     # Hide edges during drag for performance
            'hideNodesOnDrag': False,
            'zoomViewOnClick': False,
            'multiselect': False         # Disable multiselect for stability
        },
        # Simplified node styling
        nodes={
            'borderWidth': 2,
            'borderWidthSelected': 3,
            'chosen': {
                'node': True,
                'label': False  # Disable label selection for performance
            },
            'font': {
                'size': 10,
                'color': '#ffffff',
                'face': 'Arial',
                'background': 'rgba(0,0,0,0.8)',
                'strokeWidth': 1,
                'strokeColor': '#000000',
                'align': 'center'
            },
            'shape': 'dot',
            'scaling': {
                'min': 15,
                'max': 35,
                'label': {
                    'enabled': True,
                    'min': 8,
                    'max': 12,
                    'maxVisible': 200,  # Increased for better visibility
                    'drawThreshold': 1   # Lower threshold
                }
            },
            'shadow': {
                'enabled': False  # Disable shadows for performance
            },
            'mass': 1
        },
        # Simplified edge styling
        edges={
            'smooth': {
                'enabled': True,
                'type': 'continuous',
                'roundness': 0.2  # Reduced for performance
            },
            'arrows': {
                'to': {
                    'enabled': True,
                    'scaleFactor': 0.8,  # Smaller arrows
                    'type': 'arrow'
                }
            },
            'color': {
                'inherit': False,
                'opacity': 0.6  # Reduced opacity
            },
            'font': {
                'size': 10,
                'strokeWidth': 0  # Remove stroke for performance
            },
            'scaling': {
                'min': 1,
                'max': 4  # Reduced range
            },
            'shadow': {
                'enabled': False  # Disable shadows for performance
            },
            'selectionWidth': 2,
            'hoverWidth': 1,
            'length': 120
        }
    )

    st.subheader("📊 Interactive Clustered Communication Network")

    # Add enhanced explanation
    st.info("🎯 **Fully Interactive Network**: Each color represents a topic cluster. "
            "**Drag nodes** to rearrange, **hover** for details, **pan & zoom** to explore. "
            "Solid edges = same topic, dashed = cross-topic communication.")

    # Performance metrics with better layout
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Nodes", G_filtered.number_of_nodes(), help="Email addresses displayed")
    with col2:
        st.metric("Edges", G_filtered.number_of_edges(), help="Communication links")
    with col3:
        density = nx.density(G_filtered) if G_filtered.number_of_nodes() > 0 else 0
        st.metric("Density", f"{density:.3f}", help="Network interconnectedness")
    with col4:
        st.metric("Clusters", len(topic_counts), help="Number of topic clusters")

    # Render graph with progress indication
    with st.spinner("🎨 Rendering interactive visualization..."):
        agraph(nodes=nodes, edges=edges, config=config)

    # Quick statistics
    with st.expander("📊 Network Statistics", expanded=False):
        if G_filtered.number_of_nodes() > 0:
            col1, col2 = st.columns(2)

            with col1:
                st.write("**Most Active Users (Sent + Received):**")
                activity_stats = [(node, data['sent'] + data['recv'])
                                for node, data in G_filtered.nodes(data=True)]
                activity_stats.sort(key=lambda x: x[1], reverse=True)
                for i, (user, activity) in enumerate(activity_stats[:5]):
                    st.write(f"{i+1}. {user.split('@')[0]}: {activity}")

            with col2:
                st.write("**Topic Distribution:**")
                for topic_id in sorted(topic_counts.keys()):
                    count = topic_counts[topic_id]
                    percentage = (count / len(nodes)) * 100
                    st.write(f"Topic {topic_id}: {count} users ({percentage:.1f}%)")

def main():
    st.title("🔍 Enron Email Explorer with Topic Clusters")
    st.markdown("Explore email communication patterns and topic clusters in the Enron dataset.")

    # WebSocket stability information
    st.sidebar.markdown("### 🔧 Stability Info")
    st.sidebar.info("If you see WebSocket errors in the console, they're harmless background noise. "
                   "To minimize them: refresh the page, reduce node count, or close/reopen the browser tab.")

    # Initialize session state
    if 'show_stats' not in st.session_state:
        st.session_state.show_stats = True

    # Performance settings in sidebar
    st.sidebar.header("⚙️ Performance Settings")
    st.sidebar.info("For best stability with large datasets, keep nodes under 300.")

    # Add memory usage indicator (optional)
    try:
        import psutil
        import os
        process = psutil.Process(os.getpid())
        memory_usage = process.memory_info().rss / 1024 / 1024  # MB
        st.sidebar.metric("Memory Usage", f"{memory_usage:.1f} MB")
    except ImportError:
        st.sidebar.info("Install psutil for memory monitoring: pip install psutil")
    except Exception:
        pass  # Ignore other errors

    # Load data with enhanced error handling
    try:
        start_time = time.time()
        df = load_data()
        load_time = time.time() - start_time

        st.sidebar.success(f"Data loaded in {load_time:.2f}s")
        st.sidebar.write(f"Total emails: {len(df):,}")

        # Topic clustering with timeout handling
        try:
            start_time = time.time()
            df, topics = show_topic_clustering(df)
            clustering_time = time.time() - start_time

            st.sidebar.success(f"Topics computed in {clustering_time:.2f}s")
        except Exception as e:
            st.error(f"❌ Error in topic clustering: {str(e)}")
            st.info("Try reducing the dataset size or refreshing the page.")
            return

        st.markdown("---")

        # Communication network with comprehensive error handling
        try:
            start_time = time.time()
            show_comm_network(df, topics)
            network_time = time.time() - start_time

            st.sidebar.success(f"Network rendered in {network_time:.2f}s")
        except Exception as e:
            st.error(f"❌ Error rendering network: {str(e)}")
            st.info("This may be due to WebSocket timeouts with large datasets. "
                   "Try reducing the number of nodes or refreshing the page.")

            # Offer alternative
            if st.button("🔄 Try with Reduced Dataset"):
                st.session_state.reduce_dataset = True
                st.rerun()

    except FileNotFoundError:
        st.error(f"❌ Data file not found: {PROCESSED_JSON_OUTPUT}")
        st.info("Please ensure the data file exists at the specified path.")

    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        st.info("💡 Try clearing the Streamlit cache or refreshing the page.")

        # Enhanced clear cache section
        st.markdown("### 🛠️ Troubleshooting")
        col1, col2 = st.columns(2)

        with col1:
            if st.button("🗑️ Clear Cache and Restart"):
                st.cache_data.clear()
                st.rerun()

        with col2:
            if st.button("📊 Reduce Dataset Size"):
                st.session_state.reduce_dataset = True
                st.rerun()

        # Display helpful information
        st.markdown("""
        **Common Solutions:**
        - Refresh your browser page
        - Clear browser cache
        - Reduce the number of nodes displayed
        - Close and reopen the browser tab
        - Restart the Streamlit app
        """)

        # Show system information for debugging
        with st.expander("🔍 System Information"):
            import platform
            st.write(f"**Python Version:** {platform.python_version()}")
            st.write(f"**Platform:** {platform.system()} {platform.release()}")
            try:
                import streamlit as st_version
                st.write(f"**Streamlit Version:** {st_version.__version__}")
            except:
                st.write("**Streamlit Version:** Unable to determine")

            # Show current session state
            st.write("**Session State:**")
            st.json(dict(st.session_state))

if __name__ == "__main__":
    main()