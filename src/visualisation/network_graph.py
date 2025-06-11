import streamlit as st
import json
import pandas as pd
import numpy as np
import networkx as nx
import pydeck as pdk

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

from src.config.config import PROCESSED_JSON_OUTPUT

st.set_page_config(
    page_title="Email Network Analysis (GPU-Accelerated)",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def load_and_cluster():
    # 1) Load JSON
    with open(PROCESSED_JSON_OUTPUT, 'r', encoding='utf-8') as f:
        data = json.load(f)
    df = pd.DataFrame(data)

    # 2) Combine fields into one document per email
    docs = []
    for _, row in df.iterrows():
        parts = []
        if subject := row.get('subject'):
            parts.append(subject)
        if summary := row.get('summary'):
            parts.append(summary)
        parts.extend(row.get('entities', {}).get('topics', []))
        if cls := row.get('classification'):
            parts.append(cls)
        docs.append(" ".join(parts))

    # 3) TF-IDF + KMeans
    tfidf = TfidfVectorizer(max_features=200, stop_words='english')
    X = tfidf.fit_transform(docs)

    n_clusters = min(max(3, len(df)//10), 7)
    km = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = km.fit_predict(X)

    # 4) Cosine similarity matrix
    sim_mat = cosine_similarity(X)

    return df, clusters, sim_mat

def build_graph(df, clusters, sim_mat, threshold, top_k):
    G = nx.Graph()
    N = len(df)

    # Add nodes
    for i, row in df.iterrows():
        G.add_node(i,
                   email_id=row.get('email_id', i),
                   cluster=int(clusters[i]))

    # Add edges (top_k per node with sim > threshold)
    for i in range(N):
        sims = [(j, sim_mat[i, j]) for j in range(N) if j != i and sim_mat[i, j] > threshold]
        for j, score in sorted(sims, key=lambda x: x[1], reverse=True)[:top_k]:
            G.add_edge(i, j, weight=score)

    return G

def compute_layout(G, iterations):
    # GPU‐friendly 2D layout once on CPU
    return nx.spring_layout(G, seed=42, iterations=iterations)

def to_pydeck_data(G, pos):
    # Build node DataFrame
    nodes = pd.DataFrame([
        {
            "id": n,
            "x": pos[n][0],
            "y": pos[n][1],
            "cluster": G.nodes[n]["cluster"]
        }
        for n in G.nodes()
    ])

    # Build edge DataFrame
    edges = pd.DataFrame([
        {
            "sx": pos[u][0],
            "sy": pos[u][1],
            "tx": pos[v][0],
            "ty": pos[v][1],
            "weight": data["weight"]
        }
        for u, v, data in G.edges(data=True)
    ])

    return nodes, edges

def main():
    st.title("🚀 Email Network Analysis (GPU-Accelerated)")

    # Load + cluster once
    with st.spinner("Loading and clustering data…"):
        df, clusters, sim_mat = load_and_cluster()

    # Sidebar controls
    st.sidebar.header("Graph Controls")
    threshold = st.sidebar.slider("Similarity threshold", 0.0, 1.0, 0.3, 0.01)
    top_k      = st.sidebar.slider("Max edges per node",    1, 10, 3)
    iters      = st.sidebar.slider("Layout iterations",    10, 200, 50)

    # Build networkx graph
    with st.spinner("Building graph…"):
        G = build_graph(df, clusters, sim_mat, threshold, top_k)

    # Compute layout
    with st.spinner("Computing layout…"):
        pos = compute_layout(G, iters)

    # Convert to pydeck dataframes
    nodes_df, edges_df = to_pydeck_data(G, pos)

    # Summary metrics
    total = len(df)
    connections = G.number_of_edges()
    density = connections / (total * (total - 1) / 2) if total > 1 else 0
    st.markdown(
        f"**Total emails:** {total} | "
        f"**Connections:** {connections} | "
        f"**Network density:** {density:.2%}"
    )

    # Center view on mean position
    mid_x, mid_y = nodes_df["x"].mean(), nodes_df["y"].mean()
    view_state = pdk.ViewState(
        longitude=mid_x, latitude=mid_y,
        zoom=0, pitch=0
    )

    # Create GPU‐accelerated layers
    line_layer = pdk.Layer(
        "LineLayer",
        data=edges_df,
        get_source_position=["sx", "sy"],
        get_target_position=["tx", "ty"],
        get_color=[200, 200, 200],
        get_width="weight * 2",
        pickable=False
    )

    scatter_layer = pdk.Layer(
        "ScatterplotLayer",
        data=nodes_df,
        get_position=["x", "y"],
        get_fill_color="[cluster*30 % 255, (6-cluster)*30 % 255, 150]",
        get_radius=5,
        pickable=True
    )

    # Render deck.gl
    deck = pdk.Deck(
        layers=[line_layer, scatter_layer],
        initial_view_state=view_state,
        map_provider=None,
        tooltip={"text": "ID: {id}\nCluster: {cluster}"}
    )

    st.pydeck_chart(deck, use_container_width=True)

if __name__ == "__main__":
    main()
