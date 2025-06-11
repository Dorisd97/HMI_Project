import streamlit as st
import json
import pandas as pd
import numpy as np
import networkx as nx
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

from streamlit_agraph import agraph, Config, Node, Edge

from src.config.config import PROCESSED_JSON_OUTPUT

st.set_page_config(page_title="Entity-Based Email Cluster Explorer", layout="wide")


@st.cache_data
def load_and_cluster():
    # Load JSON
    with open(PROCESSED_JSON_OUTPUT, 'r', encoding='utf-8') as f:
        data = json.load(f)
    df = pd.DataFrame(data)

    # Build TF-IDF docs from classification, subject, summary, and all entities
    docs = []
    for _, row in df.iterrows():
        parts = []
        if cl := row.get('classification'): parts.append(cl)
        if subj := row.get('subject'):      parts.append(subj)
        if summ := row.get('summary'):      parts.append(summ)
        ent = row.get('entities', {})
        for fld in ['people','organizations','locations','projects','legal','topics']:
            parts += ent.get(fld, [])
        docs.append(" ".join(parts))

    tfidf = TfidfVectorizer(max_features=200, stop_words='english')
    X     = tfidf.fit_transform(docs)

    # KMeans 3–7 clusters
    n_clusters = min(max(3, len(df)//10), 7)
    km         = KMeans(n_clusters=n_clusters, random_state=42).fit(X)
    clusters   = km.labels_

    # Cosine similarity
    sim_mat = cosine_similarity(X)
    return df, clusters, sim_mat


def compute_cluster_info(df, clusters):
    """Top‐5 topics per cluster for legend."""
    info = {}
    for cid in sorted(set(clusters)):
        idxs    = np.where(clusters == cid)[0]
        members = df.iloc[idxs]
        all_t   = sum((members.iloc[i]
                       .get('entities', {})
                       .get('topics', []) for i in range(len(members))), [])
        top5    = [t for t,_ in Counter(all_t).most_common(5)]
        info[cid] = top5
    return info


def build_graph(df, clusters, sim, threshold, top_k,
                cluster_choice=None, top_n=200):
    """Build full graph or drill‐down subgraph."""
    G = nx.Graph()
    N = len(df)

    # nodes
    for i, row in df.iterrows():
        G.add_node(i,
                   email_id=str(row.get('email_id', i)),
                   classification=row.get('classification',''),
                   cluster=int(clusters[i]),
                   people="; ".join(row.get('entities',{}).get('people',[])),
                   orgs="; ".join(row.get('entities',{}).get('organizations',[])),
                   locs="; ".join(row.get('entities',{}).get('locations',[])),
                   projs="; ".join(row.get('entities',{}).get('projects',[])),
                   legal="; ".join(row.get('entities',{}).get('legal',[])),
                   topics="; ".join(row.get('entities',{}).get('topics',[])),
                   date=row.get('date',''),
                   subject=row.get('subject','')[:50] + "…"
                   )

    # edges
    for i in range(N):
        sims = [(j, sim[i,j]) for j in range(N)
                if j!=i and sim[i,j] > threshold]
        sims.sort(key=lambda x: x[1], reverse=True)
        for j, s in sims[:top_k]:
            G.add_edge(i, j, weight=round(float(s),3))

    # drill‐down
    if cluster_choice is not None:
        nodes_in = [n for n,d in G.nodes(data=True)
                    if d['cluster']==cluster_choice]
        top_deg  = sorted(G.degree(nodes_in),
                          key=lambda x: x[1], reverse=True)[:top_n]
        sel      = {n for n,_ in top_deg}
        return G.subgraph(sel).copy()

    return G


def compute_cluster_summary(subG, total_n):
    """Metrics for a cluster subgraph."""
    size      = subG.number_of_nodes()
    perc      = size/total_n*100
    poss      = size*(size-1)/2
    dens      = subG.number_of_edges()/poss if poss>0 else 0
    degs      = dict(subG.degree())
    avg_deg   = sum(degs.values())/size if size>0 else 0
    top_node  = max(degs.items(), key=lambda x:x[1]) if degs else (None,0)
    return size, perc, dens, avg_deg, top_node


def main():
    st.title("📧 Entity-Based Email Cluster Explorer")
    st.markdown("Filter the network, drill into clusters, and view detailed analysis.")

    # Load & cluster
    df, clusters, sim_mat = load_and_cluster()
    cluster_info         = compute_cluster_info(df, clusters)

    # Sidebar controls + legend
    st.sidebar.header("Graph Controls")
    threshold     = st.sidebar.slider("Similarity threshold", 0.0, 1.0, 0.3, 0.01)
    top_k         = st.sidebar.slider("Max edges per node", 1, 10, 3)
    drill_cluster = st.sidebar.selectbox(
        "Drill into cluster (or None)",
        [None] + sorted(set(int(c) for c in clusters))
    )
    drill_n       = st.sidebar.slider("Top N nodes when drilling", 50, 500, 200, step=50)

    st.sidebar.markdown("### Cluster Topic Legend")
    palette = [
        "#FF6B6B", "#4ECDC4", "#45B7D1",
        "#FFA07A", "#98D8C8", "#FFD93D", "#6C5CE7"
    ]
    for cid, topics in cluster_info.items():
        st.sidebar.markdown(
            f"<span style='display:inline-block;"
            f"width:12px;height:12px;background:{palette[cid]};"
            f"margin-right:6px;border:1px solid #333;'></span>"
            f"Cluster {cid}: {', '.join(topics)}",
            unsafe_allow_html=True
        )

    # Build main graph
    G = build_graph(df, clusters, sim_mat,
                    threshold, top_k, drill_cluster, drill_n)

    # Prepare agraph nodes & edges
    nodes, edges = [], []
    for n,d in G.nodes(data=True):
        tooltip = (
            f"Email ID:</b> {d['to']}<br>"
            f"<b>Cluster:</b> {d['cluster']}<br>"
            f"<b>Classification:</b> {d['classification']}<br>"
            f"<b>Topics:</b> {d['topics']}<br>"
        )
        nodes.append(Node(
            id=str(n),
            label=d['email_id'],           # <-- label = email_id
            color=palette[d['cluster']],
            size=20,
            title=tooltip
        ))
    for u,v,d in G.edges(data=True):
        edges.append(Edge(
            source=str(u),
            target=str(v),
            color="#aaa",
            width=1 + d['weight']*4
        ))

    # Render main network
    config_main = Config(
        width=900, height=650,
        directed=False, focus=False,
        physics=True, hierarchical=False,
        nodeHighlightBehavior=True,
        highlightColor="#ffff00",
        collapse=False
    )
    st.subheader("🔗 Network Graph")
    st.write(f"Nodes: {G.number_of_nodes()} | Edges: {G.number_of_edges()}")
    agraph(nodes=nodes, edges=edges, config=config_main)

    # Detailed cluster analysis
    st.subheader("📑 Detailed Cluster Analysis")
    total_n = len(df)
    for cid in sorted(set(clusters)):
        with st.expander(f"Cluster {cid} — top topics: {', '.join(cluster_info[cid])}"):
            subG = build_graph(df, clusters, sim_mat,
                               threshold, top_k, cluster_choice=cid, top_n=100)
            size, perc, dens, avg_deg, (topn, topd) = compute_cluster_summary(subG, total_n)

            # Metrics
            c1, c2, c3, c4 = st.columns(4)
            with c1: st.metric("Size", f"{size} ({perc:.1f}%)")
            with c2: st.metric("Density", f"{dens:.2%}")
            with c3: st.metric("Avg degree", f"{avg_deg:.1f}")
            with c4: st.metric("Top node", f"{topn} ({topd})")

            # Classification breakdown
            idxs     = np.where(clusters==cid)[0]
            cls_dist = df.iloc[idxs]['classification'].value_counts()
            st.write("**Classification distribution**")
            st.bar_chart(cls_dist)

            # Date range
            dates = pd.to_datetime(df.iloc[idxs]['date'],
                                   dayfirst=True, errors='coerce')
            if not dates.isna().all():
                st.write(f"**Date range:** {dates.min().date()} → {dates.max().date()}")

            # Sample emails
            sample = df.iloc[idxs][['date','from','to','subject','summary']]
            st.write("**Emails in this cluster**")
            st.dataframe(sample.sort_values('date').reset_index(drop=True), height=250)

            # Mini‐network for this cluster
            sub_nodes, sub_edges = [], []
            for n,d in subG.nodes(data=True):
                sub_nodes.append(Node(
                    id=str(n),
                    label=d['email_id'],       # email_id label
                    color=palette[d['cluster']],
                    size=15
                ))
            for u,v,d in subG.edges(data=True):
                sub_edges.append(Edge(
                    source=str(u),
                    target=str(v),
                    color="#aaa",
                    width=1 + d['weight']*4
                ))
            sub_cfg = Config(
                width=600, height=400,
                directed=False, physics=True,
                nodeHighlightBehavior=True,
                highlightColor="#ffff00",
                collapse=False
            )
            st.write("**Cluster network (top 100 nodes)**")
            agraph(nodes=sub_nodes, edges=sub_edges, config=sub_cfg)


if __name__ == "__main__":
    main()
