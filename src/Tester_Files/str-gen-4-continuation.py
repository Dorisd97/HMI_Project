import json
import streamlit as st
import networkx as nx
import plotly.graph_objects as go

@st.cache_data
def load_data(path="enron_visualization_data.json"):
    with open(path) as f:
        return json.load(f)

data = load_data()

nodes = data["visualization_data"]["nodes"]
edges = data["visualization_data"]["edges"]

# Sidebar filters
st.sidebar.header("Filters")
all_clusters = sorted({n["cluster_id"] for n in nodes})
selected_clusters = st.sidebar.multiselect(
    "Cluster ID", all_clusters, default=all_clusters
)

# Apply node filter
filtered_nodes = [n for n in nodes if n["cluster_id"] in selected_clusters]
filtered_ids = {n["id"] for n in filtered_nodes}
filtered_edges = [
    e for e in edges
    if e["source"] in filtered_ids and e["target"] in filtered_ids
]

# Build graph
G = nx.DiGraph()
for n in filtered_nodes:
    G.add_node(n["id"], **n)
for e in filtered_edges:
    G.add_edge(e["source"], e["target"], **e)

# Use the provided x,y as positions
pos = {n["id"]: (n["x"], n["y"]) for n in filtered_nodes}

# Edge traces
edge_x, edge_y = [], []
for u, v in G.edges():
    x0, y0 = pos[u]
    x1, y1 = pos[v]
    edge_x += [x0, x1, None]
    edge_y += [y0, y1, None]

edge_trace = go.Scatter(
    x=edge_x, y=edge_y,
    mode="lines",
    line=dict(width=1, color="#888"),
    hoverinfo="none"
)

# Node traces
node_x = [pos[n["id"]][0] for n in filtered_nodes]
node_y = [pos[n["id"]][1] for n in filtered_nodes]
node_text = [
    f"<b>{n['label']}</b><br>"
    f"Sender: {n['sender']}<br>"
    f"Date: {n['date']}<br>"
    f"<i>{n['summary'][:100]}...</i>"
    for n in filtered_nodes
]

node_trace = go.Scatter(
    x=node_x, y=node_y,
    mode="markers",
    marker=dict(size=12, line=dict(width=1, color="DarkSlateGrey")),
    hoverinfo="text",
    hovertext=node_text
)

fig = go.Figure(
    data=[edge_trace, node_trace],
    layout=go.Layout(
        title="Enron Email Communication Network",
        showlegend=False,
        hovermode="closest",
        margin=dict(b=20,l=5,r=5,t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=700
    )
)

st.plotly_chart(fig, use_container_width=True)

# Optionally show raw node table
if st.sidebar.checkbox("Show node table"):
    st.write(
        st.session_state.get("node_df") or
        st.write("Nodes", filtered_nodes)
    )
