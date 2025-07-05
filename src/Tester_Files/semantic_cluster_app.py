import streamlit as st
import pandas as pd
import json
import os
from sentence_transformers import SentenceTransformer
import umap
import hdbscan
import networkx as nx
import plotly.express as px
import plotly.graph_objects as go
import requests
import sys
import numpy as np

from src.config.config import PROCESSED_JSON_OUTPUT,CACHED_CLUSTER_STORIES,EMBEDDING_CACHE_FILE

# Add config import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
os.environ["STREAMLIT_WATCH_USE_POLLING"] = "true"

# === CONFIGURATION ===
INPUT_JSON_PATH = PROCESSED_JSON_OUTPUT
CACHED_CLUSTER_STORIES = CACHED_CLUSTER_STORIES
EMBEDDING_CACHE_FILE = EMBEDDING_CACHE_FILE

st.set_page_config(page_title="Semantic Clustering of Emails", layout="wide")
st.title("📧 Semantic Clustering and Storytelling for Emails")

@st.cache_data
def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    df = pd.DataFrame(data) if isinstance(data, list) else pd.DataFrame([data])
    df['date'] = pd.to_datetime(df['date'], format="%d.%m.%Y %H:%M:%S", errors='coerce')
    return df

def get_cached_embeddings(texts, model_name='all-MiniLM-L6-v2', cache_path=EMBEDDING_CACHE_FILE):
    if os.path.exists(cache_path):
        print("✅ Loading cached embeddings...")
        return np.load(cache_path)
    else:
        print("🔄 Generating embeddings...")
        model = SentenceTransformer(model_name)
        embeddings = model.encode(texts, show_progress_bar=True)
        np.save(cache_path, embeddings)
        print("💾 Embeddings cached to:", cache_path)
        return embeddings

def summarize_with_ollama(text, model="mistral"):
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": model, "prompt": text, "stream": False},
            timeout=120
        )
        if response.ok:
            return response.json().get('response', '')
        else:
            return f"❌ Error from Ollama: {response.status_code} - {response.text}"
    except Exception as e:
        return f"❌ Ollama request failed: {str(e)}"

def generate_network_graph(email_subset, email_count):
    G = nx.Graph()
    for _, row in email_subset.iterrows():
        sender = row['from']
        recipients = row['to'] if isinstance(row['to'], list) else [row['to']]
        for r in recipients:
            if pd.notna(r):
                G.add_edge(sender, r)
        for org in row['entities'].get('organizations', []):
            G.add_edge(sender, org)
        for topic in row['entities'].get('topics', []):
            G.add_edge(sender, topic)
    pos = nx.spring_layout(G, k=1.5)
    edge_x, edge_y = [], []
    for e in G.edges():
        x0, y0 = pos[e[0]]
        x1, y1 = pos[e[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=0.5, color='#888'),
                            hoverinfo='none', mode='lines')
    node_x, node_y, node_text = [], [], []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(str(node))
    node_trace = go.Scatter(x=node_x, y=node_y, mode='markers+text', text=node_text,
                            textposition='top center',
                            marker=dict(size=10, color='blue'),
                            hoverinfo='text')
    fig = go.Figure(data=[edge_trace, node_trace])
    title = f"Entity Relationship Network with {email_count} emails"
    fig.update_layout(showlegend=False, margin=dict(t=20, l=5, r=5, b=5),
                      title=title)
    return fig

def process_and_cluster(df):
    df['text'] = df.apply(lambda row: f"{row.get('subject', '')} {row.get('summary', '')}", axis=1)
    embeddings = get_cached_embeddings(df['text'].tolist())
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine')
    reduced = reducer.fit_transform(embeddings)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=5)
    labels = clusterer.fit_predict(reduced)
    df['cluster'] = labels
    return df, reduced, labels

def generate_cluster_summaries(df):
    summaries = []
    for cluster_id in sorted(df['cluster'].unique()):
        if cluster_id == -1:
            continue
        cluster_df = df[df['cluster'] == cluster_id]
        sample_texts = cluster_df['text'].tolist()[:40]
        prompt = (
            f"You are an expert analyst. Read the following internal email excerpts from Cluster {cluster_id}. "
            f"First, generate a short, meaningful title that captures the main theme of the cluster. "
            f"Then, write a 2–4 sentence summary explaining what the emails are about.\n\n"
            + "\n".join(sample_texts) +
            "\n\nOutput in the following format:\n"
            "Title: <your title here>\n"
            "Summary: <your summary here>"
        )

        llm_output = summarize_with_ollama(prompt)
        title, summary = f"Cluster {cluster_id}", llm_output
        if "Title:" in llm_output and "Summary:" in llm_output:
            try:
                title = llm_output.split("Title:")[1].split("Summary:")[0].strip()
                summary = llm_output.split("Summary:")[1].strip()
            except Exception:
                pass

        summaries.append({
            'cluster_id': int(cluster_id),
            'email_count': len(cluster_df),
            'title': title,
            'summary': summary
        })

    with open(CACHED_CLUSTER_STORIES, 'w', encoding='utf-8') as f:
        json.dump(summaries, f, indent=2)
    return summaries

def load_cached_summaries():
    if os.path.exists(CACHED_CLUSTER_STORIES):
        with open(CACHED_CLUSTER_STORIES, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None

# === MAIN EXECUTION ===

df = load_json(INPUT_JSON_PATH)
st.success(f"✅ Loaded {len(df)} emails from `{INPUT_JSON_PATH}`")

df, reduced, labels = process_and_cluster(df)

st.subheader("📊 Cluster Visualization")
viz_df = pd.DataFrame(reduced, columns=["x", "y"])
viz_df['label'] = labels
viz_df['text'] = df['text']
fig = px.scatter(viz_df, x='x', y='y', color=viz_df['label'].astype(str),
                 hover_data=['text'], title="Semantic Clusters of Emails")
st.plotly_chart(fig, use_container_width=True)

st.subheader("📘 Cluster Titles + Stories")

# === Load or Generate Summaries ===
try:
    if os.path.exists(CACHED_CLUSTER_STORIES):
        summaries = load_cached_summaries()
        st.info("📂 Loaded from cached cluster summaries.")
    else:
        with st.spinner("🧠 Generating summaries using Mistral..."):
            summaries = generate_cluster_summaries(df)
        st.success("✅ Summaries generated and cached.")
except Exception as e:
    st.error(f"❌ Error while generating summaries: {e}")
    summaries = []

# === Render Output ===
# === Paginate and Render Summaries ===
if summaries:
    st.subheader("📑 Paginated Cluster Summaries")
    total_clusters = len(summaries)
    page_size = 5
    pages = list(range(1, (total_clusters // page_size) + 2))
    selected_page = st.selectbox("Choose page", pages, index=0)

    start = (selected_page - 1) * page_size
    end = min(start + page_size, total_clusters)
    selected_summaries = summaries[start:end]

    for story in selected_summaries:
        try:
            st.markdown(f"### {story['title']}")
            st.write(story['summary'])

            # Generate network graph (but handle disconnection errors)
            try:
                cluster_df = df[df['cluster'] == story['cluster_id']]
                net_fig = generate_network_graph(cluster_df, story['email_count'])
                st.plotly_chart(net_fig, use_container_width=True)
            except Exception as graph_err:
                st.warning(f"⚠ Could not render graph for cluster {story['cluster_id']}: {graph_err}")
        except Exception as render_err:
            st.error(f"⚠ Rendering failed for one summary: {render_err}")
else:
    st.warning("⚠ No summaries to display.")

