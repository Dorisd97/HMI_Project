import streamlit as st
import json
import pandas as pd
import numpy as np
import re
import os
from pyvis.network import Network
import streamlit.components.v1 as components

# --- NLP and Machine Learning Imports ---
from sentence_transformers import SentenceTransformer
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import umap
import hdbscan
from sklearn.metrics.pairwise import cosine_similarity

# --- App Configuration ---
st.set_page_config(layout="wide", page_title="Enron Email Network Analysis")
st.title("🕵️‍♂️ Enron Email Network Explorer")
st.markdown("""
This app analyzes a collection of Enron emails to uncover hidden topics, key conversations, and the overall narrative.
1.  **Upload your data** or use the default `enron_full_analysis_results_100.json`.
2.  **Click 'Analyze Emails'** to process the data (this may take a few minutes).
3.  **Explore** the generated story and the interactive network graph.
""")


# --- Caching for Performance ---
# Cache the embedding model to avoid reloading it on every run
@st.cache_resource
def get_embedding_model():
    print("Loading embedding model...")
    return SentenceTransformer('all-MiniLM-L6-v2')


# Cache the LLM model
@st.cache_resource
def get_llm():
    print("Initializing LLM...")
    try:
        # Assumes Ollama is running with the 'mistral' model
        llm = Ollama(model="mistral")
        # Test connection
        llm.invoke("hello")
        return llm
    except Exception as e:
        st.error(f"Could not connect to local LLM (Ollama). Narrative generation will be skipped. Error: {e}")
        return None


# --- Analysis Functions (Adapted for Streamlit) ---

def run_full_analysis(df):
    """The main analysis pipeline function."""

    # --- Step 1: Embedding Generation ---
    with st.spinner("Generating semantic embeddings for emails..."):
        model = get_embedding_model()
        df['embedding_text'] = df['subject'].astype(str) + ' ' + df['summary'].astype(str)
        embeddings = model.encode(df['embedding_text'].tolist(), show_progress_bar=False)
        st.success("Embeddings generated!")

    # --- Step 2: Dimensionality Reduction & Clustering ---
    with st.spinner("Reducing dimensions with UMAP and finding clusters..."):
        reducer = umap.UMAP(n_neighbors=10, min_dist=0.05, metric='cosine', random_state=42)
        embeddings_2d = reducer.fit_transform(embeddings)
        df['x'] = embeddings_2d[:, 0]
        df['y'] = embeddings_2d[:, 1]

        clusterer = hdbscan.HDBSCAN(min_cluster_size=5, min_samples=1, metric='euclidean')
        df['cluster_id'] = clusterer.fit_predict(embeddings_2d)
        st.success("Clustering complete!")

    # --- Step 3: Story Extraction ---
    pivotal_emails = find_pivotal_emails(df, embeddings)

    llm = get_llm()
    if llm:
        with st.spinner("Generating narrative with AI..."):
            story = generate_narrative(pivotal_emails, llm)
            st.success("Narrative generated!")
    else:
        story = "LLM not available. Could not generate narrative."

    return df, story, pivotal_emails


def find_pivotal_emails(df, embeddings):
    """Identifies the most representative email for each cluster."""
    pivotal_emails = {}
    cluster_ids = df['cluster_id'].unique()

    for cluster_id in cluster_ids:
        if cluster_id == -1: continue
        cluster_indices = df[df['cluster_id'] == cluster_id].index
        cluster_embeddings = embeddings[cluster_indices]
        centroid = np.mean(cluster_embeddings, axis=0)
        similarities = cosine_similarity(cluster_embeddings, [centroid])
        original_df_index = cluster_indices[np.argmax(similarities)]
        pivotal_emails[cluster_id] = df.loc[original_df_index]

    return pivotal_emails


def generate_narrative(pivotal_emails, llm):
    """Uses an LLM to generate a story from pivotal email summaries."""
    sorted_pivots = sorted(pivotal_emails.values(), key=lambda x: x['date'])
    context_str = ""
    for i, email in enumerate(sorted_pivots):
        context_str += f"Event {i + 1} (Date: {email['date'].strftime('%Y-%m-%d')}): {email['summary']}\n\n"

    template = """
    You are a financial historian. Based on the following chronologically ordered email summaries from Enron, write a compelling, chapter-based narrative of the events. Connect the events logically.

    CONTEXT:
    {context}

    NARRATIVE:
    """
    prompt = PromptTemplate(template=template, input_variables=["context"])
    story_chain = LLMChain(llm=llm, prompt=prompt)
    response = story_chain.invoke({"context": context_str})
    return response['text']


# --- Visualization Function ---

def create_interactive_graph(df, pivotal_emails):
    """Creates an interactive graph using Pyvis."""
    st.subheader("Interactive Communication Network")

    net = Network(height="750px", width="100%", bgcolor="#222222", font_color="white", notebook=True,
                  cdn_resources='in_line')
    net.barnes_hut()  # Physics layout for better display

    # Create a color map for clusters
    unique_clusters = sorted(list(df['cluster_id'].unique()))
    colors = plt.cm.get_cmap('tab20', len(unique_clusters))
    color_map = {cluster_id: f'rgb({int(c[0] * 255)},{int(c[1] * 255)},{int(c[2] * 255)})' for cluster_id, c in
                 zip(unique_clusters, colors.colors)}
    if -1 in color_map:
        color_map[-1] = "#888888"  # Gray for outliers

    for index, row in df.iterrows():
        node_id = row['email_id']
        cluster_id = row['cluster_id']

        # Determine the title (hover text)
        title = f"""
        <b>Email ID:</b> {node_id}<br>
        <b>Date:</b> {row['date'].strftime('%Y-%m-%d')}<br>
        <b>From:</b> {row['from']}<br>
        <b>To:</b> {str(row['to'])[:100]}...<br>
        <b>Cluster:</b> {cluster_id}<br>
        <hr>
        <b>Summary:</b><br>{row['summary']}
        """

        # Get a label for the cluster
        if cluster_id != -1 and cluster_id in pivotal_emails:
            cluster_label = pivotal_emails[cluster_id]['classification']
        else:
            cluster_label = "Outlier"

        net.add_node(
            node_id,
            label=str(node_id),
            title=title,
            color=color_map[cluster_id],
            group=cluster_label,
            size=15
        )

    # Simplified edge creation for interactivity
    senders = set(df['from'])
    for index, row in df.iterrows():
        sender_email = row['from']
        if not isinstance(row['to'], str): continue
        recipients = [r.strip() for r in row['to'].split(',')]

        # Link sender to any recipient that is also a sender in the dataset
        for recip_email in recipients:
            if recip_email in senders:
                # Find the corresponding email_id for the recipient
                target_ids = df[df['from'] == recip_email]['email_id'].tolist()
                if target_ids:
                    net.add_edge(row['email_id'], target_ids[0], value=1)

    net.show_buttons(filter_=['physics', 'nodes', 'edges'])

    try:
        path = '/tmp'
        net.save_graph(f'{path}/pyvis_graph.html')
        HtmlFile = open(f'{path}/pyvis_graph.html', 'r', encoding='utf-8')
        components.html(HtmlFile.read(), height=800)
    except Exception as e:
        st.error(f"Could not generate or display the graph. Error: {e}")


# --- Main App Logic ---

# Sidebar for file upload and controls
with st.sidebar:
    st.header("Controls")
    uploaded_file = st.file_uploader(
        "Upload Enron JSON",
        type=['json'],
        help="Upload the 'enron_full_analysis_results_100.json' file."
    )

    analyze_button = st.button("🚀 Analyze Emails", type="primary")

# Initialize session state to hold results
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
    st.session_state.results_df = None
    st.session_state.story = ""
    st.session_state.pivotal_emails = {}

# --- Main Page Display ---

if analyze_button:
    if uploaded_file is not None:
        try:
            data = json.load(uploaded_file)
            df = pd.DataFrame(data['emails'])
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df.dropna(subset=['date'], inplace=True)

            st.info(f"Loaded {len(df)} emails from your file. Starting analysis...")

            # Run the heavy lifting
            df_processed, story, pivotal_emails = run_full_analysis(df)

            # Store results in session state
            st.session_state.analysis_complete = True
            st.session_state.results_df = df_processed
            st.session_state.story = story
            st.session_state.pivotal_emails = pivotal_emails

        except Exception as e:
            st.error(f"An error occurred during analysis: {e}")
            st.session_state.analysis_complete = False
    else:
        st.warning("Please upload a JSON file to begin analysis.")

# Display results if analysis is complete
if st.session_state.analysis_complete:
    st.header("📊 Analysis Results")

    # Display Story and Graph in Tabs
    tab1, tab2, tab3 = st.tabs(["📖 The Enron Story", "🕸️ Interactive Network Graph", "📈 Cluster Details"])

    with tab1:
        st.subheader("The Narrative of the Enron Emails")
        st.markdown(st.session_state.story)

    with tab2:
        if st.session_state.results_df is not None:
            create_interactive_graph(st.session_state.results_df, st.session_state.pivotal_emails)
        else:
            st.warning("No data to display.")

    with tab3:
        st.subheader("Summary of Discovered Clusters")
        for cluster_id, email_data in sorted(st.session_state.pivotal_emails.items()):
            with st.expander(f"**Cluster {cluster_id}: {email_data['classification']}**"):
                st.markdown(f"**Pivotal Email Summary:**")
                st.write(email_data['summary'])

        st.subheader("Raw Data with Cluster IDs")
        st.dataframe(st.session_state.results_df[['email_id', 'from', 'subject', 'cluster_id', 'classification']])