import streamlit as st
import pandas as pd
import networkx as nx
import plotly.express as px
import json

from src.config.config import PROCESSED_JSON_OUTPUT_100


# === LOAD DATA ===
@st.cache_data
def load_data():
    nodes = pd.read_csv("graph_nodes.csv")
    edges = pd.read_csv("graph_edges.csv")
    return nodes, edges

def load_full_emails():
    with open(PROCESSED_JSON_OUTPUT_100, "r", encoding="utf-8") as f:
        return json.load(f)["emails"]

# === PLOT NETWORK ===
def plot_clusters(nodes_df):
    fig = px.scatter(
        nodes_df, x="x", y="y",
        color=nodes_df["cluster"].astype(str),
        hover_data=["email_id", "from", "to", "date", "cluster"],
        title="📌 Email Clusters (HDBSCAN + PCA)",
        height=600
    )
    st.plotly_chart(fig, use_container_width=True)

# === DISPLAY DETAILS ===
def display_email_details(email_id, full_emails):
    match = next((e for e in full_emails if e["email_id"] == int(email_id)), None)
    if match:
        st.markdown(f"### ✉️ Subject: {match['subject']}")
        st.markdown(f"**From:** `{match['from']}`  \n**To:** `{match['to']}`  \n**Date:** `{match['date']}`")
        st.markdown(f"**Tone:** {match['tone_analysis']}  \n**Type:** {match['classification']}")
        st.markdown("#### Summary:")
        st.info(match["summary"])
        with st.expander("🔎 View Raw Entities"):
            st.json(match["entities"])
    else:
        st.warning("Email not found.")

# === STREAMLIT APP ===
def main():
    st.set_page_config(layout="wide")
    st.title("📧 Enron Email Network Analyzer")
    nodes, edges = load_data()
    full_emails = load_full_emails()

    st.sidebar.header("🔍 Filter Options")
    cluster_ids = sorted(nodes["cluster"].dropna().unique().astype(int))
    selected_cluster = st.sidebar.selectbox("Select Cluster", [-1] + cluster_ids, index=0)

    filtered_nodes = nodes[nodes["cluster"] == selected_cluster] if selected_cluster != -1 else nodes

    plot_clusters(filtered_nodes)

    st.subheader("📋 Email List")
    selected_email = st.selectbox("Select an email to view details", filtered_nodes["email_id"].astype(str))

    display_email_details(selected_email, full_emails)

    if st.button("🔄 Refresh"):
        st.experimental_rerun()

if __name__ == "__main__":
    main()
