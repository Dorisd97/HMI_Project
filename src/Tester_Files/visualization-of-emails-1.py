import pandas as pd
import numpy as np
import hdbscan
import networkx as nx
import json
from datetime import datetime

# === Load Email Embeddings and Metadata ===
NODES_PATH = "email_nodes.csv"  # From the previous step

def load_nodes():
    df = pd.read_csv(NODES_PATH)
    df["date"] = pd.to_datetime(df["date"], errors='coerce')
    return df

# === Cluster Emails using HDBSCAN ===
def cluster_emails(df, embeddings):
    clusterer = hdbscan.HDBSCAN(min_cluster_size=3, metric='euclidean')
    cluster_labels = clusterer.fit_predict(embeddings)
    df["cluster"] = cluster_labels
    return df, cluster_labels

# === Build Edge List from Sender/Receiver + Date Proximity ===
def build_edges(df, max_days_gap=5):
    G = nx.Graph()
    for _, row in df.iterrows():
        G.add_node(row["email_id"], **row.to_dict())

    for i, row1 in df.iterrows():
        for j, row2 in df.iloc[i+1:].iterrows():
            if row1["from"] == row2["from"] or row1["to"] == row2["to"]:
                date_diff = abs((row1["date"] - row2["date"]).days)
                if date_diff <= max_days_gap:
                    G.add_edge(row1["email_id"], row2["email_id"], weight=1.0)

    edges_df = pd.DataFrame([
        {"source": u, "target": v, "weight": d["weight"]}
        for u, v, d in G.edges(data=True)
    ])
    return edges_df

# === Save Output for Gephi or Visualization ===
def save_outputs(nodes_df, edges_df):
    nodes_df[["email_id", "from", "to", "date", "x", "y", "cluster"]].to_csv("graph_nodes.csv", index=False)
    edges_df.to_csv("graph_edges.csv", index=False)
    print("✅ Saved 'graph_nodes.csv' and 'graph_edges.csv'.")

# === Main ===
def main():
    df = load_nodes()
    embeddings = df[["x", "y"]].to_numpy()  # Already reduced via PCA

    print("Clustering emails...")
    df, _ = cluster_emails(df, embeddings)

    print("Building relationship edges...")
    edges_df = build_edges(df)

    print("Saving...")
    save_outputs(df, edges_df)

if __name__ == "__main__":
    main()
