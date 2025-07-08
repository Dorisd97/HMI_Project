import json
import requests
from collections import defaultdict
from datetime import datetime
from sentence_transformers import SentenceTransformer
import umap
import hdbscan
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from src.config.config import PROCESSED_JSON_OUTPUT, THEMATIC_STORIES

# ================= CONFIG ===================
INPUT_FILE = PROCESSED_JSON_OUTPUT
OLLAMA_URL = "http://localhost:11434"
MODEL_NAME = "mistral"
MAX_EMAILS_PER_CLUSTER = 5

# ============ Step 1: Load & Clean Emails ==============
def load_emails(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if isinstance(data, list):
        return data
    elif isinstance(data, dict):
        # case 1: JSON contains "emails" list
        if "emails" in data:
            return data["emails"]
        # case 2: JSON is a single email dictionary
        elif "subject" in data and "summary" in data:
            return [data]
        else:
            raise ValueError("Unknown email structure in JSON dict.")
    else:
        raise ValueError("Invalid JSON structure — must be a dict or list.")

def parse_date(date_str):
    try:
        return datetime.strptime(date_str, "%d.%m.%Y %H:%M:%S")
    except:
        return datetime.now()  # fallback if format is unexpected

# ============ Step 2: Generate Embeddings ==============
def generate_embeddings(emails):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    texts = [f"{e.get('subject', '')}. {e.get('summary', '')}" for e in emails]
    return model.encode(texts, show_progress_bar=True)

# ============ Step 3: Cluster Using UMAP + HDBSCAN ==============
def cluster_embeddings(embeddings):
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine')
    reduced = reducer.fit_transform(embeddings)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=5, metric='euclidean')
    labels = clusterer.fit_predict(reduced)
    return reduced, labels

# ============ Step 4: Use Mistral to Generate Thematic Stories ==============
def query_ollama(prompt):
    try:
        res = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": MODEL_NAME, "prompt": prompt, "stream": False},
            timeout=180
        )
        res.raise_for_status()
        return res.json()["response"]
    except Exception as e:
        return f"❌ Error calling Ollama: {e}"

def generate_theme_story(theme_title, cluster_emails):
    email_blocks = "\n\n".join(
        [f"Subject: {e['subject']}\nSummary: {e['summary']}" for e in cluster_emails]
    )
    prompt = f"""
You are an expert AI narrative analyst specializing in corporate investigations. Using the summaries of these Enron emails, do the following:

1. Infer the dominant theme across all the emails.
2. Write a cohesive investigative‐style **story** (about 200–300 words) that spans the entire cluster.
3. Structure your output as:
   Title: A concise, evocative title for the story.
   Actors: A bullet list of key people and organizations involved.
   Story: The narrative, laid out chronologically, highlighting major events, decisions, legal or regulatory concerns, and their evolution.
   Conclusion: A brief wrap-up of the outcome and lasting impact.

Suggested theme (for guidance): "{theme_title}"

Emails to analyze:
{email_blocks}
"""
    return query_ollama(prompt)



# ============ Step 5: Build and Visualize Network ==============
def build_network(theme_stories, cluster_map):
    G = nx.Graph()
    for theme, story in theme_stories.items():
        G.add_node(theme, type='theme')
        for email in cluster_map.get(theme, []):
            label = f"{email['email_id']} - {email['subject'][:30]}"
            G.add_node(label, type='email')
            G.add_edge(theme, label)
    return G

def draw_graph(G):
    pos = nx.spring_layout(G, seed=42)
    colors = ['skyblue' if G.nodes[n]['type'] == 'theme' else 'lightgray' for n in G.nodes]

    plt.figure(figsize=(14, 10))
    nx.draw(G, pos, with_labels=True, node_color=colors, node_size=2200, font_size=8)
    plt.title("📡 Theme–Email Network Graph")
    plt.show()

# ============ MAIN PIPELINE ==============
def main():
    print("🚀 Loading and preprocessing emails...")
    emails = load_emails(INPUT_FILE)
    embeddings = generate_embeddings(emails)
    reduced, labels = cluster_embeddings(embeddings)

    print("📌 Grouping emails by cluster...")
    cluster_dict = defaultdict(list)
    for idx, label in enumerate(labels):
        if label == -1:
            continue  # noise
        cluster_dict[label].append(emails[idx])

    print("🤖 Generating stories using Mistral...")
    theme_stories = {}
    cluster_map = {}

    for cluster_id, cluster_emails in cluster_dict.items():
        top_emails = sorted(cluster_emails, key=lambda e: parse_date(e.get("date", "")))[:MAX_EMAILS_PER_CLUSTER]
        theme_title = f"Theme-{cluster_id}"
        story = generate_theme_story(theme_title, top_emails)
        theme_stories[theme_title] = story
        cluster_map[theme_title] = top_emails
        print(f"\n📚 {theme_title}:\n{'-' * 60}\n{story[:600]}...\n")

    # Save stories to file
    with open(THEMATIC_STORIES, "w", encoding="utf-8") as out_file:
        for theme, story in theme_stories.items():
            out_file.write(f"📚 {theme}\n{'=' * 60}\n{story}\n\n\n")
    print("✅ All thematic stories saved to 'thematic_stories_output.txt'")

    print("🕸️ Building and displaying theme-email network...")
    G = build_network(theme_stories, cluster_map)
    draw_graph(G)

if __name__ == "__main__":
    main()
