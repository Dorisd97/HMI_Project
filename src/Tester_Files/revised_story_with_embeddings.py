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
from sklearn.metrics.pairwise import cosine_similarity
from src.config.config import PROCESSED_JSON_OUTPUT

# ================= CONFIG ===================
INPUT_FILE = PROCESSED_JSON_OUTPUT
OLLAMA_URL = "http://localhost:11434"
MODEL_NAME = "mistral"
MAX_EMAILS_PER_CLUSTER = 10
CLUSTER_SIMILARITY_THRESHOLD = 0.85

# ============ Load & Clean Emails ==============
def load_emails(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data if isinstance(data, list) else data.get("emails", [])

def parse_date(date_str):
    try:
        return datetime.strptime(date_str, "%d.%m.%Y %H:%M:%S")
    except:
        return datetime.now()

# ============ Embedding Generation ==============
def generate_embeddings(emails):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    texts = [f"{e.get('subject', '')}. {e.get('summary', '')}" for e in emails]
    return model.encode(texts, show_progress_bar=True)

# ============ Clustering ==============
def cluster_embeddings(embeddings):
    reducer = umap.UMAP(n_neighbors=30, min_dist=0.4, metric='cosine')
    reduced = reducer.fit_transform(embeddings)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=25, metric='euclidean')
    labels = clusterer.fit_predict(reduced)
    return reduced, labels

# ============ Cluster Merging (Optional) ==============
def merge_similar_clusters(cluster_dict, embeddings_dict):
    cluster_keys = list(cluster_dict.keys())
    merged = {}
    merged_keys = set()

    for i, k1 in enumerate(cluster_keys):
        if k1 in merged_keys:
            continue
        merged[k1] = cluster_dict[k1]
        for k2 in cluster_keys[i+1:]:
            if k2 in merged_keys:
                continue
            sim = cosine_similarity(
                [embeddings_dict[k1]], [embeddings_dict[k2]]
            )[0][0]
            if sim > CLUSTER_SIMILARITY_THRESHOLD:
                merged[k1].extend(cluster_dict[k2])
                merged_keys.add(k2)
    return merged

# ============ Mistral LLM Query ==============
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
    email_blocks = "\n\n".join([
        f"Subject: {e['subject']}\nSummary: {e['summary']}" for e in cluster_emails
    ])
    prompt = f'''
You are an investigative journalist analyzing a collection of internal Enron emails.
Write a detailed, narrative-style report about the central financial, political, or legal issue reflected in these emails.

Focus on:
- Event chronology
- Key individuals and companies
- Market or regulatory impact
- Broader implications

Emails:
{email_blocks}

Conclude with a reflective summary about the consequences of these actions.
'''
    return query_ollama(prompt)

# ============ Main ==============
def main():
    emails = load_emails(INPUT_FILE)
    embeddings = generate_embeddings(emails)
    reduced, labels = cluster_embeddings(embeddings)

    cluster_dict = defaultdict(list)
    embeddings_dict = {}

    for idx, label in enumerate(labels):
        if label == -1:
            continue
        cluster_dict[label].append(emails[idx])

    for label, items in cluster_dict.items():
        idxs = [i for i, l in enumerate(labels) if l == label]
        emb_avg = np.mean([embeddings[i] for i in idxs], axis=0)
        embeddings_dict[label] = emb_avg

    merged_clusters = merge_similar_clusters(cluster_dict, embeddings_dict)

    theme_stories = {}
    for i, (label, cluster_emails) in enumerate(merged_clusters.items()):
        selected = sorted(cluster_emails, key=lambda e: parse_date(e.get("date", "")))[:MAX_EMAILS_PER_CLUSTER]
        theme = f"MajorTheme-{i}"
        print(f"🧠 Generating story for {theme} with {len(selected)} emails...")
        story = generate_theme_story(theme, selected)
        theme_stories[theme] = story

    with open("thematic_stories_output.txt", "w", encoding="utf-8") as f:
        for theme, story in theme_stories.items():
            f.write(f"📚 {theme}\n{'='*60}\n{story}\n\n\n")

    print("✅ Saved condensed thematic stories to 'thematic_stories_output.txt'")

if __name__ == "__main__":
    main()
