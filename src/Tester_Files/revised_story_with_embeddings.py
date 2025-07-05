
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
    clusterer = hdbscan.HDBSCAN(min_cluster_size=10, metric='euclidean')
    labels = clusterer.fit_predict(reduced)
    return reduced, labels

# ============ Cluster Merging ==============
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
            sim = cosine_similarity([embeddings_dict[k1]], [embeddings_dict[k2]])[0][0]
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
    prompt = f"""
You are an analyst reviewing a cluster of Enron emails. For the cluster below, generate:

1. "title": A compelling, informative title for the story.
2. "actors": A list of the key people or organizations involved in this theme.
3. "content": A short narrative (300–500 words) that explains what happened, when, and why it matters.

Emails:
{email_blocks}

Format your response as JSON with keys: "title", "actors", and "content".
"""
    response = query_ollama(prompt)
    try:
        return json.loads(response)
    except Exception:
        return {
            "title": f"{theme_title} (Parse Error)",
            "actors": [],
            "content": response
        }

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

    story_json_list = []
    with open("thematic_stories_output.txt", "w", encoding="utf-8") as f:
        for i, (label, cluster_emails) in enumerate(merged_clusters.items()):
            selected = sorted(cluster_emails, key=lambda e: parse_date(e.get("date", "")))[:MAX_EMAILS_PER_CLUSTER]
            theme = f"MajorTheme-{i}"
            print(f"🧠 Generating story for {theme} with {len(selected)} emails...")
            story = generate_theme_story(theme, selected)
            story_json_list.append(story)
            f.write(f"📚 {story['title']}\n{'='*60}\nActors: {', '.join(story['actors'])}\n\n{story['content']}\n\n\n")

    with open("structured_thematic_stories.json", "w", encoding="utf-8") as jf:
        json.dump(story_json_list, jf, indent=2)

    print("✅ Saved both structured JSON and text summaries.")

if __name__ == "__main__":
    main()
