import json
import os
import pandas as pd
from datetime import datetime
from langchain.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA

from src.config.config import PROCESSED_JSON_OUTPUT_100

# === CONFIG ===
JSON_PATH = PROCESSED_JSON_OUTPUT_100
VECTOR_STORE_DIR = "faiss_index"
MODEL_NAME = "mistral"

# === STEP 1: Load and Preprocess Emails ===
def load_and_preprocess_emails(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)["emails"]

    processed_emails = []
    for email in data:
        try:
            timestamp = datetime.strptime(email["date"], "%d.%m.%Y %H:%M:%S").isoformat()
        except:
            timestamp = None

        content = f"{email['subject']}\n{email['summary']}\n{email.get('tone_analysis', '')}\n{email.get('classification', '')}"
        processed_emails.append({
            "email_id": email["email_id"],
            "from": email["from"],
            "to": email["to"],
            "date": timestamp,
            "content": content,
            "raw": email
        })

    return pd.DataFrame(processed_emails)

# === STEP 2: Generate Embeddings using LangChain + Ollama ===
def generate_embeddings(df):
    embedding_model = OllamaEmbeddings(model=MODEL_NAME)
    texts = df['content'].tolist()
    embeddings = embedding_model.embed_documents(texts)
    return embeddings

# === STEP 3: Reduce Embeddings (Optional) ===
def reduce_dimensions(embeddings, n_components=2):
    pca = PCA(n_components=n_components)
    reduced = pca.fit_transform(embeddings)
    return reduced

# === STEP 4: Save for Visualization ===
def save_visualization_ready_csv(df, reduced_coords, embeddings, output_path="email_nodes.csv"):
    df["x"] = reduced_coords[:, 0]
    df["y"] = reduced_coords[:, 1]
    df["embedding"] = embeddings
    df[["email_id", "from", "to", "date", "x", "y"]].to_csv(output_path, index=False)

# === Main ===
def main():
    print("Loading and preprocessing emails...")
    df = load_and_preprocess_emails(JSON_PATH)

    print("Generating embeddings with Mistral...")
    embeddings = generate_embeddings(df)

    print("Reducing dimensions with PCA...")
    reduced = reduce_dimensions(embeddings)

    print("Saving results...")
    save_visualization_ready_csv(df, reduced, embeddings)

    print("✅ Done. Output saved to 'email_nodes.csv'.")

if __name__ == "__main__":
    main()
