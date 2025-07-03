import json
import pandas as pd
import numpy as np
import re

# NLP and Machine Learning Imports
from sentence_transformers import SentenceTransformer
from langchain_community.llms import Ollama  # Using a local LLM for example, you can swap with MistralAI, OpenAI, etc.
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import umap
import hdbscan
from sklearn.metrics.pairwise import cosine_similarity

from src.config.config import PROCESSED_JSON_OUTPUT_100

# --- CONFIGURATION ---
INPUT_FILE = PROCESSED_JSON_OUTPUT_100
OUTPUT_FILE = 'enron_visualization_data.json'
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'  # High-quality model that runs locally


# --- STEP 1: DATA INGESTION & PREPROCESSING ---

def load_and_preprocess(filepath):
    """Loads the JSON email dataset, cleans it, and prepares it for embedding."""
    print(f"Loading data from {filepath}...")
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    df = pd.DataFrame(data['emails'])

    # Ensure date is in datetime format for sorting
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df.dropna(subset=['date'], inplace=True)  # Drop emails with invalid dates

    # Create a single text field for embedding
    def clean_text(text):
        """A simple text cleaning function."""
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(r'\s+', ' ', text).strip()  # Normalize whitespace
        return text

    df['embedding_text'] = df['subject'].apply(clean_text) + ' ' + df['summary'].apply(clean_text)
    print(f"Data loaded and preprocessed. Total emails: {len(df)}")
    return df


# --- STEP 2: EMBEDDING GENERATION ---

def generate_embeddings(texts):
    """Generates sentence embeddings for a list of texts."""
    print(f"Generating embeddings using '{EMBEDDING_MODEL}'...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    embeddings = model.encode(texts.tolist(), show_progress_bar=True)
    print(f"Embeddings generated with shape: {embeddings.shape}")
    return embeddings


# --- STEP 3: RELATIONSHIP ANALYSIS (DIMENSIONALITY REDUCTION & CLUSTERING) ---

def reduce_dimensions(embeddings):
    """Reduces embedding dimensions using UMAP for 2D visualization."""
    print("Reducing dimensions with UMAP...")
    reducer = umap.UMAP(n_neighbors=10, min_dist=0.05, metric='cosine', random_state=42)
    embeddings_2d = reducer.fit_transform(embeddings)
    print("Dimensionality reduction complete.")
    return embeddings_2d


def cluster_emails(embeddings_2d):
    """Clusters emails using HDBSCAN to find thematic groups."""
    print("Clustering emails with HDBSCAN...")
    clusterer = hdbscan.HDBSCAN(min_cluster_size=5, min_samples=1, metric='euclidean', gen_min_span_tree=True)
    cluster_labels = clusterer.fit_predict(embeddings_2d)

    # -1 represents noise/outliers
    num_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    print(f"Clustering complete. Found {num_clusters} clusters and {np.sum(cluster_labels == -1)} outliers.")
    return cluster_labels


# --- STEP 4: STORY EXTRACTION ---

def find_pivotal_emails(df, embeddings):
    """Identifies the most representative email for each cluster (closest to centroid)."""
    print("Identifying pivotal emails for each cluster...")
    pivotal_emails = {}
    cluster_ids = df['cluster_id'].unique()

    for cluster_id in cluster_ids:
        if cluster_id == -1:  # Skip noise points
            continue

        cluster_indices = df[df['cluster_id'] == cluster_id].index
        cluster_embeddings = embeddings[cluster_indices]

        # Calculate the centroid of the cluster embeddings
        centroid = np.mean(cluster_embeddings, axis=0)

        # Find the email closest to the centroid
        similarities = cosine_similarity(cluster_embeddings, [centroid])
        most_representative_idx_in_cluster = np.argmax(similarities)

        # Get the original DataFrame index
        original_df_index = cluster_indices[most_representative_idx_in_cluster]

        pivotal_emails[cluster_id] = df.loc[original_df_index]

    print(f"Found {len(pivotal_emails)} pivotal emails.")
    return pivotal_emails


def generate_narrative(pivotal_emails):
    """Uses an LLM to generate a story from pivotal email summaries."""
    print("Generating narrative using LLM...")

    # Sort clusters by their average date to create a timeline
    sorted_pivots = sorted(pivotal_emails.values(), key=lambda x: x['date'])

    # Prepare the context for the LLM
    context_str = ""
    for i, email in enumerate(sorted_pivots):
        context_str += f"Event {i + 1} (Date: {email['date'].strftime('%Y-%m-%d')}):\n"
        context_str += f"Topic: {email['classification']}\n"
        context_str += f"Summary: {email['summary']}\n\n"

    # Set up LangChain with a local model (Ollama)
    # NOTE: You can replace this with any LangChain compatible LLM (MistralAI, OpenAI, Anthropic, etc.)
    # Make sure you have Ollama running with a model like 'mistral' or 'llama3'
    # ollama run mistral
    try:
        llm = Ollama(model="mistral")  # Assumes Ollama is running

        template = """
        You are a financial historian and investigative journalist specializing in corporate history.
        Based on the following chronologically ordered summaries of key email clusters from the Enron dataset, write a compelling, chapter-based narrative that tells the story of the events unfolding at Enron.

        Your story should have a clear title, chapters, and should connect the events logically to show the progression from business operations to crisis and collapse.

        Use the provided context to build your story. Do not invent facts, but interpret the connections between the events.

        CONTEXT:
        {context}

        NARRATIVE:
        """

        prompt = PromptTemplate(template=template, input_variables=["context"])
        story_chain = LLMChain(llm=llm, prompt=prompt)

        response = story_chain.invoke({"context": context_str})
        story = response['text']
        print("Narrative generation complete.")
        return story

    except Exception as e:
        print(f"Could not connect to local LLM (Ollama). Error: {e}")
        print("Skipping narrative generation. Please ensure Ollama is running with a model like 'mistral'.")
        return "Narrative generation failed. LLM not available."


# --- STEP 5: VISUALIZATION-READY OUTPUT ---

def create_visualization_output(df):
    """Creates a JSON file with nodes and edges for graph visualization."""
    print("Preparing visualization output...")
    nodes = []
    edges = []

    # Create nodes
    for index, row in df.iterrows():
        nodes.append({
            'id': row['email_id'],
            'label': f"Email {row['email_id']}",
            'summary': row['summary'],
            'date': row['date'].isoformat(),
            'sender': row['from'],
            'cluster_id': int(row['cluster_id']),
            'classification': row['classification'],
            'tone': row['tone_analysis'],
            'x': row['x'],  # UMAP coordinate
            'y': row['y'],  # UMAP coordinate
        })

    # Create edges based on sender -> recipient
    # This requires expanding the 'to' field
    for index, row in df.iterrows():
        sender = row['from']

        # Handle the 'to' field which can be a single string or a list of strings
        recipients_raw = row['to']
        if isinstance(recipients_raw, str):
            # Split comma-separated recipients
            recipients = [r.strip() for r in recipients_raw.split(',')]
        elif isinstance(recipients_raw, list):
            recipients = recipients_raw
        else:
            recipients = []

        # Create an edge if a recipient is also a sender in our dataset
        # This creates a more connected graph of internal communication
        all_senders = set(df['from'])
        for recipient in recipients:
            if recipient in all_senders:
                # Find the email_id of a message sent by this recipient
                # For simplicity, we link to the first one found, but more complex logic could be used
                target_ids = df[df['from'] == recipient]['email_id'].tolist()
                if target_ids:
                    # Create an edge from the current email to the first email sent by the recipient
                    edges.append({
                        'source': row['email_id'],
                        'target': target_ids[0],
                        'type': 'communication_flow'
                    })

    # Deduplicate edges
    unique_edges = [dict(t) for t in {tuple(d.items()) for d in edges}]

    return {'nodes': nodes, 'edges': unique_edges}


# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # 1. Load and preprocess data
    df = load_and_preprocess(INPUT_FILE)

    # 2. Generate embeddings
    embeddings = generate_embeddings(df['embedding_text'])

    # 3. Reduce dimensions and cluster
    embeddings_2d = reduce_dimensions(embeddings)
    df['x'] = embeddings_2d[:, 0]
    df['y'] = embeddings_2d[:, 1]

    cluster_labels = cluster_emails(embeddings_2d)
    df['cluster_id'] = cluster_labels

    # 4. Extract story
    pivotal_emails = find_pivotal_emails(df, embeddings)
    generated_story = generate_narrative(pivotal_emails)

    # ——— NEW: save narrative to a standalone .txt file ———
    narrative_path = 'generated_narrative.txt'
    with open(narrative_path, 'w', encoding='utf-8') as txtf:
        txtf.write(generated_story)
    print(f"Narrative saved to '{narrative_path}'")

    print("\n--- GENERATED NARRATIVE ---")
    print(generated_story)
    print("---------------------------\n")

    # 5. Create final structured output
    visualization_data = create_visualization_output(df)

    # Combine everything into a single final JSON
    final_output = {
        'analysis_metadata': {
            'embedding_model': EMBEDDING_MODEL,
            'clusters_found': len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0),
            'outliers_found': int(np.sum(cluster_labels == -1))
        },
        'generated_story': generated_story,
        'visualization_data': visualization_data,
        'original_emails': json.load(open(INPUT_FILE, 'r'))['emails']  # Include original data for context
    }

    # Save the file
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, indent=2, default=str)  # Use default=str to handle numpy types

    print(f"Analysis complete. Visualization-ready data saved to '{OUTPUT_FILE}'")