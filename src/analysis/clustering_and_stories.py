import json
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
from collections import Counter
import os
import sys

# Add project root to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.config.config import *


class EmailClusteringAndStories:
    def __init__(self, use_entities=False):
        """
        Initialize clustering analysis

        Args:
            use_entities: If True, use extracted entities JSON, else use refined JSON
        """
        self.json_file_path = EXTRACTED_ENTITIES_JSON_PATH if use_entities else REFINED_JSON_PATH
        self.data = None
        self.df = None
        self.clusters = None
        self.dense_clusters = None
        self.config = CLUSTERING_CONFIG

    def load_data(self):
        """Load JSON data and convert to DataFrame"""
        print(f"Loading data from {self.json_file_path}...")

        if not os.path.exists(self.json_file_path):
            raise FileNotFoundError(f"Data file not found: {self.json_file_path}")

        with open(self.json_file_path, 'r') as f:
            self.data = json.load(f)

        # Convert to DataFrame
        self.df = pd.DataFrame(self.data)
        print(f"Loaded {len(self.df)} emails")
        return self.df

    def cluster_emails(self):
        """Perform clustering on email content using config parameters"""
        print("Clustering emails...")

        # Extract text for clustering
        text_field = self.config['text_field']
        if text_field not in self.df.columns:
            available_cols = list(self.df.columns)
            print(f"Warning: '{text_field}' not found. Available columns: {available_cols}")
            # Try common alternatives
            for alt in ['content', 'body', 'text', 'message', 'email_content']:
                if alt in available_cols:
                    text_field = alt
                    print(f"Using '{alt}' instead")
                    break
            else:
                raise ValueError(f"No suitable text field found. Available: {available_cols}")

        texts = self.df[text_field].fillna('').astype(str)

        # Create TF-IDF features
        vectorizer = TfidfVectorizer(
            max_features=self.config['max_features'],
            stop_words='english',
            ngram_range=self.config['ngram_range'],
            min_df=2
        )
        features = vectorizer.fit_transform(texts)

        # Perform DBSCAN clustering
        clustering = DBSCAN(
            eps=self.config['eps'],
            min_samples=self.config['min_samples']
        )
        self.clusters = clustering.fit_predict(features.toarray())

        # Add clusters to dataframe
        self.df['cluster'] = self.clusters

        # Get clustering statistics
        n_clusters = len(set(self.clusters)) - (1 if -1 in self.clusters else 0)
        n_noise = list(self.clusters).count(-1)

        print(f"Found {n_clusters} clusters with {n_noise} noise points")
        return self.clusters

    def find_dense_clusters(self):
        """Find the densest clusters using config parameters"""
        print("Finding dense clusters...")

        # Count emails per cluster (exclude -1 which is noise)
        cluster_counts = Counter([c for c in self.clusters if c != -1])

        if not cluster_counts:
            print("No valid clusters found!")
            return []

        # Get top N densest clusters
        top_n = self.config['top_n_clusters']
        self.dense_clusters = [cluster for cluster, count in cluster_counts.most_common(top_n)]

        print(f"Dense clusters: {self.dense_clusters}")
        print("Cluster sizes:", {c: cluster_counts[c] for c in self.dense_clusters})

        return self.dense_clusters

    def generate_stories(self, max_samples=3):
        """Generate stories for dense clusters"""
        print("Generating stories...")

        if not self.dense_clusters:
            return {}

        stories = {}
        text_field = self.config['text_field']

        # Find actual text field if config one doesn't exist
        if text_field not in self.df.columns:
            for alt in ['content', 'body', 'text', 'message', 'email_content']:
                if alt in self.df.columns:
                    text_field = alt
                    break

        for cluster_id in self.dense_clusters:
            cluster_emails = self.df[self.df['cluster'] == cluster_id]

            # Basic statistics
            num_emails = len(cluster_emails)

            # Get sample content
            samples = cluster_emails[text_field].head(max_samples).tolist()

            # Extract common words/entities
            all_text = ' '.join(cluster_emails[text_field].fillna('').astype(str))
            common_words = self._extract_common_words(all_text)

            # Try to get date range if available
            date_info = ""
            if 'date' in cluster_emails.columns:
                try:
                    dates = pd.to_datetime(cluster_emails['date'])
                    date_range = f"from {dates.min().strftime('%Y-%m-%d')} to {dates.max().strftime('%Y-%m-%d')}"
                    date_info = f"\n\n**Time period:** {date_range}"
                except:
                    pass

            # Create story
            story = f"""
# Cluster {cluster_id} Analysis

**Overview:** This cluster contains {num_emails} emails that share similar themes and content patterns.{date_info}

**Key themes:** {', '.join(common_words[:10]) if common_words else 'No significant themes identified'}

**Sample emails:**
"""

            for i, sample in enumerate(samples, 1):
                # Clean and truncate sample
                clean_sample = str(sample).replace('\n', ' ').strip()
                if len(clean_sample) > 300:
                    clean_sample = clean_sample[:300] + "..."
                story += f"\n**{i}.** {clean_sample}\n"

            stories[cluster_id] = story

        return stories

    def _extract_common_words(self, text, top_n=15):
        """Extract most common meaningful words from text"""
        if not text or not text.strip():
            return []

        words = text.lower().split()

        # Extended stop words list
        stop_words = {
            'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
            'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does',
            'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can',
            'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they',
            'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his', 'her', 'its', 'our', 'their',
            'a', 'an', 'from', 'up', 'about', 'into', 'through', 'during', 'before', 'after',
            'above', 'below', 'between', 'among', 'since', 'until', 'while', 'because',
            'if', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',
            'most', 'other', 'some', 'such', 'only', 'own', 'same', 'so', 'than', 'too', 'very'
        }

        # Filter words
        filtered_words = []
        for w in words:
            # Remove punctuation and check conditions
            clean_word = ''.join(c for c in w if c.isalnum())
            if len(clean_word) > 3 and clean_word not in stop_words and clean_word.isalpha():
                filtered_words.append(clean_word)

        word_counts = Counter(filtered_words)
        return [word for word, count in word_counts.most_common(top_n)]

    def save_results(self):
        """Save clustering results and stories to config paths"""
        print(f"Saving results to {CLUSTERING_RESULTS_DIR}...")

        # Create output directory
        os.makedirs(CLUSTERING_RESULTS_DIR, exist_ok=True)

        # Generate stories
        stories = self.generate_stories()

        # Save stories
        with open(CLUSTER_STORIES_PATH, 'w') as f:
            json.dump(stories, f, indent=2)

        # Save DataFrame with clusters
        self.df.to_csv(EMAILS_WITH_CLUSTERS_PATH, index=False)

        # Save cluster summary
        cluster_counts = Counter([c for c in self.clusters if c != -1])
        cluster_summary = {
            'total_clusters': len(set(self.clusters)) - (1 if -1 in self.clusters else 0),
            'noise_points': list(self.clusters).count(-1),
            'dense_clusters': self.dense_clusters,
            'cluster_sizes': {str(c): cluster_counts[c] for c in self.dense_clusters},
            'config_used': self.config
        }

        with open(CLUSTER_SUMMARY_PATH, 'w') as f:
            json.dump(cluster_summary, f, indent=2)

        print("Results saved!")
        return stories

    def run_full_pipeline(self):
        """Run the complete pipeline using config parameters"""
        print("=== Email Clustering and Story Generation Pipeline ===")

        try:
            # Step 1: Load data
            self.load_data()

            # Step 2: Cluster emails
            self.cluster_emails()

            # Step 3: Find dense clusters
            self.find_dense_clusters()

            # Step 4: Generate and save stories
            stories = self.save_results()

            print("\n=== Pipeline Complete! ===")
            print(f"Found {len(self.dense_clusters)} dense clusters")
            print(f"Generated {len(stories)} stories")

            return stories

        except Exception as e:
            print(f"Error in pipeline: {str(e)}")
            raise


# Usage example
if __name__ == "__main__":
    # Run clustering analysis
    analyzer = EmailClusteringAndStories(use_entities=False)  # Set to True to use entities JSON
    stories = analyzer.run_full_pipeline()

    # Print first story as example
    if stories:
        first_cluster = list(stories.keys())[0]
        print(f"\nExample story for cluster {first_cluster}:")
        print(stories[first_cluster])
