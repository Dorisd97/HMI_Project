import json
import re
import pandas as pd
from datetime import datetime
from collections import defaultdict, Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sentence_transformers import SentenceTransformer
import networkx as nx


class EmailThreadAnalyzer:
    def __init__(self, email_data):
        self.emails = email_data
        self.df = pd.DataFrame(email_data)
        self.threads = {}
        self.clusters = {}

    def clean_subject(self, subject):
        """Clean subject line for thread matching"""
        if not subject:
            return ""
        # Remove Re:, Fw:, etc.
        subject = re.sub(r'^(Re:|RE:|Fw:|FW:|Fwd:)\s*', '', subject, flags=re.IGNORECASE)
        # Remove extra whitespace
        subject = re.sub(r'\s+', ' ', subject).strip()
        return subject.lower()

    def group_by_threads(self):
        """Group emails into conversation threads"""
        threads = defaultdict(list)

        for email in self.emails:
            clean_subj = self.clean_subject(email.get('subject', ''))
            thread_key = clean_subj if clean_subj else f"no_subject_{email['email_id']}"
            threads[thread_key].append(email)

        # Sort each thread by date
        for thread_key in threads:
            threads[thread_key].sort(key=lambda x: datetime.strptime(x['date'], '%d.%m.%Y %H:%M:%S'))

        self.threads = dict(threads)
        return self.threads

    def analyze_thread_patterns(self):
        """Analyze communication patterns within threads"""
        thread_stats = []

        for thread_key, emails in self.threads.items():
            if len(emails) < 2:  # Skip single emails
                continue

            participants = set()
            for email in emails:
                participants.add(email['from'])
                participants.add(email['to'])

            # Calculate thread metrics
            duration_days = 0
            if len(emails) > 1:
                start_date = datetime.strptime(emails[0]['date'], '%d.%m.%Y %H:%M:%S')
                end_date = datetime.strptime(emails[-1]['date'], '%d.%m.%Y %H:%M:%S')
                duration_days = (end_date - start_date).days

            thread_stats.append({
                'thread_key': thread_key,
                'email_count': len(emails),
                'participant_count': len(participants),
                'duration_days': duration_days,
                'participants': list(participants),
                'classifications': [email['classification'] for email in emails],
                'start_date': emails[0]['date'],
                'end_date': emails[-1]['date']
            })

        return sorted(thread_stats, key=lambda x: x['email_count'], reverse=True)


class TopicClusterer:
    def __init__(self, email_data, use_embeddings=True):
        self.emails = email_data
        self.df = pd.DataFrame(email_data)
        self.use_embeddings = use_embeddings
        if use_embeddings:
            self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def create_classification_dashboard(self):
        """Create topic dashboard based on existing classifications"""
        classification_stats = {}

        # Basic classification distribution
        classifications = self.df['classification'].value_counts()

        for classification in classifications.index:
            class_emails = self.df[self.df['classification'] == classification]

            # Extract common entities for this classification
            all_entities = defaultdict(list)
            for _, email in class_emails.iterrows():
                entities = email['entities']
                for entity_type, entity_list in entities.items():
                    all_entities[entity_type].extend(entity_list)

            # Get top entities for each type
            top_entities = {}
            for entity_type, entity_list in all_entities.items():
                top_entities[entity_type] = Counter(entity_list).most_common(10)

            classification_stats[classification] = {
                'email_count': len(class_emails),
                'date_range': [class_emails['date'].min(), class_emails['date'].max()],
                'top_entities': top_entities,
                'sample_subjects': class_emails['subject'].head(5).tolist()
            }

        return classification_stats

    def semantic_clustering(self, n_clusters=8):
        """Perform semantic clustering using embeddings"""
        if not self.use_embeddings:
            return self.tfidf_clustering(n_clusters)

        # Combine subject and summary for better context
        texts = []
        for email in self.emails:
            text = f"{email.get('subject', '')} {email.get('summary', '')}"
            texts.append(text)

        # Generate embeddings
        print("Generating embeddings...")
        embeddings = self.model.encode(texts)

        # Cluster
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(embeddings)

        # Analyze clusters
        clusters = defaultdict(list)
        for i, label in enumerate(cluster_labels):
            clusters[label].append({
                'email': self.emails[i],
                'embedding': embeddings[i]
            })

        return self.analyze_clusters(clusters)

    def tfidf_clustering(self, n_clusters=8):
        """Fallback clustering using TF-IDF"""
        texts = []
        for email in self.emails:
            text = f"{email.get('subject', '')} {email.get('summary', '')}"
            texts.append(text)

        vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(texts)

        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(tfidf_matrix)

        clusters = defaultdict(list)
        for i, label in enumerate(cluster_labels):
            clusters[label].append({'email': self.emails[i]})

        return self.analyze_clusters(clusters)

    def analyze_clusters(self, clusters):
        """Analyze and label clusters"""
        cluster_analysis = {}

        for cluster_id, cluster_emails in clusters.items():
            emails = [item['email'] for item in cluster_emails]

            # Extract common themes
            all_words = []
            all_entities = defaultdict(list)
            classifications = []

            for email in emails:
                all_words.extend(email.get('summary', '').split())
                classifications.append(email['classification'])

                entities = email.get('entities', {})
                for entity_type, entity_list in entities.items():
                    all_entities[entity_type].extend(entity_list)

            # Find common terms and entities
            word_freq = Counter(all_words)
            top_words = [word for word, count in word_freq.most_common(10)
                         if len(word) > 3 and word.isalpha()]

            top_entities = {}
            for entity_type, entity_list in all_entities.items():
                top_entities[entity_type] = Counter(entity_list).most_common(5)

            # Generate cluster label
            main_classification = Counter(classifications).most_common(1)[0][0]
            cluster_label = f"{main_classification}"
            if top_entities.get('topics'):
                cluster_label += f" - {top_entities['topics'][0][0]}"

            cluster_analysis[cluster_id] = {
                'label': cluster_label,
                'size': len(emails),
                'top_words': top_words[:5],
                'top_entities': top_entities,
                'main_classification': main_classification,
                'sample_emails': emails[:3],
                'date_range': [
                    min(email['date'] for email in emails),
                    max(email['date'] for email in emails)
                ]
            }

        return cluster_analysis


def main():
    # Load your JSON data
    with open('enron_emails.json', 'r') as f:
        email_data = json.load(f)

    # Thread Analysis
    print("Analyzing email threads...")
    thread_analyzer = EmailThreadAnalyzer(email_data)
    threads = thread_analyzer.group_by_threads()
    thread_stats = thread_analyzer.analyze_thread_patterns()

    print(f"Found {len(threads)} conversation threads")
    print("Top 5 longest threads:")
    for stat in thread_stats[:5]:
        print(
            f"- {stat['thread_key'][:50]}... ({stat['email_count']} emails, {stat['participant_count']} participants)")

    # Topic Clustering
    print("\nPerforming topic clustering...")
    clusterer = TopicClusterer(email_data)

    # Classification dashboard
    classification_stats = clusterer.create_classification_dashboard()
    print(f"\nFound {len(classification_stats)} classification categories:")
    for classification, stats in classification_stats.items():
        print(f"- {classification}: {stats['email_count']} emails")

    # Semantic clustering
    cluster_analysis = clusterer.semantic_clustering(n_clusters=8)
    print(f"\nSemantic clustering results ({len(cluster_analysis)} clusters):")
    for cluster_id, analysis in cluster_analysis.items():
        print(f"- Cluster {cluster_id}: {analysis['label']} ({analysis['size']} emails)")


if __name__ == "__main__":
    main()
