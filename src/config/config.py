import os

# Project root directory
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# Paths
ZIP_PATH = os.path.join(BASE_DIR, 'data', 'Enron.zip')
UNZIP_DIR = os.path.join(BASE_DIR, 'data', 'Enron_data')
LOG_FILE_PATH = os.path.join(BASE_DIR, 'log', 'deleted_duplicates_log.txt')
REFINED_CSV_PATH = os.path.join(BASE_DIR, 'data', 'refined_enron_emails.csv')
REFINED_JSON_PATH = os.path.join(BASE_DIR, 'data', 'refined_enron.json')
CLEANED_JSON_PATH = os.path.join(BASE_DIR, 'data', 'cleaned_enron.json')
EXTRACTED_ENTITIES_JSON_PATH = os.path.join(BASE_DIR, 'data', 'enron_entities.json')
#BODY_CHAIN_OUTPUT_PATH = os.path.join(BASE_DIR, 'data', 'processed_body_chain_output.json')

# Clustering and Analysis Paths
CLUSTERING_RESULTS_DIR = os.path.join(BASE_DIR, 'data', 'clustering_results')
CLUSTER_STORIES_PATH = os.path.join(CLUSTERING_RESULTS_DIR, 'cluster_stories.json')
EMAILS_WITH_CLUSTERS_PATH = os.path.join(CLUSTERING_RESULTS_DIR, 'emails_with_clusters.csv')
CLUSTER_SUMMARY_PATH = os.path.join(CLUSTERING_RESULTS_DIR, 'cluster_summary.json')

# Clustering Parameters
CLUSTERING_CONFIG = {
    'eps': 0.3,
    'min_samples': 5,
    'top_n_clusters': 5,
    'text_field': 'Body',  # Adjust based on your JSON structure
    'max_features': 1000,
    'ngram_range': (1, 2)
}
