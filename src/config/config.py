import os

# Project root directory
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# Paths
DATA_DIR = os.path.join(BASE_DIR, 'data')
ZIP_PATH = os.path.join(BASE_DIR, 'data', 'Enron.zip')
UNZIP_DIR = os.path.join(BASE_DIR, 'data', 'Enron_data')

# Preprocessed data paths
PREPROCESSED_DATA_DIR = os.path.join(BASE_DIR, 'data', 'preprocessed_data')
CLEANED_BODYCHAIN_JSON_PATH = os.path.join(BASE_DIR, 'data','preprocessed_data', 'cleaned_body_chain_enron.json')
CLEANED_JSON_PATH = os.path.join(BASE_DIR, 'data','preprocessed_data', 'cleaned_enron.json')
PROCESSED_JSON_OUTPUT = os.path.join(BASE_DIR, 'data','preprocessed_data', 'enron_full_analysis_results.json')


# Visualization data paths
VISUALIZATION_DATA_DIR = os.path.join(BASE_DIR, 'data', 'visualization_data')
EMBEDDING_CACHE_FILE = os.path.join(BASE_DIR, 'data','visualization_data', 'cached_embeddings.npy')

#Generated stories paths
GENERATED_STORIES_DIR = os.path.join(BASE_DIR, 'data', 'generated_stories')
CLUSTER_STORIES =  os.path.join(BASE_DIR, 'data','generated_stories', 'cached_cluster_stories.json')
STORIES_PATH = os.path.join(BASE_DIR, 'data','generated_stories', 'cached_stories.json')
THEMATIC_STORIES =  os.path.join(BASE_DIR, 'data', 'generated_stories', 'thematic_stories_output.txt')

# LOG and output paths
LOG_FILE_PATH = os.path.join(BASE_DIR, 'log', 'deleted_duplicates_log.txt')

# Assets paths
ENRON_LOGO = os.path.join(BASE_DIR, 'assets', 'enron_logo.png')

# Preprocessed UI data paths
PICKLE_DIR = os.path.join(BASE_DIR, 'data', 'preprocessed_ui')


