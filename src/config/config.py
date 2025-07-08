import os

# Project root directory (two levels up from this file)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# Directory for all data files
DATA_DIR = os.path.join(BASE_DIR, 'data')
# Path to the original Enron zip file
ZIP_PATH = os.path.join(BASE_DIR, 'data', 'Enron.zip')
# Directory where the Enron data is unzipped
UNZIP_DIR = os.path.join(BASE_DIR, 'data', 'Enron_data')

# Directory for preprocessed data files
PREPROCESSED_DATA_DIR = os.path.join(BASE_DIR, 'data', 'preprocessed_data')
# Path to cleaned body chain JSON file
CLEANED_BODYCHAIN_JSON_PATH = os.path.join(BASE_DIR, 'data','preprocessed_data', 'cleaned_body_chain_enron.json')
# Path to cleaned Enron JSON file
CLEANED_JSON_PATH = os.path.join(BASE_DIR, 'data','preprocessed_data', 'cleaned_enron.json')
# Path to the main processed JSON output file
PROCESSED_JSON_OUTPUT = os.path.join(BASE_DIR, 'data','preprocessed_data', 'enron_full_analysis_results.json')

# Directory for visualization data files
VISUALIZATION_DATA_DIR = os.path.join(BASE_DIR, 'data', 'visualization_data')
# Path to cached embeddings numpy file for visualization
EMBEDDING_CACHE_FILE = os.path.join(BASE_DIR, 'data','visualization_data', 'cached_embeddings.npy')

# Directory for generated story files
GENERATED_STORIES_DIR = os.path.join(BASE_DIR, 'data', 'generated_stories')
# Path to cached cluster stories JSON file
CLUSTER_STORIES =  os.path.join(BASE_DIR, 'data','generated_stories', 'cached_cluster_stories.json')
# Path to cached stories JSON file
STORIES_PATH = os.path.join(BASE_DIR, 'data','generated_stories', 'cached_stories.json')
# Path to thematic stories output text file
THEMATIC_STORIES =  os.path.join(BASE_DIR, 'data', 'generated_stories', 'thematic_stories_output.txt')

# Path to log file for deleted duplicates
LOG_FILE_PATH = os.path.join(BASE_DIR, 'log', 'deleted_duplicates_log.txt')

# Path to Enron logo image asset
ENRON_LOGO = os.path.join(BASE_DIR, 'assets', 'enron_logo.png')

# Directory for preprocessed UI data (pickled files)
PICKLE_DIR = os.path.join(BASE_DIR, 'data', 'preprocessed_ui')
