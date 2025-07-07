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
CLEANED_BODYCHAIN_JSON_PATH = os.path.join(BASE_DIR, 'data', 'cleaned_body_chain_enron.json')
PROCESSED_JSON_OUTPUT = os.path.join(BASE_DIR, 'data', 'enron_full_analysis_results.json')
PROCESSED_JSON_OUTPUT_100 = os.path.join(BASE_DIR, 'data', 'enron_full_analysis_results_100.json')
GENERATED_THEME_STORY_PATH = os.path.join(BASE_DIR, 'src', 'Tester_Files', 'generated_thematic_story.json')
GENERATED_THEME_STORY_PATH2 = os.path.join(BASE_DIR, 'src', 'Tester_Files', 'enron_thematic_analysis_output1.json')
CACHED_STORIES_PATH = os.path.join(BASE_DIR, 'data', 'cached_stories.json')
EMBEDDING_CACHE_FILE = os.path.join(BASE_DIR, 'data', 'cached_embeddings.npy')
CACHED_CLUSTER_STORIES =  os.path.join(BASE_DIR, 'data', 'cached_cluster_stories.json')
THEMATIC_STORIES =  os.path.join(BASE_DIR, 'data', 'thematic_stories_output.txt')
AI_SUMMARY =  os.path.join(BASE_DIR, 'data', 'ai_summary.txt')
ENRON_LOGO = os.path.join(BASE_DIR, 'assets', 'enron_logo.png')
PICKLE_DIR = os.path.join(BASE_DIR, 'data', 'preprocessed_ui')

THEMATIC_STORIES_100 =  os.path.join(BASE_DIR, 'data', 'thematic_stories_output.txt')

