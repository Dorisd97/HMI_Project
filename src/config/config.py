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

