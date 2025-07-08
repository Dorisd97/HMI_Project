import json
import pickle
import os
import re
from collections import Counter
import pandas as pd

from src.config.config import PICKLE_DIR, THEMATIC_STORIES, CLUSTER_STORIES, STORIES_PATH

# Define the directory to save pickle files
PICKLE_DIR = PICKLE_DIR # Modify this to the correct path
if not os.path.exists(PICKLE_DIR):
    os.makedirs(PICKLE_DIR)

# Define file paths for your raw data (Modify this as needed)
THEMATIC_STORIES = THEMATIC_STORIES
CLUSTER_STORIES = CLUSTER_STORIES
STORIES_PATH = STORIES_PATH


# --- Function to generate required data ---
def generate_data():
    print("Generating data...")

    # --- Function 1: Process Conversation Data ---
    def process_conversation_data():
        print("Processing conversation data...")
        with open(CLUSTER_STORIES, 'r', encoding='utf-8') as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        df.rename(columns={'cluster_id': 'id', 'title': 'topic', 'summary': 'summary', 'email_count': 'email_count'},
                  inplace=True)

        output_path = os.path.join(PICKLE_DIR, 'conversation_df.pkl')
        with open(output_path, 'wb') as f:
            pickle.dump(df, f)
        print(f"✅ Saved conversation DataFrame to {output_path}")

    # --- Function 2: Process Narrative Data ---
    def process_narrative_data():
        print("Processing structured narrative data (this might be slow)...")
        with open(THEMATIC_STORIES, 'r', encoding='utf-8') as f:
            content = f.read()

        stories_raw = re.split(r'\n📚 Theme-\d+\n============================================================\n', content)
        stories_structured = {}
        for story_text in stories_raw:
            title_match = re.search(r'Title: (.*?)\n', story_text)
            if not title_match:
                continue
            title = title_match.group(1).strip()
            actors_section_match = re.search(r'Actors\s*:(.*?)(?:Story:|Summary:|The story begins)', story_text,
                                             re.DOTALL | re.IGNORECASE)
            actors = []
            if actors_section_match:
                actor_lines = actors_section_match.group(1).strip().split('\n')
                actors = [actor.strip().lstrip('- ').strip() for actor in actor_lines if actor.strip()]
            story_section_match = re.search(r'(Story:|Summary:|The story begins)(.*)', story_text,
                                            re.DOTALL | re.IGNORECASE)
            body = story_section_match.group(2).strip() if story_section_match else title
            stories_structured[title] = {'actors': actors, 'body': body}

        output_path = os.path.join(PICKLE_DIR, 'narratives_dict.pkl')
        with open(output_path, 'wb') as f:
            pickle.dump(stories_structured, f)
        print(f"✅ Saved narratives dictionary to {output_path}")

    # --- Function 3: Process Activity Spike Data ---
    def process_activity_spike_data():
        print("Processing activity spike data...")
        with open(STORIES_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
        activity_data = [item for item in data if item.get('type') == 'activity_burst']
        for item in activity_data:
            date_match = re.search(r'\d{4}-\d{2}-\d{2}', item['title'])
            item['date'] = date_match.group(0) if date_match else '2000-01-01'
        activity_data.sort(key=lambda x: x['date'])

        # We also need the spike map
        activity_spike_map = {spike['title']: spike for spike in activity_data}

        output_path_list = os.path.join(PICKLE_DIR, 'activity_spikes.pkl')
        output_path_map = os.path.join(PICKLE_DIR, 'activity_spike_map.pkl')

        with open(output_path_list, 'wb') as f:
            pickle.dump(activity_data, f)
        print(f"✅ Saved activity spikes list to {output_path_list}")

        with open(output_path_map, 'wb') as f:
            pickle.dump(activity_spike_map, f)
        print(f"✅ Saved activity spike map to {output_path_map}")

    # Run the processing functions
    process_conversation_data()
    process_narrative_data()
    process_activity_spike_data()

    print("\n🎉 All data has been generated and saved.")


# Now, the function is ready to be imported and used in the main app
