import json
import ollama
import re
from src.config.config import GENERATED_THEME_FULL_STORY_PATH, COMBINED_THEMATIC_STORIES_PATH

# Path to your thematic stories file
INPUT_FILE = GENERATED_THEME_FULL_STORY_PATH
OUTPUT_FILE = COMBINED_THEMATIC_STORIES_PATH


# Step 1: Read and segment the thematic stories
def load_stories(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        content = file.read()

    story_blocks = re.split(r"\n📚\s+Theme-\d+", content)
    theme_ids = re.findall(r"📚\s+Theme-(\d+)", content)

    stories = []
    for i, block in enumerate(story_blocks[1:]):
        theme_id = theme_ids[i]
        story_text = block.strip()
        stories.append({'theme_id': f'Theme-{theme_id}', 'content': story_text})
    return stories


# Step 2: Use Ollama Mistral to group and merge similar stories
def group_and_merge(stories):
    story_chunks = "\n\n".join([f"[{story['theme_id']}]\n{story['content']}" for story in stories])

    prompt = f"""
You are an expert story analyst. I will give you multiple Enron email story summaries, each marked with a [Theme-ID] and its content. Your task is to:
- Group together stories that share similar actors, topics, or regulatory/financial/legal themes.
- For each group, generate:
    - a combined title
    - the merged themes (Theme-IDs)
    - a short summary of the unified narrative
    - a list of common actors/entities
    - a list of key topics

Return this as a JSON list.

Stories:
{story_chunks}
"""

    response = ollama.chat(
        model='mistral',
        messages=[
            {'role': 'system', 'content': 'You are a skilled analyst specialized in thematic story consolidation.'},
            {'role': 'user', 'content': prompt}
        ]
    )

    try:
        json_data = json.loads(response['message']['content'])
    except json.JSONDecodeError:
        raise ValueError("The model response could not be parsed as JSON.")

    return json_data


# Step 3: Write the result to a JSON file
def save_json(data, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"✅ Combined thematic stories saved to {path}")


# Main execution
if __name__ == "__main__":
    print("📖 Loading thematic stories...")
    stories = load_stories(INPUT_FILE)

    print("🧠 Combining similar stories using Ollama Mistral...")
    merged_stories = group_and_merge(stories)

    print("💾 Saving output as JSON...")
    save_json(merged_stories, OUTPUT_FILE)
