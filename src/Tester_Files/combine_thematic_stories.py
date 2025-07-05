import json
import ollama
import re
from src.config.config import GENERATED_THEME_FULL_STORY_PATH, COMBINED_THEMATIC_STORIES_PATH

# Path to your thematic stories file
INPUT_FILE = GENERATED_THEME_FULL_STORY_PATH
OUTPUT_FILE = COMBINED_THEMATIC_STORIES_PATH

# Step 1: Load thematic stories from file
def load_stories(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        content = file.read()

    story_blocks = re.split(r"\n📚\s+Theme-\d+", content)
    theme_ids = re.findall(r"📚\s+Theme-(\d+)", content)

    stories = []
    for i, block in enumerate(story_blocks[1:]):
        theme_id = f'Theme-{theme_ids[i]}'
        stories.append({'theme_id': theme_id, 'content': block.strip()})
    return stories

# Step 2: Prompt Ollama to combine similar stories into natural narratives
def generate_combined_stories(stories):
    combined_input = "\n\n".join([f"[{s['theme_id']}]\n{s['content']}" for s in stories])

    prompt = f"""
You are a narrative analyst. I will give you multiple short thematic stories from Enron emails, each marked with a [Theme-ID] and its content.

Your job is to read through all these themes and combine related ones into **natural, readable stories**. Each combined story should:
- Have a **strong, descriptive title**
- Mention important people, companies, or groups as part of the story
- Be written like a **brief but coherent narrative** — not a list, summary, or report
- Avoid bullet points or JSON structure — just output each story as plain text

Write about 5–10 combined stories depending on thematic overlap.

Here are the stories to combine:
{combined_input}
"""

    response = ollama.chat(
        model='mistral',
        messages=[
            {'role': 'system', 'content': 'You are a skilled narrative writer who combines factual event summaries into compelling short stories.'},
            {'role': 'user', 'content': prompt}
        ]
    )

    return response['message']['content']

# Step 3: Save as JSON with just one key
def save_story_text(text, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump({"combined_stories": text.strip()}, f, indent=2, ensure_ascii=False)
    print(f"✅ Combined narrative saved to {path}")

# Main runner
if __name__ == "__main__":
    print("📚 Loading thematic stories...")
    stories = load_stories(INPUT_FILE)

    print("✍️ Generating combined narratives using Mistral...")
    output = generate_combined_stories(stories)

    print("💾 Saving as JSON...")
    save_story_text(output, OUTPUT_FILE)
