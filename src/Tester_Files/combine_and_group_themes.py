import re
import json
import ollama
from src.config.config import GENERATED_THEME_FULL_STORY_PATH, COMBINED_THEMES_PATH

INPUT_FILE = GENERATED_THEME_FULL_STORY_PATH
OUTPUT_FILE = COMBINED_THEMES_PATH


# Step 1: Load and parse the .txt file
def parse_themes(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Split on theme ID markers
    theme_raw_blocks = re.split(r"\n📚\s+Theme-(\d+)", content)
    stories = []

    for i in range(1, len(theme_raw_blocks), 2):
        theme_id = f"Theme-{theme_raw_blocks[i].strip()}"
        block = theme_raw_blocks[i + 1].strip()
        title_match = re.search(r"Title:\s*(.+)", block)
        title = title_match.group(1).strip() if title_match else "Untitled"
        stories.append({
            "theme_id": theme_id,
            "title": title,
            "story": block
        })

    return stories


# Step 2: Ask Mistral to group theme_ids and give a combined theme title for each group
def ask_mistral_to_group_themes(stories):
    theme_summaries = "\n".join([f"{s['theme_id']}: {s['title']}" for s in stories])
    prompt = f"""
You are an expert at identifying thematic similarities. Below is a list of themes from Enron's email dataset.
Each line has a Theme-ID and its title.

Your task is to:
1. Group similar themes based on shared topic, actors, or situation.
2. Give each group a short combined title.
3. Return ONLY JSON like this:

[
  {{
    "combined_theme": "Short title here",
    "themes": ["Theme-230", "Theme-330", "Theme-358"]
  }},
  ...
]

Here are the themes:
{theme_summaries}
"""
    response = ollama.chat(
        model='mistral',
        messages=[
            {"role": "system", "content": "You group and name related themes from email story titles."},
            {"role": "user", "content": prompt}
        ]
    )

    raw_text = response['message']['content']

    # Extract just the JSON part using a regex
    json_match = re.search(r"(\[\s*{.*?}\s*\])", raw_text, re.DOTALL)
    if not json_match:
        raise ValueError("Could not find JSON array in Mistral response.")

    json_data = json.loads(json_match.group(1))
    return json_data


# Step 3: Match theme IDs back to full content and structure final JSON
def build_grouped_json(grouped_ids, all_stories):
    story_dict = {s['theme_id']: s for s in all_stories}

    final_output = []
    for group in grouped_ids:
        theme_objects = []
        for tid in group['themes']:
            if tid in story_dict:
                theme_objects.append({
                    "theme_id": tid,
                    "title": story_dict[tid]['title'],
                    "story": story_dict[tid]['story']
                })
        final_output.append({
            "combined_theme": group['combined_theme'],
            "themes": theme_objects
        })
    return final_output


# MAIN
if __name__ == "__main__":
    print("📘 Parsing .txt file...")
    parsed_stories = parse_themes(INPUT_FILE)

    print("🧠 Asking Mistral to group related themes...")
    groupings = ask_mistral_to_group_themes(parsed_stories)

    print("🧩 Building structured output...")
    final_json = build_grouped_json(groupings, parsed_stories)

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(final_json, f, indent=2, ensure_ascii=False)

    print(f"✅ Done! JSON saved to: {OUTPUT_FILE}")
