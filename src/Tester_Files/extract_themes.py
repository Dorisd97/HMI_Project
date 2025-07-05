import json
import re
from src.config.config import GENERATED_THEME_FULL_STORY_PATH, GENERATED_THEME_FULL_STORY_JSON_PATH

# Path to your uploaded file
input_file = GENERATED_THEME_FULL_STORY_PATH
output_file = GENERATED_THEME_FULL_STORY_JSON_PATH

def extract_themes_from_txt(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Split each block by the unique pattern 📚 Theme-<id>
    blocks = re.split(r"📚 Theme-(\d+)", content)

    themes = []
    # Each theme will be in the format: ['', 'theme_id', 'block_content', 'theme_id2', 'block_content2', ...]
    for i in range(1, len(blocks) - 1, 2):
        theme_id = blocks[i].strip()
        block = blocks[i + 1].strip()

        # Extract the title
        title_match = re.search(r"Title:\s*(.+)", block)
        title = title_match.group(1).strip() if title_match else "Untitled"

        # Remove title and any line of '=' from story content
        story = re.sub(r"Title:.+\n", "", block)
        story = re.sub(r"=+\n", "", story).strip()

        themes.append({
            "theme_id": theme_id,
            "title": title,
            "story": story
        })

    return themes

# Extract and write to JSON
themes_json = extract_themes_from_txt(input_file)
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(themes_json, f, indent=2, ensure_ascii=False)

print(f"✅ Extracted {len(themes_json)} themes and saved to {output_file}")
