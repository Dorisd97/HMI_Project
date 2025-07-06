import json
import re
import ollama
from pathlib import Path

# File paths
INPUT_FILE = "thematic_stories_full_output.json"
OUTPUT_FILE = "combined_themes_output.json"

# Step 1: Load parsed JSON themes
def load_json_themes(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

# Step 2: Ask Mistral to group theme_ids and give a combined theme title for each group
def ask_mistral_to_group_themes(stories):
    theme_summaries = "\n".join([f"{s['theme_id']}: {s['title']}" for s in stories])
    prompt = f"""
You are an expert at identifying thematic similarities in documents.

You will receive a list of Enron themes in the format:
Theme-ID: Title

Your task is to:
1. Group similar themes that are about the same topic or incident.
2. Assign a short "combined_theme" title to each group.
3. Return ONLY a JSON array in this format (wrapped in triple backticks):

[
{{
"combined_theme": "Short shared title",
"themes": ["Theme-123", "Theme-456"]
}},
...
]

⚠️ DO NOT write an essay. DO NOT explain anything. ONLY return the JSON inside triple backticks.

Here are the themes:
{theme_summaries}
"""

    response = ollama.chat(
        model='mistral',
        messages=[
            {"role": "system", "content": "You group related themes from email story titles and return clean JSON only."},
            {"role": "user", "content": prompt}
        ]
    )

    raw_text = response['message']['content']

    print("\n🔍 Raw Mistral Response:\n" + "-" * 50)
    print(raw_text)
    print("-" * 50)

    # Optional: Save raw response for debugging
    with open("raw_mistral_response.txt", "w", encoding="utf-8") as f:
        f.write(raw_text)

    # Try extracting from triple backticks
    code_block_match = re.search(r"```(?:json)?\s*(\[\s*{.*?}\s*\])\s*```", raw_text, re.DOTALL)
    if code_block_match:
        return json.loads(code_block_match.group(1))

    # Fallback: match plain JSON array
    json_match = re.search(r"(\[\s*{.*?}\s*\])", raw_text, re.DOTALL)
    if json_match:
        return json.loads(json_match.group(1))

    raise ValueError("Could not find JSON array in Mistral response.")

# Step 3: Build final structured output
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
    print("📘 Loading thematic stories from JSON...")
    parsed_stories = load_json_themes(INPUT_FILE)

    print("🧠 Asking Mistral to group related themes...")
    groupings = ask_mistral_to_group_themes(parsed_stories)

    print("🧩 Building structured grouped output...")
    final_json = build_grouped_json(groupings, parsed_stories)

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(final_json, f, indent=2, ensure_ascii=False)

    print(f"✅ Done! Combined themes JSON saved to: {OUTPUT_FILE}")
