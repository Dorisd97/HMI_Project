import json
from tqdm import tqdm
from langchain_ollama import OllamaLLM
import sys
import os

# Load config.py from correct relative path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.config import config

# Input and output paths from config
input_path = config.PROCESSED_JSON_OUTPUT_100  # e.g., 'data/enron_full_analysis_results_100.json'
output_path = config.GENERATED_THEME_STORY_PATH      # or define a new constant like GENERATED_STORY_PATH

# Step 1: Load Enron emails
with open(input_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

emails = data.get("emails", [])

# Step 2: Prepare summaries for LLM
email_summaries = "\n\n".join([
    f"""
📧 Email ID: {email.get('id', 'N/A')}
🗓 Date: {email.get('date', 'N/A')}
👤 From: {email.get('sender', 'Unknown')}
🏷 Subject: {email.get('subject', 'No Subject')}
📑 Summary: {email.get('summary', 'No Summary')}
🧠 Tone: {email.get('tone_analysis', 'N/A')}
📂 Classification: {email.get('classification', 'N/A')}
🔍 Entities:
  - People: {', '.join(email.get('entities', {}).get('people', []))}
  - Organizations: {', '.join(email.get('entities', {}).get('organizations', []))}
  - Locations: {', '.join(email.get('entities', {}).get('locations', []))}
  - Dates: {', '.join(email.get('entities', {}).get('dates', []))}
  - Projects: {', '.join(email.get('entities', {}).get('projects', []))}
  - Legal: {', '.join(email.get('entities', {}).get('legal', []))}
  - Topics: {', '.join(email.get('entities', {}).get('topics', []))}
""".strip()
for email in emails
])

# Step 3: Create LLM prompt
prompt = f"""
You are an expert AI narrative analyst.

You will be given summaries of internal corporate emails. Each includes:
- Metadata like sender, subject, tone, classification
- Extracted entity mentions (people, organizations, topics, legal, projects)
- Summarized content

🧩 Your task:
1. Analyze the combined dataset for patterns and context clues.
2. Infer the **dominant theme** across the emails (e.g., "Energy crisis and regulatory backlash", "Fraudulent accounting practices").
3. Write a **coherent, investigative-style story** (~200–300 words) about that theme, supported by events, tone, legal concerns, and involved people or organizations.

📛 Important:
- Do not invent information outside what is present in summaries or entities.
- Focus on internal operations, decisions, and consequences.

🎯 Final output must be a valid JSON like this:
{{
  "theme": "Energy crisis and SEC investigation",
  "story": "Your detailed narrative goes here. It should feel like a compelling, fact-based summary of the situation described in the emails."
}}

Here are the emails:
{email_summaries}
"""

# Step 4: Run Ollama LLM (Mistral)
llm = OllamaLLM(model="mistral", temperature=0.7)
response = llm.invoke(prompt)

# Step 5: Try parsing the output
try:
    result_json = json.loads(response)
except json.JSONDecodeError:
    print("⚠️ LLM output is not valid JSON. Here's what was returned:\n")
    print(response[:500])  # Trimmed for readability
    result_json = {"theme": "unknown", "story": response.strip()}

# Step 6: Save result safely
os.makedirs(os.path.dirname(output_path), exist_ok=True)

with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(result_json, f, indent=2)

print(f"\n✅ Thematic story written to: {output_path}")
