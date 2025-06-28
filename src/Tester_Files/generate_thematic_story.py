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
email_summaries = "\n\n".join(
    [f"Subject: {email['subject']}\nSummary: {email['summary']}" for email in emails]
)

# Step 3: Create LLM prompt
prompt = f"""
You are an investigative journalist AI with access to internal corporate emails.

Analyze the following email summaries and:
1. Identify the central *theme* (e.g., energy crisis, corporate fraud, market manipulation, merger conflict).
2. Write a coherent story narrative (200-300 words) about this theme using the provided email content.

Email Summaries:
{email_summaries}

Return a JSON object like:
{{
  "theme": "energy crisis and regulatory backlash",
  "story": "..."
}}
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
