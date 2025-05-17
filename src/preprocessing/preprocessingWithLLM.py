import json
from pathlib import Path
from tqdm import tqdm
import logging
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import Ollama

# -------- CONFIG --------
INPUT_JSON = "D:/Coding_Projects/Git_Hub_Projects/HMI_Project/data/refined_enron_1_6.json"
OUTPUT_JSON = "D:/Coding_Projects/Git_Hub_Projects/HMI_Project/data/refined_enron_with_bodychain.json"
MODEL_NAME = "mistral"  # Try "mistral" if accuracy is more important
# ------------------------

# -------- LOGGING --------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("bodychain_extraction.log", mode='w', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)
# ------------------------

# -------- LLM & PROMPT --------
logger.info(f"Loading model '{MODEL_NAME}' via Ollama...")
llm = Ollama(model=MODEL_NAME)

prompt = PromptTemplate.from_template("""
Parse the following email body. Extract any quoted or forwarded replies, and return them as a JSON array named "BodyChain".

Each entry must include:
- From
- To
- cc (if available)
- Subject
- Body

If no replies or forwards exist, return:
{{
  "BodyChain": []
}}

Email body:
\"\"\"
{body}
\"\"\"
""")

chain = prompt | llm
# ------------------------

# -------- MAIN FUNCTION --------
def process_record(record, idx):
    body = record.get("Body", "")
    if not body.strip():
        logger.warning(f"Record {idx}: Empty body")
        record["BodyChain"] = []
        return record

    try:
        logger.info(f"Processing record {idx}")
        result = chain.invoke({"body": body})

        # Handle result output structure
        if isinstance(result, str):
            raw_output = result
        elif hasattr(result, "content"):  # Some models return objects with .content
            raw_output = result.content
        else:
            raw_output = str(result)

        parsed = json.loads(raw_output.strip())
        record["BodyChain"] = parsed.get("BodyChain", [])
        logger.info(f"Record {idx}: Extracted {len(record['BodyChain'])} chain(s)")
    except Exception as e:
        logger.error(f"Record {idx}: Failed to extract BodyChain: {e}")
        record["BodyChain"] = []

    return record

def process_json_file(input_path, output_path):
    logger.info(f"Reading input JSON from {input_path}")
    with open(input_path, 'r', encoding='utf-8') as f:
        records = json.load(f)

    logger.info(f"Found {len(records)} records")

    updated = []
    for idx, record in enumerate(tqdm(records, desc="Parsing email bodies")):
        updated.append(process_record(record, idx))

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(updated, f, indent=2, ensure_ascii=False)
    logger.info(f"Completed. Output saved to {output_path}")
# ------------------------

if __name__ == "__main__":
    process_json_file(INPUT_JSON, OUTPUT_JSON)
