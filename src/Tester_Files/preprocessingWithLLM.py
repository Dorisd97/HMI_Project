import json  # For reading and writing JSON files
from pathlib import Path  # For file path manipulations
from tqdm import tqdm  # For progress bars
import logging  # For logging progress and errors
from langchain_core.prompts import PromptTemplate  # For prompt templating
from langchain_community.llms import Ollama  # For LLM access via Ollama

# -------- CONFIG --------
INPUT_JSON = "D:/Coding_Projects/Git_Hub_Projects/HMI_Project/data/refined_enron_1_6.json"  # Input file path
OUTPUT_JSON = "D:/Coding_Projects/Git_Hub_Projects/HMI_Project/data/refined_enron_with_bodychain.json"  # Output file path
MODEL_NAME = "mistral"  # Try "mistral" if accuracy is more important
# ------------------------

# -------- LOGGING --------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),  # Log to console
        logging.FileHandler("../../log/bodychain_extraction.log", mode='w', encoding='utf-8')  # Log to file
    ]
)
logger = logging.getLogger(__name__)  # Get logger for this module
# ------------------------

# -------- LLM & PROMPT --------
logger.info(f"Loading model '{MODEL_NAME}' via Ollama...")
llm = Ollama(model=MODEL_NAME)  # Initialize Ollama LLM

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

chain = prompt | llm  # Compose prompt and LLM into a chain
# ------------------------

# -------- MAIN FUNCTION --------
def process_record(record, idx):
    body = record.get("Body", "")  # Get email body
    if not body.strip():
        logger.warning(f"Record {idx}: Empty body")
        record["BodyChain"] = []
        return record

    try:
        logger.info(f"Processing record {idx}")
        result = chain.invoke({"body": body})  # Run LLM chain

        # Handle result output structure
        if isinstance(result, str):
            raw_output = result
        elif hasattr(result, "content"):  # Some models return objects with .content
            raw_output = result.content
        else:
            raw_output = str(result)

        parsed = json.loads(raw_output.strip())  # Parse LLM output as JSON
        record["BodyChain"] = parsed.get("BodyChain", [])  # Attach BodyChain to record
        logger.info(f"Record {idx}: Extracted {len(record['BodyChain'])} chain(s)")
    except Exception as e:
        logger.error(f"Record {idx}: Failed to extract BodyChain: {e}")
        record["BodyChain"] = []

    return record

def process_json_file(input_path, output_path):
    logger.info(f"Reading input JSON from {input_path}")
    with open(input_path, 'r', encoding='utf-8') as f:
        records = json.load(f)  # Load all records

    logger.info(f"Found {len(records)} records")

    updated = []
    for idx, record in enumerate(tqdm(records, desc="Parsing email bodies")):
        updated.append(process_record(record, idx))  # Process each record

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)  # Ensure output directory exists
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(updated, f, indent=2, ensure_ascii=False)  # Write updated records
    logger.info(f"Completed. Output saved to {output_path}")
# ------------------------

if __name__ == "__main__":
    process_json_file(INPUT_JSON, OUTPUT_JSON)  # Run main function if script is executed
