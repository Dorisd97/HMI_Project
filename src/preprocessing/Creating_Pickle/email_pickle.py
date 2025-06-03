# build_email_pickles.py

import os
import re
import json
import pickle
import logging
from datetime import datetime
from typing import List, Dict

import pandas as pd
from transformers import pipeline
from tqdm import tqdm

from src.config.config import CLEANED_JSON_PATH, PICKLE_FILE

# ─────────── LOGGER CONFIG ───────────
logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# ─────────── 0) CONFIGURE PATHS HERE ───────────
# Edit these paths (or set them via environment variables) before running.
RAW_JSON_PATH = CLEANED_JSON_PATH
OUTPUT_PICKLE = PICKLE_FILE

# ─────────── 0.5) HF CACHE LOCATION ───────────
# Change this to a folder on D: (or set via env var if you prefer).
HF_CACHE_DIR = os.environ.get("HF_CACHE_DIR", "D:/hf_cache")


# ─────────── 1) CLEANING HELPERS ───────────
def clean_email_content(content: str) -> str:
    """
    - Collapse multiple whitespace into single spaces.
    - Remove quoted‐reply artifacts (“On ... wrote:”).
    - Remove forwarded blocks (“-----Original Message-----”).
    - Strip HTML tags, URLs, email addresses.
    """
    if not content:
        return ""
    text = re.sub(r"\s+", " ", content.strip())
    text = re.sub(r"On .* wrote:", "", text)
    text = re.sub(r"-----Original Message-----.*", "", text, flags=re.DOTALL)
    text = re.sub(r"<[^>]+>", " ", text)              # strip HTML tags
    text = re.sub(r"https?://\S+", "", text)          # strip URLs
    text = re.sub(r"\S+@\S+\.\S+", "", text)          # strip email addresses
    return re.sub(r"\s+", " ", text).strip()


# ─────────── 2) LOAD & NORMALIZE JSON ───────────
def load_and_prepare_emails(path: str) -> List[Dict]:
    """
    1) Load raw JSON list of emails.
    2) Clean each body, parse "Date" → datetime.
    3) Skip if date cannot be parsed.
    4) Return a list of dicts: {EmailID, From, Subject, DateTime, Body}.
    """
    if not os.path.isfile(path):
        logger.error("JSON file not found: %s", path)
        raise FileNotFoundError(f"JSON file not found: {path}")

    logger.info("Loading raw JSON from %s", path)
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    normalized: List[Dict] = []
    skipped = 0

    for idx, e in enumerate(raw):
        # If a Message‐ID exists, use it; otherwise fallback to an index‐based ID
        email_id = e.get("Message-ID") or f"email_{idx}"

        # Get the body (“Body” or “Content”)
        raw_body = e.get("Body", "") or e.get("Content", "")
        body = clean_email_content(raw_body)

        # Parse date (try a few common formats)
        dt_obj = None
        dt_value = e.get("Date", "")
        for fmt in ("%d.%m.%Y %H:%M:%S", "%d.%m.%Y %H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
            try:
                dt_obj = datetime.strptime(dt_value, fmt)
                break
            except ValueError:
                continue

        if dt_obj is None:
            skipped += 1
            logger.warning("Skipping email %s: unparseable Date '%s'", email_id, dt_value)
            continue

        normalized.append({
            "EmailID": email_id,
            "From": e.get("From", "Unknown Sender"),
            "Subject": e.get("Subject", ""),
            "DateTime": dt_obj,
            "Body": body
        })

    normalized.sort(key=lambda x: x["DateTime"])
    logger.info("Prepared %d emails (skipped %d due to unparseable dates)", len(normalized), skipped)
    return normalized


# ─────────── 3) NLP PIPELINES ───────────
def build_nlp_pipelines():
    """
    1) sentiment‐analyzer (for “tone”)
       → Using a RoBERTa‐based model for better quality.
    2) question‐answerer (for “Why was this email was sent?”)
       → Using distilbert‐base‐uncased‐distilled‐squad.
    """
    # Ensure cache directory exists and is writable
    os.makedirs(HF_CACHE_DIR, exist_ok=True)
    logger.info("Loading NLP pipelines with cache_dir=%s", HF_CACHE_DIR)

    # 1) Sentiment‐Analysis
    sentiment_model = "siebert/sentiment-roberta-large-english"
    try:
        logger.info("About to load sentiment model: %s", sentiment_model)
        sentiment = pipeline(
            "sentiment-analysis",
            model=sentiment_model,
            tokenizer=sentiment_model,
            cache_dir=HF_CACHE_DIR
        )
        logger.info("Successfully loaded sentiment model: %s", sentiment_model)
    except Exception as e:
        logger.error("Failed to load sentiment model '%s': %s", sentiment_model, e)
        raise

    # 2) Question‐Answering
    qa_model = "distilbert-base-uncased-distilled-squad"
    try:
        logger.info("About to load QA model: %s", qa_model)
        qa = pipeline(
            "question-answering",
            model=qa_model,
            tokenizer=qa_model,
            cache_dir=HF_CACHE_DIR
        )
        logger.info("Successfully loaded QA model: %s", qa_model)
    except Exception as e:
        logger.error("Failed to load QA model '%s': %s", qa_model, e)
        raise

    logger.info("NLP pipelines loaded successfully")
    return sentiment, qa


def extract_first_sentence(text: str) -> str:
    """
    Return the first “sentence” (up to the first period/question/exclamation).
    If no punctuation, return the first ~200 characters as a fallback.
    """
    match = re.split(r"(?<=[\.\?\!])\s+", text.strip())
    if match and match[0]:
        return match[0].strip()
    return text[:200].strip()


def answer_why_sent(qa_pipeline, body: str) -> str:
    """
    Use a QA model to answer: “Why was this email sent?”
    Truncate context to 1 024 chars if body is longer.
    """
    if not body:
        return ""
    snippet = body if len(body) <= 1024 else body[:1024]
    try:
        out = qa_pipeline(question="Why was this email sent?", context=snippet)
        return out.get("answer", "").strip()
    except Exception as e:
        logger.warning("QA pipeline failed on snippet: %s", e)
        return ""


# ─────────── 4) MAIN: BUILD & PICKLE DATAFRAME ───────────
def build_and_pickle(email_json_path: str, output_pickle_path: str):
    logger.info("Starting build_and_pickle process")
    # 1) Load + clean all emails
    emails = load_and_prepare_emails(email_json_path)

    # 2) Build NLP pipelines (sentiment + QA)
    sentiment_pipe, qa_pipe = build_nlp_pipelines()

    logger.info("Beginning to process %d emails", len(emails))
    # 3) For each email, compute Purpose / Tone / TimelinePoint
    records = []
    for e in tqdm(emails, desc="Processing emails"):
        body = e["Body"]
        # 3a) Tone (sentiment) → feed up to 512 chars
        try:
            s = sentiment_pipe(body[:512])[0]
            tone = f"{s['label']} ({s['score']:.2f})"
        except Exception as ex:
            tone = "UNKNOWN"
            logger.warning("Sentiment analysis failed for EmailID %s: %s", e["EmailID"], ex)

        # 3b) Why was this email sent? (QA on first 1 024 chars)
        purpose = answer_why_sent(qa_pipe, body)

        # 3c) Timeline point → first sentence
        timeline_pt = extract_first_sentence(body)

        records.append({
            "EmailID":       e["EmailID"],
            "From":          e["From"],
            "Subject":       e["Subject"],
            "DateTime":      e["DateTime"],
            "Body":          body,
            "Purpose":       purpose,
            "Tone":          tone,
            "TimelinePoint": timeline_pt
        })

    # 4) Build DataFrame & pickle
    df = pd.DataFrame(records)
    df.sort_values("DateTime", inplace=True)
    df.reset_index(drop=True, inplace=True)

    os.makedirs(os.path.dirname(output_pickle_path), exist_ok=True)
    with open(output_pickle_path, "wb") as fout:
        pickle.dump(df, fout)
        logger.info(
            "Pickled DataFrame with shape %s → %s",
            df.shape,
            output_pickle_path
        )

    # 5) Final confirmation log
    logger.info("SUCCESS: Pickle file is now present at %s", output_pickle_path)


if __name__ == "__main__":
    try:
        logger.info("Starting email_pickle.py")
        build_and_pickle(RAW_JSON_PATH, OUTPUT_PICKLE)
        logger.info("Done building and pickling emails.")
    except Exception as exc:
        logger.error("ERROR in build_and_pickle: %s", exc)
        exit(1)
