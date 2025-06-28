import os
import re
import json
import pickle
import logging
from datetime import datetime
from typing import List, Dict

import torch
import pandas as pd
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForQuestionAnswering,
    AutoModelForSeq2SeqLM,
    pipeline
)
from tqdm import tqdm

# ─────────── Configure NLTK to use local resource folder ───────────
import nltk
from src.config.config import NLTK_FILE, HF_CACHE_DIR  # base directory for NLTK data

# Insert the folder where you placed punkt, stopwords, etc.
nltk.data.path.insert(0, NLTK_FILE)

# Verify “punkt” is available under that path; if not, raise an error
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    raise RuntimeError(
        f"NLTK could not find 'punkt' under the specified path: {NLTK_FILE}"
    )

# Verify “stopwords” is available under that path; if not, raise an error
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    raise RuntimeError(
        f"NLTK could not find 'stopwords' under the specified path: {NLTK_FILE}"
    )

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import networkx as nx
from itertools import combinations

# Load stopwords
stop_words = set(stopwords.words("english"))

from src.config.config import CLEANED_JSON_PATH, PICKLE_FILE

# ─────────── LOGGER CONFIG ───────────
logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# ─────────── 0) CONFIGURE PATHS HERE ───────────
RAW_JSON_PATH = CLEANED_JSON_PATH
OUTPUT_PICKLE = PICKLE_FILE

# ─────────── 0.5) HF CACHE LOCATION ───────────
HF_CACHE_DIR = os.environ.get("HF_CACHE_DIR", HF_CACHE_DIR)


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
        email_id = e.get("Message-ID") or f"email_{idx}"
        raw_body = e.get("Body", "") or e.get("Content", "")
        body = clean_email_content(raw_body)

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


# ─────────── 3) EXTRACTIVE TextRank HELPER ───────────
def textrank_extract(text: str, top_n: int = 12) -> List[str]:
    """
    Use a simple TextRank to extract the top_n most important sentences from `text`.
    Returns those sentences in their original order.
    """
    sentences = sent_tokenize(text)
    if len(sentences) <= top_n:
        return sentences

    def sent_to_set(s: str) -> set:
        words = word_tokenize(s)
        return {
            w.lower()
            for w in words
            if w.isalpha() and w.lower() not in stop_words
        }

    vectors = [sent_to_set(s) for s in sentences]
    G = nx.Graph()
    G.add_nodes_from(range(len(sentences)))

    for i, j in combinations(range(len(sentences)), 2):
        inter = vectors[i].intersection(vectors[j])
        union = vectors[i].union(vectors[j])
        if not union:
            continue
        sim = len(inter) / len(union)
        if sim > 0:
            G.add_edge(i, j, weight=sim)

    scores = nx.pagerank(G, weight="weight")
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_indices = sorted(idx for idx, _ in ranked[:top_n])
    return [sentences[i] for i in top_indices]


# ─────────── 4) NLP PIPELINES ───────────
def build_nlp_pipelines():
    """
    1) sentiment‐analyzer (for “tone”)          → GPU if available
    2) question‐answerer (for “Why was this email sent?”) → GPU if available
    3) tokenizer + seq2seq model for CodeT5 (no pipeline)
       so we can call `.generate(...)` directly and avoid conflicting kwargs.
    """
    # Check GPU
    cuda_ok = torch.cuda.is_available()
    logger.info("torch.cuda.is_available(): %s", cuda_ok)
    if cuda_ok:
        device = torch.device("cuda")
        logger.info("Using GPU: %s", torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        logger.info("No GPU found; pipelines will run on CPU")

    # Ensure cache directory exists
    os.makedirs(HF_CACHE_DIR, exist_ok=True)
    logger.info("Loading NLP pipelines with cache_dir=%s", HF_CACHE_DIR)

    # ─── (1) Sentiment‐Analysis ─────────────────────────────────────────────────
    sentiment_model = "siebert/sentiment-roberta-large-english"
    tokenizer_sent = AutoTokenizer.from_pretrained(
        sentiment_model,
        cache_dir=HF_CACHE_DIR
    )
    model_sent = AutoModelForSequenceClassification.from_pretrained(
        sentiment_model,
        cache_dir=HF_CACHE_DIR
    ).to(device)
    sentiment_pipe = pipeline(
        "sentiment-analysis",
        model=model_sent,
        tokenizer=tokenizer_sent,
        device=0 if cuda_ok else -1
    )
    logger.info("Successfully loaded sentiment model: %s", sentiment_model)

    # ─── (2) Question‐Answering ──────────────────────────────────────────────────
    qa_model = "distilbert-base-uncased-distilled-squad"
    tokenizer_qa = AutoTokenizer.from_pretrained(
        qa_model,
        cache_dir=HF_CACHE_DIR
    )
    model_qa = AutoModelForQuestionAnswering.from_pretrained(
        qa_model,
        cache_dir=HF_CACHE_DIR
    ).to(device)
    qa_pipe = pipeline(
        "question-answering",
        model=model_qa,
        tokenizer=tokenizer_qa,
        device=0 if cuda_ok else -1
    )
    logger.info("Successfully loaded QA model: %s", qa_model)

    # ─── (3) Summarization: load CodeT5 tokenizer + model for direct `.generate(...)` ──────────────────
    summarization_model = "Salesforce/codet5-base"
    logger.info("Loading summarization model: %s", summarization_model)

    tokenizer_sum = AutoTokenizer.from_pretrained(
        summarization_model,
        cache_dir=HF_CACHE_DIR
    )

    model_sum = AutoModelForSeq2SeqLM.from_pretrained(
        summarization_model,
        cache_dir=HF_CACHE_DIR
    ).to(device)

    logger.info("Successfully loaded CodeT5 model and tokenizer: %s", summarization_model)

    # Return everything we need (instead of a “pipeline” for summarization)
    return sentiment_pipe, qa_pipe, (tokenizer_sum, model_sum, device)


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


def summarize_body(code_t5_components, body: str) -> str:
    """
    Summarize the full body of an email in two stages with overlapping chunks,
    now by calling CodeT5’s `.generate(...)` directly:

    1) (Optional) Extract top sentences via TextRank (for >300 words).
    2) Tokenize and split into ~512-token chunks with 100-token overlap.
    3) Summarize each chunk by computing max_length/min_length manually and
       calling model.generate(...) directly (no pipeline, no early_stopping).
    4) Concatenate chunk summaries.
    5) Final summarization pass on concatenated text, again calling `.generate(...)`
       without `early_stopping`.
    """
    tokenizer_sum, model_sum, device = code_t5_components

    if not body:
        return ""

    # (1) Extractive pre-filter via TextRank if body is very long (>300 words)
    words = body.split()
    if len(words) > 300:
        try:
            key_sentences = textrank_extract(body, top_n=12)
            body = " ".join(key_sentences)
        except Exception as e:
            logger.warning("TextRank extraction failed: %s", e)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", body).strip()

    # (2) Tokenize entire text so we can create overlapping chunks
    all_tokens = tokenizer_sum.encode(text, add_special_tokens=False)
    max_tokens = 512            # CodeT5’s true encoder max
    overlap_tokens = 100

    chunks = []
    idx = 0
    total_tokens = len(all_tokens)
    while idx < total_tokens:
        end_idx = min(idx + max_tokens, total_tokens)
        chunk_tokens = all_tokens[idx:end_idx]
        chunk_text = tokenizer_sum.decode(chunk_tokens, clean_up_tokenization_spaces=False)
        chunks.append(chunk_text)
        step = max_tokens - overlap_tokens
        idx = end_idx if (end_idx <= idx + step) else idx + step

    # (3) Summarize each chunk by direct model.generate(...)
    chunk_summaries = []
    for chunk_text in chunks:
        if not isinstance(chunk_text, str):
            logger.warning("Chunk is not a string, skipping: %r", chunk_text)
            continue

        # Tokenize for input
        inputs = tokenizer_sum(
            chunk_text,
            return_tensors="pt",
            truncation=True,
            max_length=max_tokens
        ).to(device)

        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask
        input_len = input_ids.shape[-1]
        if input_len <= 20:
            # If chunk is very short, keep it as-is
            chunk_summaries.append(chunk_text.strip())
            continue

        # Decide how many new tokens we want for this chunk's summary:
        if input_len <= 200:
            desired_summary_tokens = max(20, int(input_len * 0.3))
        else:
            desired_summary_tokens = 100

        # Compute total max_length = input_len + desired_summary_tokens
        max_length = input_len + desired_summary_tokens
        # Ensure min_length < max_length:
        min_length = min(20, max_length - 1)

        try:
            # Directly call generate(), without early_stopping
            out_ids = model_sum.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_length,
                min_length=min_length,
                do_sample=False,
                no_repeat_ngram_size=2   # optional: encourage diversity
            )
            summary_text = tokenizer_sum.decode(out_ids[0], skip_special_tokens=True).strip()
            chunk_summaries.append(summary_text)
        except Exception as e:
            logger.warning("Chunk summarization failed: %s", e)
            chunk_summaries.append(chunk_text[:200].strip())

    if not chunk_summaries:
        return ""

    combined_summary = " ".join(chunk_summaries)

    # (4) Final summarization pass on the combined summary
    inputs = tokenizer_sum(
        combined_summary,
        return_tensors="pt",
        truncation=True,
        max_length=max_tokens
    ).to(device)

    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask
    input_len = input_ids.shape[-1]
    if input_len <= 20:
        return combined_summary.strip()

    if input_len <= 200:
        desired_summary_tokens = max(20, int(input_len * 0.3))
    else:
        desired_summary_tokens = 150

    max_length = input_len + desired_summary_tokens
    min_length = min(30, max_length - 1)

    try:
        out_ids = model_sum.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            min_length=min_length,
            do_sample=False,
            no_repeat_ngram_size=2
        )
        return tokenizer_sum.decode(out_ids[0], skip_special_tokens=True).strip()
    except Exception as e:
        logger.warning("Final summarization pass failed: %s", e)
        return combined_summary


# ─────────── 5) MAIN: BUILD & PICKLE DATAFRAME IN BATCHES OF 50 ───────────
def build_and_pickle(email_json_path: str, output_pickle_path: str, batch_size: int = 50):
    logger.info("Starting build_and_pickle process (with CodeT5 summaries)")
    # 1) Load + clean all emails
    all_emails = load_and_prepare_emails(email_json_path)

    # Only process the first 50 emails (single batch)
    emails_to_process = all_emails[:batch_size]
    logger.info("Processing first %d emails (batch size)", len(emails_to_process))

    # 2) Build NLP pipelines (sentiment + QA) and CodeT5 (tokenizer, model, device)
    sentiment_pipe, qa_pipe, code_t5_components = build_nlp_pipelines()

    # 3) Process this batch of emails
    records = []
    for e in tqdm(emails_to_process, desc="Processing emails"):
        body = e["Body"] or ""
        # 3a) Tone (sentiment) → feed up to 512 chars
        try:
            s = sentiment_pipe(body[:512])[0]
            tone = f"{s['label']} ({s['score']:.2f})"
        except Exception as ex:
            tone = "UNKNOWN"
            logger.warning("Sentiment analysis failed for EmailID %s: %s", e["EmailID"], ex)

        # 3b) Why was this email sent? (QA on first 1 024 chars)
        purpose = answer_why_sent(qa_pipe, body)

        # 3c) Body summary (extractive + overlapping-chunk summarization via CodeT5)
        body_summary = summarize_body(code_t5_components, body)

        records.append({
            "EmailID":     e["EmailID"],
            "From":        e["From"],
            "Subject":     e["Subject"],
            "DateTime":    e["DateTime"],
            "Body":        body,
            "Purpose":     purpose,
            "Tone":        tone,
            "BodySummary": body_summary
        })

    # 4) Build DataFrame & pickle
    df = pd.DataFrame(records)
    df.sort_values("DateTime", inplace=True)
    df.reset_index(drop=True, inplace=True)

    os.makedirs(os.path.dirname(output_pickle_path), exist_ok=True)
    with open(output_pickle_path, "wb") as fout:
        pickle.dump(df, fout)
        logger.info(
            "Pickled DataFrame (first %d emails) with shape %s → %s",
            len(df),
            df.shape,
            output_pickle_path
        )

    # 5) Final confirmation log
    logger.info("SUCCESS: Pickle file is now present at %s", output_pickle_path)


if __name__ == "__main__":
    try:
        logger.info("Starting email_pickle.py")
        build_and_pickle(RAW_JSON_PATH, OUTPUT_PICKLE, batch_size=50)
        logger.info("Done building and pickling first 50 emails.")
    except Exception as exc:
        logger.error("ERROR in build_and_pickle: %s", exc)
        exit(1)
