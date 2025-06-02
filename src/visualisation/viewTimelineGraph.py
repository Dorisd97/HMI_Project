# app.py

import os
import re
import json
import requests
import logging
import pandas as pd
import plotly.express as px
import streamlit as st
from math import ceil
from datetime import datetime
from typing import List, Dict
from streamlit_plotly_events import plotly_events
from src.config.config import CLEANED_JSON_PATH, HUGGING_FACE_API_KEY

# ─────────── LOGGER CONFIGURATION ───────────
logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# ─────────── Hugging Face Inference API Token ───────────
# Make sure you set your HF token in the environment:
#   export HUGGINGFACE_API_TOKEN="hf_XXXXXXXXXXXX"
HF_TOKEN = HUGGING_FACE_API_KEY

# ─────────── 1) PAGE CONFIG ───────────
st.set_page_config(layout="wide", page_title="Email Timeline & Abstractive Summary")

# ─────────── 2) UTILITY: CLEAN EMAIL CONTENT ───────────
def clean_email_content(content: str) -> str:
    """
    - Collapse multiple whitespace into a single space.
    - Strip quoted‐reply artifacts (“On ... wrote:”).
    - Remove forwarded blocks (“-----Original Message-----”).
    - Remove HTML tags, URLs, and email addresses.
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

# ─────────── 3) LOAD & NORMALIZE JSON ―───────────
@st.cache_data
def load_and_prepare(path: str) -> List[Dict]:
    """
    1) Load raw JSON (list of email dicts).
    2) Clean each body and parse the "Date" field into a Python datetime (field renamed to "DateTime").
    3) Skip any emails where "Date" cannot be parsed.
    4) Return a list of dicts with keys: From, Subject, DateTime, Body.
    """
    logger.info("Loading and preparing JSON from %s", path)
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    normalized: List[Dict] = []
    for e in raw:
        raw_body = e.get("Body", "") or e.get("Content", "")
        body = clean_email_content(raw_body)

        dt_value = e.get("Date")
        dt_obj = None
        if isinstance(dt_value, str):
            # Try multiple date formats
            for fmt in ("%d.%m.%Y %H:%M:%S", "%d.%m.%Y %H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
                try:
                    dt_obj = datetime.strptime(dt_value, fmt)
                    break
                except ValueError:
                    continue

        if dt_obj is None:
            logger.warning("Skipping email with unparseable Date: %s", dt_value)
            continue

        normalized.append({
            "From": e.get("From", "Unknown Sender"),
            "Subject": e.get("Subject", ""),
            "DateTime": dt_obj,
            "Body": body
        })

    normalized.sort(key=lambda x: x["DateTime"])
    logger.info("Finished preparing %d emails with valid dates", len(normalized))
    return normalized

# ─────────── 4) CHUNKING UTILITIES ───────────
def chunk_emails(emails: List[Dict], chunk_size: int = 50) -> List[List[Dict]]:
    """
    Split a list of email‐dicts into sublists each of length `chunk_size`.
    """
    n = len(emails)
    num_chunks = ceil(n / chunk_size)
    return [emails[i * chunk_size : (i + 1) * chunk_size] for i in range(num_chunks)]

def chunk_texts(texts: List[str], chunk_size: int = 10) -> List[List[str]]:
    """
    Split a list of strings (each ~100 words) into sublists each of length `chunk_size`.
    """
    n = len(texts)
    num_chunks = ceil(n / chunk_size)
    return [texts[i * chunk_size : (i + 1) * chunk_size] for i in range(num_chunks)]

# ─────────── 5) HUGGING FACE INFERENCE API HELPERS ───────────
def hf_summarize(text: str, model: str = "facebook/bart-large-cnn") -> str:
    """
    Send `text` to Hugging Face Inference API for summarization.
    Returns the "summary_text" from the response, or raises an error with details.
    """
    logger.info("Calling Hugging Face summarize endpoint (model=%s) with text length %d", model, len(text))
    api_url = f"https://api-inference.huggingface.co/models/{model}"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    payload = {
        "inputs": text,
        "parameters": {
            "max_length": 150,
            "min_length": 80,
            "do_sample": False
        }
    }
    response = requests.post(api_url, headers=headers, json=payload)
    if response.status_code != 200:
        # Log error and raise with details
        err_text = response.text
        logger.error("Hugging Face summarize failed (status %d): %s", response.status_code, err_text)
        raise RuntimeError(f"Hugging Face summarize error {response.status_code}: {err_text}")
    data = response.json()
    summary = data[0]["summary_text"].strip()
    logger.info("Received summary of length %d", len(summary))
    return summary

def hf_instructional_summarize(prompt: str, model: str = "facebook/bart-large-cnn") -> str:
    """
    Similar to hf_summarize, but used when sending an instructional prompt.
    Returns the "summary_text" or raises an error with details.
    """
    logger.info("Calling Hugging Face instructional summarize endpoint (model=%s) with prompt length %d", model, len(prompt))
    api_url = f"https://api-inference.huggingface.co/models/{model}"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_length": 200,
            "min_length": 120,
            "do_sample": False
        }
    }
    response = requests.post(api_url, headers=headers, json=payload)
    if response.status_code != 200:
        err_text = response.text
        logger.error("Hugging Face instructional summarize failed (status %d): %s", response.status_code, err_text)
        raise RuntimeError(f"Hugging Face instructional summarize error {response.status_code}: {err_text}")
    data = response.json()
    summary = data[0]["summary_text"].strip()
    logger.info("Received final instructional summary of length %d", len(summary))
    return summary

# ─────────── 6) BATCH‐LEVEL SUMMARIZATION ───────────
def summarize_one_batch(batch: List[Dict]) -> str:
    """
    Build a combined text for this batch:
      [Email 1: From … | Subject … | Date …]
      <body text>
    Then call hf_summarize(...) to get a ~4–5 sentence summary.
    If an error occurs, return a message containing the error details.
    """
    parts = []
    for i, email in enumerate(batch, start=1):
        sender = email["From"]
        subject = email["Subject"]
        dt_str = email["DateTime"].strftime("%Y-%m-%d %H:%M")
        header = f"[Email {i}: From {sender} | Subject: {subject} | Date: {dt_str}]\n"
        parts.append(header + email["Body"])

    combined = "\n\n".join(parts)
    try:
        summary = hf_summarize(combined)
        return summary
    except Exception as e:
        # Return the exception message so the UI can show it
        return f"⚠️ Error summarizing this batch: {str(e)}"

# ─────────── 7) FINAL SUMMARY WORKFLOW ───────────
def generate_one_paragraph_summary(emails: List[Dict]) -> str:
    """
    Multi‐stage abstractive summarization pipeline:
      1) Chunk emails into batches of 50, produce batch_summaries.
      2) Chunk batch_summaries into groups of 10, produce intermediate_summaries.
      3) Combine intermediate_summaries into mega_document.
      4) Send an instructional prompt to HF to cover:
         (1) key events, (2) tone shift, (3) crucial messages,
         (4) why it occurred, (5) conclusion.
      Return a single cohesive paragraph, or an error message if something fails.
    """
    logger.info("Starting final summarization pipeline on %d emails", len(emails))

    # 1) Batch-level summaries
    batches = chunk_emails(emails, chunk_size=50)
    batch_summaries: List[str] = []
    for i, batch in enumerate(batches, start=1):
        logger.info("Summarizing batch %d/%d (size=%d)", i, len(batches), len(batch))
        summary = summarize_one_batch(batch)
        batch_summaries.append(summary)

    # 2) Chunk batch_summaries into subchunks of 10
    summary_subchunks = chunk_texts(batch_summaries, chunk_size=10)

    # 3) Summarize each subchunk
    intermediate_summaries: List[str] = []
    for i, sub in enumerate(summary_subchunks, start=1):
        combined_sub = "\n\n".join(sub)
        logger.info("Summarizing intermediate subchunk %d/%d", i, len(summary_subchunks))
        try:
            interm = hf_summarize(combined_sub)
        except Exception as e:
            interm = f"⚠️ Error summarizing this subchunk: {str(e)}"
        intermediate_summaries.append(interm)

    # 4) Combine intermediate summaries into mega_document
    mega_document = "\n\n".join(intermediate_summaries)
    logger.info("Combined intermediate summaries into mega document of length %d", len(mega_document))

    # 5) Final instructional prompt
    prompt = f"""
Below are {len(intermediate_summaries)} intermediate summaries of email batches.
Please write a single, cohesive paragraph (4–6 sentences) that covers precisely:
  1. The key events that unfolded across the timeline.
  2. How the tone shifted over time (e.g., from collaborative to urgent to reconciliatory).
  3. The most crucial messages or decision points.
  4. Why this email exchange occurred in the first place.
  5. The final conclusion or status at the end.

Intermediate Summaries:
{mega_document}

One‐Paragraph Summary:
"""
    logger.info("Sending final instructional prompt (length=%d) to model", len(prompt))
    try:
        final_para = hf_instructional_summarize(prompt)
        logger.info("Received final paragraph summary")
    except Exception as e:
        # Return the exception message so the UI can display it
        final_para = f"❌ Error producing final summary: {str(e)}"
    return final_para

# ─────────── 8) STREAMLIT APP UI ───────────
all_emails = load_and_prepare(CLEANED_JSON_PATH)
df = pd.json_normalize(all_emails)

# --- Sidebar Filters ---
st.sidebar.header("🔍 Filters")

# Date range picker
valid_dates = df["DateTime"]
min_date = valid_dates.min().date()
max_date = valid_dates.max().date()
start_date, end_date = st.sidebar.date_input(
    "Date range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)

# Sender multiselect
senders = sorted(df["From"].unique())
selected_senders = st.sidebar.multiselect("Sender", senders, default=senders)

# Subject keyword filter
subject_kw = st.sidebar.text_input("Subject contains")

# Apply filters to DataFrame
mask = (
    df["DateTime"].dt.date.between(start_date, end_date) &
    df["From"].isin(selected_senders)
)
if subject_kw:
    mask &= df["Subject"].str.contains(subject_kw, case=False, na=False)

filtered_df = df[mask]

# --- 1) Interactive Timeline ---
st.subheader("📈 Select Emails on the Timeline")
fig = px.scatter(
    filtered_df,
    x="DateTime",
    y="From",
    hover_data=["Subject"],
    render_mode="webgl",
    title="Drag a box or lasso to select emails for summary",
    height=350
)
fig.update_traces(marker={"size": 8, "opacity": 0.7})
fig.update_layout(
    xaxis_title="Date & Time",
    yaxis_title="Sender",
    yaxis={"categoryorder": "array", "categoryarray": senders}
)
selected_points = plotly_events(fig, select_event=True, override_height=350)

# --- 2) Build Working Set (at Least 1 Email) ---
if selected_points:
    idxs = [pt["pointIndex"] for pt in selected_points]
    if len(idxs) < 1:
        idxs = [filtered_df.index.min()]
        st.warning("Added the earliest email so we have at least one.")
    working_df = filtered_df.loc[idxs].sort_values("DateTime")
    st.success(f"✅ {len(idxs)} emails selected for summary")
else:
    st.info("No selection—using the earliest email.")
    working_df = filtered_df.head(1)

# --- 3) Combined Abstractive Summary ---
st.subheader("✍️ One‐Paragraph Abstractive Summary")
if st.button("Generate Crystal‐Clear Summary"):
    email_records = working_df.to_dict("records")
    with st.spinner("🛠️ Generating summary—this may take a minute..."):
        final_summary = generate_one_paragraph_summary(email_records)
    st.markdown("**📜 Final One‐Paragraph Summary:**")
    # Now the `final_summary` either contains the paragraph or a detailed error
    st.markdown(final_summary)

# --- 4) Charts & Visualizations (Optional) ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Daily Email Volume")
    if not filtered_df.empty:
        daily_counts = (
            filtered_df
            .set_index("DateTime")
            .resample("D")
            .size()
            .rename("Count")
            .reset_index()
        )
        fig_line = px.line(
            daily_counts,
            x="DateTime",
            y="Count",
            title="Number of Emails per Day",
            height=300
        )
        fig_line.update_layout(xaxis_title="Date", yaxis_title="Email Count")
        st.plotly_chart(fig_line, use_container_width=True)

with col2:
    st.subheader("👥 Sender Distribution")
    if not working_df.empty:
        sender_counts = working_df["From"].value_counts().head(10)
        fig_pie = px.pie(
            values=sender_counts.values,
            names=sender_counts.index,
            title="Selected Emails by Sender",
            height=300
        )
        st.plotly_chart(fig_pie, use_container_width=True)

# --- 5) Heatmap: Hour vs Day of Week ---
st.subheader("🔥 Email Traffic Heatmap")
if not filtered_df.empty:
    heat_data = (
        filtered_df
        .assign(
            DayOfWeek=filtered_df["DateTime"].dt.day_name(),
            Hour=filtered_df["DateTime"].dt.hour
        )
        .groupby(["DayOfWeek", "Hour"])
        .size()
        .rename("Count")
        .reset_index()
    )
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    heat_data["DayOfWeek"] = pd.Categorical(heat_data["DayOfWeek"], categories=days, ordered=True)
    fig_heat = px.density_heatmap(
        heat_data,
        x="Hour",
        y="DayOfWeek",
        z="Count",
        title="Email Traffic by Hour & Day",
        height=300,
        nbinsx=24
    )
    fig_heat.update_layout(xaxis_title="Hour of Day", yaxis_title="Day of Week")
    st.plotly_chart(fig_heat, use_container_width=True)

# --- 6) Detailed Email View (with Expanders) ---
st.subheader(f"✉️ Email Details ({len(working_df)} selected)")
if working_df.empty:
    st.warning("Nothing to display.")
else:
    pages = ceil(len(working_df) / 50)
    page = st.sidebar.number_input("Page", 1, pages, 1)
    chunk = working_df.iloc[(page - 1) * 50 : page * 50]

    display_cols = ["DateTime", "From", "Subject"]
    if "To" in chunk.columns:
        display_cols.append("To")

    st.dataframe(
        chunk[display_cols].style.format({"DateTime": "{:%Y-%m-%d %H:%M}"}),
        use_container_width=True
    )

    for idx, row in chunk.iterrows():
        ts = row["DateTime"].strftime("%Y-%m-%d %H:%M")
        subject = row.get("Subject", "(no subject)")

        with st.expander(f"📧 {ts} — {subject}", expanded=False):
            col_a, col_b = st.columns([1, 1])
            with col_a:
                st.write(f"**From:** {row.get('From', 'Unknown')}")
                if "To" in row and pd.notna(row["To"]):
                    st.write(f"**To:** {row['To']}")
                if "Cc" in row and pd.notna(row["Cc"]):
                    st.write(f"**Cc:** {row['Cc']}")
            with col_b:
                st.write(f"**Date:** {ts}")
                if "Message-ID" in row and pd.notna(row["Message-ID"]):
                    st.write(f"**Message ID:** {row['Message-ID'][:50]}...")

            # Short preview of the body (first 40 words)
            preview = " ".join(row["Body"].split()[:40]) + ("..." if len(row["Body"].split()) > 40 else "")
            st.markdown("**📝 Content Preview:**")
            st.markdown(preview)

            # “Show Body” toggle
            show_body_key = f"show_body_{idx}"
            if st.button("Show Body", key=show_body_key):
                full_body = row["Body"] or "_No content available_"
                wc = len(full_body.split()) if full_body != "_No content available_" else 0
                st.write(f"**Content ({wc} words):**")
                st.markdown(full_body)
