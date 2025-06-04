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

# ─────────── LOGGER CONFIG ───────────
logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# ─────────── HUGGING FACE TOKEN ───────────
# Set your HF token as an environment variable, e.g.:
#   export HUGGINGFACE_API_TOKEN="hf_XXXXXXXXXXXX"
HF_TOKEN = HUGGING_FACE_API_KEY

# ─────────── 1) PAGE CONFIG ───────────
st.set_page_config(layout="wide", page_title="Email Timeline & DistilBART Summary")

# ─────────── 2) EMAIL‐CLEANING UTILITY ───────────
def clean_email_content(content: str) -> str:
    """
    - Collapse multiple whitespace into a single space.
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

# ─────────── 3) LOAD & NORMALIZE JSON ───────────
@st.cache_data
def load_and_prepare(path: str) -> List[Dict]:
    """
    1) Load raw JSON list of emails.
    2) Clean each body and parse "Date" → Python datetime (store as "DateTime").
    3) Skip any email if its date cannot be parsed.
    4) Return a list of dicts: {From, Subject, DateTime, Body}.
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
    logger.info("Prepared %d emails with valid DateTime", len(normalized))
    return normalized

# ─────────── 4) HUGGING FACE INFERENCE HELPERS ───────────
def hf_summarize_distilbart(text: str, model: str = "sshleifer/distilbart-cnn-12-6") -> str:
    """
    Send `text` to HF Inference API using DistilBART CNN/DM.
    - Model is ~420 MB, max input length 1024 tokens.
    - We truncate to first ~1000 tokens if necessary.
    Returns the "summary_text" from HF, or raises a RuntimeError with details.
    """
    logger.info("Calling DistilBART Summarize endpoint (model=%s)", model)
    api_url = f"https://api-inference.huggingface.co/models/{model}"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}

    # If input is too long (>1000 tokens), truncate to roughly 7500 chars (~1000 tokens).
    # You can adjust this character‐based truncation if you have a tokenizer offline.
    if len(text) > 7500:
        logger.warning("Input exceeds 7500 characters, truncating to first 7500 chars.")
        text = text[:7500]

    payload = {
        "inputs": text,
        "parameters": {
            "max_length": 150,
            "min_length": 60,
            "do_sample": False
        }
    }

    response = requests.post(api_url, headers=headers, json=payload)
    if response.status_code != 200:
        err_text = response.text
        logger.error("DistilBART summarize failed (status %d): %s", response.status_code, err_text)
        raise RuntimeError(f"DistilBART error {response.status_code}: {err_text}")

    data = response.json()
    summary = data[0]["summary_text"].strip()
    logger.info("Received DistilBART summary (length=%d)", len(summary))
    return summary

def hf_instructional_summarize_distilbart(prompt: str, model: str = "sshleifer/distilbart-cnn-12-6") -> str:
    """
    Use DistilBART CNN/DM to process an instructional prompt + content.
    Returns the "summary_text" or raises a RuntimeError with details.
    """
    logger.info("Calling DistilBART Instructional Summarize (model=%s)", model)
    api_url = f"https://api-inference.huggingface.co/models/{model}"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}

    # Truncate if the prompt+content is too long
    if len(prompt) > 7500:
        logger.warning("Prompt exceeds 7500 characters, truncating.")
        prompt = prompt[:7500]

    payload = {
        "inputs": prompt,
        "parameters": {
            "max_length": 200,
            "min_length": 80,
            "do_sample": False
        }
    }

    response = requests.post(api_url, headers=headers, json=payload)
    if response.status_code != 200:
        err_text = response.text
        logger.error("DistilBART instructional summarize failed (status %d): %s", response.status_code, err_text)
        raise RuntimeError(f"DistilBART instructional error {response.status_code}: {err_text}")

    data = response.json()
    summary = data[0]["summary_text"].strip()
    logger.info("Received final DistilBART summary (length=%d)", len(summary))
    return summary

# ─────────── 5) SIMPLE “SHORT‐CIRCUIT” SUMMARY PIPELINE ───────────
def generate_one_paragraph_summary(emails: List[Dict]) -> str:
    """
    If ≤ 50 emails, skip multi‐stage. Instead:
      1) Concatenate all emails (chronologically) with headers.
      2) Prepend the exact instruction about key events, tone, conclusion.
      3) Call DistilBART once on that combined prompt.
    Otherwise (for >50), you could chunk—but most real‐world selections will be <50.
    """
    n = len(emails)
    logger.info("Generating summary for %d emails", n)

    if n <= 50:
        # 1) Build combined bodies
        lines = []
        for i, e in enumerate(emails, start=1):
            sender = e["From"]
            subject = e["Subject"]
            dt_str = e["DateTime"].strftime("%Y-%m-%d %H:%M")
            header = f"[Email {i}: From {sender} | Subject: {subject} | Date: {dt_str}]\n"
            lines.append(header + e["Body"])
        combined_bodies = "\n\n".join(lines)

        # 2) Prepend explicit instruction
        prompt = f"""
Below is a sequence of {n} related emails, in chronological order. 
Please write a single, cohesive paragraph (4–6 sentences) that covers precisely:
  1. The key events or decision points that happened over the timeline.
  2. How the tone shifted over time (for example, from collaborative to urgent to reconciliatory).
  3. The most crucial messages or decision points.
  4. Why the exchange occurred in the first place.
  5. The final conclusion or status at the end—how it differs from the start.

Emails:
{combined_bodies}

One‐Paragraph Summary:
"""
        try:
            result = hf_instructional_summarize_distilbart(prompt)
            return result
        except Exception as e:
            return f"❌ Error producing final summary: {str(e)}"

    # (Optional) fallback multi‐stage for >50 emails could go here.
    else:
        return "⚠️ Too many emails (>50). Multi‐stage summarization not implemented."

# ─────────── 6) STREAMLIT APP UI ───────────
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

# --- 2) Build Working Set (≥1 Email) ---
if selected_points:
    idxs = [pt["pointIndex"] for pt in selected_points]
    if not idxs:
        idxs = [filtered_df.index.min()]
        st.warning("Added the earliest email so we have at least one.")
    working_df = filtered_df.loc[idxs].sort_values("DateTime")
    st.success(f"✅ {len(idxs)} emails selected for summary")
else:
    st.info("No selection—using the single earliest email.")
    working_df = filtered_df.head(1)

# --- 3) One‐Paragraph Abstractive Summary ---
st.subheader("✍️ One‐Paragraph Abstractive Summary")
if st.button("Generate Crystal‐Clear Summary"):
    email_records = working_df.to_dict("records")
    with st.spinner("🛠️ Generating summary—this may take a minute..."):
        final_summary = generate_one_paragraph_summary(email_records)
    st.markdown("**📜 Final One‐Paragraph Summary:**")
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
            preview = " ".join(row["Body"].split()[:40]) + (
                "..." if len(row["Body"].split()) > 40 else ""
            )
            st.markdown("**📝 Content Preview:**")
            st.markdown(preview)

            # “Show Body” toggle
            show_body_key = f"show_body_{idx}"
            if st.button("Show Body", key=show_body_key):
                full_body = row["Body"] or "_No content available_"
                wc = len(full_body.split()) if full_body != "_No content available_" else 0
                st.write(f"**Content ({wc} words):**")
                st.markdown(full_body)
