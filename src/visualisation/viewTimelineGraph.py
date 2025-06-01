import streamlit as st
import pandas as pd
import plotly.express as px
from streamlit_plotly_events import plotly_events
import json
import re
from datetime import date
from typing import List, Dict
import logging

# Configure logging (optional)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

# Import LangChain’s Ollama wrapper
from langchain.llms import Ollama

from src.config.config import CLEANED_JSON_PATH

# ————— Page Config —————
st.set_page_config(layout="wide", page_title="Email Timeline & AI Summary")

# ————— Constants —————
JSON_PATH = CLEANED_JSON_PATH
PAGE_SIZE = 50
MIN_SELECTED = 1

# ————— Instantiate the Ollama LLM —————
# Assumes Ollama is running locally (default port 11434) with the "mistral" model available.
llm = Ollama(
    model="mistral",
    base_url="http://localhost:11434",  # adjust if your Ollama endpoint differs
    verbose=False
)

# ————— Cleaning Helper —————
def clean_email_content(content: str) -> str:
    """
    Clean email content for better AI processing.
    - Collapse multiple whitespace characters into single spaces.
    - Strip out common quoted‐reply artifacts (e.g., 'On ... wrote:' lines).
    """
    if not content:
        return ""
    content = re.sub(r'\s+', ' ', content.strip())
    content = re.sub(r'On .* wrote:', '', content)
    content = re.sub(r'From:.*?Subject:', '', content, flags=re.DOTALL)
    content = re.sub(r'-----Original Message-----.*', '', content, flags=re.DOTALL)
    return content

# ————— Ollama Summarization Functions —————
def summarize_single_email_ollama(email: Dict) -> str:
    """
    Generate a concise, one‐paragraph summary of a single email using Ollama’s Mistral model.
    """
    logger.info("→ Enter summarize_single_email_ollama()")
    sender = email.get("From", "Unknown Sender")
    subject = email.get("Subject", "No Subject")
    dt = email.get("DateTime", "Unknown Date")
    if hasattr(dt, "strftime"):
        dt = dt.strftime("%Y-%m-%d %H:%M")

    raw_body = email.get("Body", "") or email.get("Content", "")
    body = clean_email_content(raw_body)

    prompt = f"""
Below is an email. Please write a concise, one‐paragraph summary that captures:
  • The key points discussed and any requested actions or next steps.
  • The overall tone or sentiment of the message.

---
From: {sender}
Subject: {subject}
Date: {dt}

{body}

Summary:
"""
    logger.debug("Prompt to Ollama (single email):\n%s", prompt)
    response = llm(prompt)
    summary = response.strip()
    logger.info("← Exit summarize_single_email_ollama()")
    return summary

def summarize_emails_insightful_ollama(emails: List[Dict]) -> str:
    """
    Given a list of email‐dicts, call Ollama’s Mistral model to produce a single, cohesive paragraph
    of roughly ten sentences (~200 words) that deeply analyzes:
      1. Purpose/Trigger
      2. Key Keywords & Significance
      3. Communication Pattern
      4. Insights/Findings
      5. Tone & Sentiment

    Returns a detailed paragraph.
    """
    logger.info("→ Enter summarize_emails_insightful_ollama() with %d emails", len(emails))

    # 1. Build a combined block of all emails
    blocks = []
    for i, email in enumerate(emails, start=1):
        sender = email.get("From", "Unknown Sender")
        subject = email.get("Subject", "No Subject")
        dt = email.get("DateTime", "Unknown Date")
        if hasattr(dt, "strftime"):
            dt = dt.strftime("%Y-%m-%d %H:%M")

        raw_body = email.get("Body", "") or email.get("Content", "")
        body = re.sub(r'\s+', ' ', raw_body.strip())
        body = re.sub(r'On .* wrote:', '', body)
        body = re.sub(r'From:.*?Subject:', '', body, flags=re.DOTALL)
        body = re.sub(r'-----Original Message-----.*', '', body, flags=re.DOTALL)

        blocks.append(
            f"---\nEmail {i}:\nFrom: {sender}\nSubject: {subject}\nDate: {dt}\n\n{body}\n"
        )

    combined_block = "\n".join(blocks)
    logger.debug("Combined block for insightful summary:\n%s", combined_block)

    # 2. Build the forcing prompt
    prompt = f"""
Below are {len(emails)} related emails (thread or selection). 
Your task is to write a single cohesive paragraph of approximately ten sentences (~200 words) that covers:
  1. **Purpose/Trigger**: Why did these emails exist? What problem or question initiated this thread?
  2. **Key Keywords & Significance**: Identify 3–5 recurring keywords or phrases (e.g., “deadline,” “approval,” “budget”) and briefly explain why each is important in context.
  3. **Communication Pattern**: Describe how participants are interacting—who replies to whom, any notable CC chains, or decision‐making flow.
  4. **Insights/Findings**: What conclusions or important insights emerge from this set of messages?
  5. **Tone & Sentiment**: Characterize the overall sentiment (e.g., collaborative, urgent, frustrated, polite) and reference any phrases that shape that tone.

Make sure this is a single, fluid paragraph—do NOT list bullet points or separate sections. Aim for roughly ten sentences so the explanation feels thorough.

Here are the emails:
{combined_block}

**DETAILED ANALYSIS PARAGRAPH:**
"""
    logger.debug("Prompt to Ollama (insightful):\n%s", prompt)
    response = llm(prompt)
    result = response.strip()
    logger.info("← Exit summarize_emails_insightful_ollama()")
    return result

# ————— Load & Normalize JSON —————
@st.cache_data
def load_raw(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

raw = load_raw(JSON_PATH)
if not isinstance(raw, list) or len(raw) == 0:
    st.error("JSON is empty or not an array of emails.")
    st.stop()

df = pd.json_normalize(raw)

# Find any date‐like column
candidates = [col for col in df.columns if any(k in col.lower() for k in ("date", "time", "timestamp"))]
if not candidates:
    st.error("No date‐like field found in your JSON.")
    st.stop()

date_col = candidates[0]
df["DateTime"] = pd.to_datetime(df[date_col], errors="coerce", dayfirst=True)
df = df.dropna(subset=["DateTime"]).sort_values("DateTime").reset_index(drop=True)
if df.empty:
    st.error(f"Could not parse any dates from field `{date_col}`.")
    st.stop()

# ————— Sidebar Configuration —————
st.sidebar.header("🔍 Filters")

# Date & Sender Filters
valid_dates = df["DateTime"]
min_date = valid_dates.min().date()
max_date = valid_dates.max().date()

start_date, end_date = st.sidebar.date_input(
    "Date range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)

senders = sorted(df.get("From", pd.Series(["(unknown)"])).fillna("(unknown)").unique())
selected_senders = st.sidebar.multiselect("Sender", senders, default=senders)
subject_kw = st.sidebar.text_input("Subject contains")

# Apply filters
mask = (
    df["DateTime"].dt.date.between(start_date, end_date) &
    df.get("From", "").isin(selected_senders)
)
if subject_kw:
    mask &= df.get("Subject", "").str.contains(subject_kw, case=False, na=False)

filtered = df[mask]

# ————— 1) Interactive Timeline —————
st.subheader("📈 Select Emails on the Timeline")
st.write("💡 **Tip**: Use the lasso or box select tool to choose emails for a combined summary.")

fig = px.scatter(
    filtered,
    x="DateTime",
    y="From",
    hover_data=["Subject"],
    render_mode="webgl",
    title="Drag a box or lasso to select emails for summary",
    height=350,
)
fig.update_traces(marker={"size": 8, "opacity": 0.7})
fig.update_layout(
    xaxis_title="Date & Time",
    yaxis_title="Sender",
    yaxis={"categoryorder": "array", "categoryarray": senders},
)

selected = plotly_events(fig, select_event=True, override_height=350)

# ————— 2) Build Working Set (≥ MIN_SELECTED) —————
if selected:
    idxs = [pt["pointIndex"] for pt in selected]
    if len(idxs) < MIN_SELECTED:
        remaining = [i for i in filtered.index if i not in idxs]
        idxs += remaining[: MIN_SELECTED - len(idxs)]
        st.warning(f"Only {len(selected)} selected—added oldest to reach {MIN_SELECTED}.")
    working = filtered.loc[idxs]
    st.success(f"✅ {len(idxs)} emails selected for summary")
else:
    st.info(f"No selection—showing first {MIN_SELECTED} chronologically.")
    working = filtered.head(MIN_SELECTED)

# ————— 3) Summary of All Selected Emails (Ollama Only) —————
st.subheader("📝 Combined Summary of Selected Emails")

if working.empty:
    st.warning("No emails selected for summarization.")
else:
    email_records = working.to_dict("records")
    # Always call Ollama’s insightful summarization
    summary_text = summarize_emails_insightful_ollama(email_records)
    st.markdown(summary_text)

# ————— 4) Charts and Visualizations —————
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Daily Email Volume")
    if not filtered.empty:
        daily_counts = (
            filtered
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
    if not working.empty:
        sender_counts = working['From'].value_counts().head(10)
        fig_pie = px.pie(
            values=sender_counts.values,
            names=sender_counts.index,
            title="Selected Emails by Sender",
            height=300
        )
        st.plotly_chart(fig_pie, use_container_width=True)

# ————— 5) Heatmap: Hour vs Day of Week —————
st.subheader("🔥 Email Traffic Heatmap")
if not filtered.empty:
    heat = filtered.assign(
        DayOfWeek=filtered["DateTime"].dt.day_name(),
        Hour=filtered["DateTime"].dt.hour
    ).groupby(["DayOfWeek", "Hour"]).size().rename("Count").reset_index()
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    heat["DayOfWeek"] = pd.Categorical(heat["DayOfWeek"], categories=days, ordered=True)
    fig_heat = px.density_heatmap(
        heat,
        x="Hour",
        y="DayOfWeek",
        z="Count",
        title="Email Traffic by Hour & Day",
        height=300,
        nbinsx=24
    )
    fig_heat.update_layout(xaxis_title="Hour of Day", yaxis_title="Day of Week")
    st.plotly_chart(fig_heat, use_container_width=True)

# ————— 6) Detailed Email View (with Expander + “Show Body” toggle) —————
st.subheader(f"✉️ Email Details ({len(working)} selected)")
if working.empty:
    st.warning("Nothing to display.")
else:
    pages = (len(working) - 1) // PAGE_SIZE + 1
    page = st.sidebar.number_input("Page", 1, pages, 1)
    chunk = working.iloc[(page - 1) * PAGE_SIZE : page * PAGE_SIZE]

    # Display a small table of metadata first
    display_cols = ["DateTime", "From", "Subject"]
    if "To" in chunk.columns:
        display_cols.append("To")

    st.dataframe(
        chunk[display_cols].style.format({"DateTime": "{:%Y-%m-%d %H:%M}"}),
        use_container_width=True
    )

    # Iterate over each row and wrap in an expander
    for idx, row in chunk.iterrows():
        ts = row["DateTime"].strftime("%Y-%m-%d %H:%M")
        subject = row.get("Subject", "(no subject)")

        # 1) Create an expander per email
        with st.expander(f"📧 {ts} — {subject}", expanded=False):
            # 2) Show metadata (From, To, Cc, Message ID)
            col1, col2 = st.columns([1, 1])
            with col1:
                st.write(f"**From:** {row.get('From', 'Unknown')}")
                if "To" in row and pd.notna(row["To"]):
                    st.write(f"**To:** {row['To']}")
                if "Cc" in row and pd.notna(row["Cc"]):
                    st.write(f"**Cc:** {row['Cc']}")
            with col2:
                st.write(f"**Date:** {ts}")
                if "MessageId" in row and pd.notna(row["MessageId"]):
                    st.write(f"**Message ID:** {row['MessageId'][:50]}...")

            # 3) Always display the Ollama‐powered summary
            email_dict = row.to_dict()
            if hasattr(email_dict["DateTime"], "strftime"):
                email_dict["DateTime"] = email_dict["DateTime"].strftime("%Y-%m-%d %H:%M")

            summary_text = summarize_single_email_ollama(email_dict)
            st.markdown("**📝 Detailed Summary:**")
            st.markdown(summary_text)

            # 4) “Show Body” button to reveal the full email content on demand
            show_body_key = f"show_body_{idx}"
            if st.button("Show Body", key=show_body_key):
                raw_body = row.get("Body") or row.get("Content") or "_No content available_"
                word_count = len(raw_body.split()) if raw_body != "_No content available_" else 0

                st.write(f"**Content** ({word_count} words):")
                st.markdown(raw_body)
