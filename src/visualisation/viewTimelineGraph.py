import streamlit as st
import pandas as pd
import plotly.express as px
from streamlit_plotly_events import plotly_events
import json
import re
from datetime import date
from typing import List, Dict

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
# Assumes Ollama is running locally (default port 11434) and has the "mistral" model available.
llm = Ollama(
    model="mistral",
    base_url="http://localhost:11434",  # adjust if your Ollama REST endpoint is different
    verbose=False
)

# ————— Cleaning & Summarization Helpers —————
def clean_email_content(content: str) -> str:
    """
    Clean email content for better AI processing.
    - Collapse multiple whitespace characters into single spaces.
    - Strip out common quoted‐reply artifacts (e.g., 'On ... wrote:' lines).
    """
    if not content:
        return ""
    # Collapse excessive whitespace
    content = re.sub(r'\s+', ' ', content.strip())
    # Remove quoted‐reply headers and forwarded blocks
    content = re.sub(r'On .* wrote:', '', content)
    content = re.sub(r'From:.*?Subject:', '', content, flags=re.DOTALL)
    content = re.sub(r'-----Original Message-----.*', '', content, flags=re.DOTALL)
    return content

def summarize_single_email_ollama(email: Dict) -> str:
    """
    Generate a concise, one‐paragraph summary of a single email using Ollama’s Mistral model via LangChain.

    Parameters:
        email (Dict): A dictionary with keys like "From", "Subject", "DateTime", and "Body"/"Content".

    Returns:
        str: A one‐paragraph summary capturing key points, requested actions, and overall tone.
    """
    try:
        sender = email.get("From", "Unknown Sender")
        subject = email.get("Subject", "No Subject")
        dt = email.get("DateTime", "Unknown Date")
        # If DateTime is a pandas.Timestamp, convert to string
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
        # Use the Ollama LLM to get a completion
        response = llm(prompt)
        return response.strip()

    except Exception as e:
        return f"❌ Error generating AI summary: {str(e)}"

def summarize_emails_detailed_ollama(emails: List[Dict]) -> str:
    """
    Given a list of email‐dicts, call Ollama’s Mistral model to produce a detailed, multi‐section summary that covers:
      1. A deeper explanation of each email’s content.
      2. Key keywords/phrases and why they matter.
      3. Communication pattern: who is replying to whom, any CC chains, decision‐making flow.
      4. The overarching purpose or trigger for this thread.
      5. Findings, conclusions, or next‐step implications.
      6. Tone and sentiment analysis.

    Returns one long, cohesive response. If something fails, returns an error string.
    """
    try:
        # 1. Build a combined block of all emails
        blocks = []
        for i, email in enumerate(emails, start=1):
            sender = email.get("From", "Unknown Sender")
            subject = email.get("Subject", "No Subject")
            dt = email.get("DateTime", "Unknown Date")
            if hasattr(dt, "strftime"):
                dt = dt.strftime("%Y-%m-%d %H:%M")
            raw_body = email.get("Body", "") or email.get("Content", "")
            # Reuse cleaning steps
            body = re.sub(r'\s+', ' ', raw_body.strip())
            body = re.sub(r'On .* wrote:', '', body)
            body = re.sub(r'From:.*?Subject:', '', body, flags=re.DOTALL)
            body = re.sub(r'-----Original Message-----.*', '', body, flags=re.DOTALL)

            blocks.append(
                f"---\nEmail {i}:\nFrom: {sender}\nSubject: {subject}\nDate: {dt}\n\n{body}\n"
            )

        combined_block = "\n".join(blocks)

        # 2. Write a more prescriptive prompt
        prompt = f"""
Below are {len(emails)} related emails (a thread or selection). Provide a single, cohesive deep analysis that includes all of the following:

1. **Detailed Overview of Each Email**  
   For each message in chronological order, summarize its content in 2–3 sentences. Mention who said what and any explicit requests or decisions.

2. **Key Keywords and Their Importance**  
   Identify 5–7 keywords or phrases that recur (e.g., “budget,” “deadline,” “action item”). For each keyword, explain why it was important in this thread.

3. **Communication Pattern**  
   Describe how participants are interacting (e.g., “Alice replied directly to Bob’s budget question, CC’ing finance@company.com,” “This thread branches into two sub‐conversations after the third message,” etc.). Note any loops or decision flows.

4. **Overall Purpose / Why This Thread Was Started**  
   Explain what triggered this entire exchange.

5. **Findings and Conclusions**  
   Based on the content, what can a reader conclude? E.g., “They have agreed to a $50k budget, pending final sign‐off.” Or “No clear decision was reached.”

6. **Tone and Sentiment Analysis**  
   Characterize the general sentiment (e.g., “polite but urgent,” “some frustration over missed deadlines,” “collaborative and constructive,” etc.).

Do NOT just list bullet‐points. Instead, weave everything into coherent paragraphs (roughly 6–8 sentences per section). Use bold section headers as shown below:

---
{combined_block}

**DETAILED SUMMARY**  
"""
        # 3. Call the Ollama LLM
        response = llm(prompt)
        return response.strip()

    except Exception as e:
        return f"❌ Error generating deep AI summary: {str(e)}"


def summarize_single_email_simple(email: Dict) -> str:
    """
    Fallback summary if AI is disabled. Produces a simple, human‐readable paragraph with:
      - Sender, Subject, and Date
      - A 40‐word preview of the cleaned email body

    Returns:
        str: A short paragraph previewing the email.
    """
    sender = email.get("From", "Unknown Sender")
    subject = email.get("Subject", "No Subject")
    dt = email.get("DateTime", "Unknown Date")
    if hasattr(dt, "strftime"):
        dt = dt.strftime("%Y-%m-%d %H:%M")

    raw_body = email.get("Body", "") or email.get("Content", "")
    body = clean_email_content(raw_body)
    tokens = body.split()
    snippet = " ".join(tokens[:40]) + ("..." if len(tokens) > 40 else "")

    paragraph = (
        f"This email was sent by {sender} on {dt} (Subject: “{subject}”). "
        f"Content preview: “{snippet}”"
    )
    return paragraph


def summarize_single_email(
    email: Dict,
    use_ai: bool,
    method: str
) -> str:
    """
    Wrapper that chooses between the Ollama‐powered summary and the simple fallback.

    Parameters:
        email (Dict): A dictionary representing a single email.
        use_ai (bool): Whether to attempt AI summarization.
        method (str): Should be "Ollama Mistral" to use the AI path; otherwise, falls back.

    Returns:
        str: A one‐paragraph, human‐readable summary of the email.
    """
    if use_ai and method == "Ollama Mistral":
        return summarize_single_email_ollama(email)
    else:
        return summarize_single_email_simple(email)


# ————— Revised Summarization Functions —————
def summarize_emails_paragraph_ollama(emails: List[Dict]) -> str:
    """
    Given a list of email-dicts, call Ollama’s Mistral model to produce a multi-sentence paragraph
    that explains:
      1. Why these emails were sent (purpose/intent connecting them).
      2. The main topics/themes discussed.
      3. Any key action items or decisions.
      4. The tone or sentiment among participants.

    Returns a single cohesive paragraph (3–5 sentences).
    """
    try:
        # Build a combined block of all emails
        blocks = []
        for i, email in enumerate(emails, start=1):
            sender = email.get("From", "Unknown Sender")
            subject = email.get("Subject", "No Subject")
            dt = email.get("DateTime", "Unknown Date")
            if hasattr(dt, "strftime"):
                dt = dt.strftime("%Y-%m-%d %H:%M")
            raw_body = email.get("Body", "") or email.get("Content", "")
            body = clean_email_content(raw_body)

            blocks.append(
                f"---\nEmail {i}:\nFrom: {sender}\nSubject: {subject}\nDate: {dt}\n\n{body}\n"
            )

        combined_block = "\n".join(blocks)

        prompt = f"""
Below are {len(emails)} related emails from a single thread (or selection). 
Please write a detailed, multi-sentence paragraph (approximately 4–6 sentences) that covers:
  1. Why these emails were exchanged—i.e., the overarching purpose or problem they address.
  2. The main topics or themes that emerge across them.
  3. Any concrete action items, decisions, or requests mentioned.
  4. The overall tone or sentiment among participants (e.g., collaborative, urgent, polite).

Do NOT list each email separately. Instead, weave everything into one cohesive paragraph.

{combined_block}

Summary:
"""
        response = llm(prompt)
        return response.strip()

    except Exception as e:
        return f"❌ Error generating AI summary: {str(e)}"


def summarize_emails_paragraph_simple(emails: List[Dict]) -> str:
    """
    Fallback summary (multi-sentence paragraph):
      • Attempts to infer why these emails exist (e.g., coordinating a meeting,
        updating on a project, requesting approvals, etc.)
      • Identifies 2–3 key themes or topics found in the bodies.
      • Notes any clear action items or next steps.
      • Characterizes the tone.

    Returns one paragraph of roughly 4–5 sentences.
    """
    if not emails:
        return "No emails to summarize."

    # 1) Merge all bodies into one lowercase string for keyword scanning
    all_text = " ".join(
        clean_email_content(e.get("Body", "") or e.get("Content", ""))
        for e in emails
    ).lower()

    # 2) Infer “purpose/context”
    purpose = "These emails appear to be general correspondence."
    if any(k in all_text for k in ["meeting", "schedule", "calendar", "reschedule"]):
        purpose = "These messages revolve around scheduling and coordinating one or more meetings."
    elif any(k in all_text for k in ["project", "milestone", "phase", "deliverable"]):
        purpose = "They focus on providing updates and coordination for an ongoing project."
    elif any(k in all_text for k in ["report", "analysis", "results", "data"]):
        purpose = "The thread is primarily about sharing reports and analysis results."
    elif any(k in all_text for k in ["invoice", "billing", "payment"]):
        purpose = "This sequence concerns billing or payment inquiries."
    elif any(k in all_text for k in ["approval", "sign off", "authorize", "approved"]):
        purpose = "The emails are largely about seeking or granting approvals."

    # 3) Identify 2–3 key themes/topics
    themes = []
    if "deadline" in all_text:
        themes.append("upcoming deadlines")
    if "follow up" in all_text or "follow-up" in all_text:
        themes.append("follow-up tasks")
    if "feedback" in all_text:
        themes.append("requesting feedback")
    if "update" in all_text:
        themes.append("status updates")
    if "budget" in all_text:
        themes.append("budget or financial planning")

    # 4) Detect action items / next steps
    actions = []
    if "action item" in all_text or "to do" in all_text:
        actions.append("defined action items")
    if "please review" in all_text or "kindly review" in all_text:
        actions.append("requests for review")
    if "confirm" in all_text:
        actions.append("requests to confirm details")
    if "deadline" in all_text:
        actions.append("noted deadlines")
    if "meeting" in all_text and "agenda" in all_text:
        actions.append("agenda planning for upcoming meetings")

    # 5) Characterize tone
    tone = "The tone comes across as straightforward and neutral."
    if any(w in all_text for w in ["thanks", "thank you", "appreciate"]):
        tone = "Overall, the tone seems polite and appreciative."
    elif any(w in all_text for w in ["urgent", "asap", "immediately"]):
        tone = "The tone indicates urgency and a need for prompt action."
    elif any(w in all_text for w in ["issue", "concern", "problem"]):
        tone = "There is a sense of concern or troubleshooting in the tone."

    # 6) Build a multi-sentence paragraph
    sentences = []
    sentences.append(purpose)

    if themes:
        theme_sentence = (
            "Key themes include “" + ", ".join(themes[:2]) + "”"
            + (" and other related topics." if len(themes) > 2 else ".")
        )
        sentences.append(theme_sentence)
    else:
        sentences.append("No strong topic emerged beyond the primary purpose.")

    if actions:
        action_sentence = (
            "The emails outline "
            + ", ".join(actions[:2])
            + (" and other follow-up tasks." if len(actions) > 2 else ".")
        )
        sentences.append(action_sentence)
    else:
        sentences.append("There are no explicit action items spelled out.")

    sentences.append(tone)

    # Join into one paragraph
    paragraph = " ".join(sentences).strip()
    return paragraph


def summarize_emails_insightful_ollama(emails: List[Dict]) -> str:
    """
    Given a list of email‐dicts, call Ollama’s Mistral model to produce a single, cohesive paragraph
    of roughly 10 sentences (~200 words) that deeply analyzes:
      • Why these emails were exchanged (purpose/intent).
      • The key keywords and their significance.
      • Communication patterns.
      • Major insights or findings one can draw.
      • Tone and sentiment.

    Returns a detailed paragraph. On error, returns a string starting with "❌".
    """
    try:
        # 1. Build a combined block of all emails
        blocks = []
        for i, email in enumerate(emails, start=1):
            sender = email.get("From", "Unknown Sender")
            subject = email.get("Subject", "No Subject")
            dt = email.get("DateTime", "Unknown Date")
            if hasattr(dt, "strftime"):
                dt = dt.strftime("%Y-%m-%d %H:%M")
            raw_body = email.get("Body", "") or email.get("Content", "")
            # Minimal cleaning to remove forwarded/reply metadata
            body = re.sub(r'\s+', ' ', raw_body.strip())
            body = re.sub(r'On .* wrote:', '', body)
            body = re.sub(r'From:.*?Subject:', '', body, flags=re.DOTALL)
            body = re.sub(r'-----Original Message-----.*', '', body, flags=re.DOTALL)

            blocks.append(
                f"---\nEmail {i}:\nFrom: {sender}\nSubject: {subject}\nDate: {dt}\n\n{body}\n"
            )

        combined_block = "\n".join(blocks)

        # 2. Build a forcing prompt that asks for ~10 sentences
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
        # 3. Call the Ollama LLM
        response = llm(prompt)
        return response.strip()

    except Exception as e:
        return f"❌ Error generating deep AI summary: {str(e)}"


def summarize_all_selected(
    emails: List[Dict],
    use_ai: bool,
    method: str
) -> str:
    """
    Wrapper that chooses AI vs. simple for the entire list of selected emails.
    Returns one multi‐sentence paragraph.
    """
    if use_ai and method == "Ollama Mistral":
        return summarize_emails_insightful_ollama(emails)
    else:
        return summarize_emails_paragraph_simple(emails)


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

# AI Configuration
st.sidebar.header("🤖 AI Summary Settings")
use_ai_summary = st.sidebar.checkbox("Enable AI Summary", value=True)
summary_method = st.sidebar.selectbox(
    "Summary Method",
    ["Simple Analysis", "Ollama Mistral"],
    help="Choose how to generate the summary of all selected emails"
)

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

# ————— 3) Summary of All Selected Emails —————
st.subheader("📝 Combined Summary of Selected Emails")
if working.empty:
    st.warning("No emails selected for summarization.")
else:
    email_records = working.to_dict("records")
    summary_text = summarize_all_selected(
        emails=email_records,
        use_ai=use_ai_summary,
        method=summary_method
    )
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

            # 3) Always display the detailed summary paragraph
            email_dict = row.to_dict()
            if hasattr(email_dict["DateTime"], "strftime"):
                email_dict["DateTime"] = email_dict["DateTime"].strftime("%Y-%m-%d %H:%M")

            summary_text = summarize_single_email(
                email=email_dict,
                use_ai=use_ai_summary,
                method=summary_method
            )
            st.markdown("**📝 Detailed Summary:**")
            st.markdown(summary_text)

            # 4) “Show Body” button to reveal the full email content on demand
            show_body_key = f"show_body_{idx}"
            if st.button("Show Body", key=show_body_key):
                raw_body = row.get("Body") or row.get("Content") or "_No content available_"
                word_count = len(raw_body.split()) if raw_body != "_No content available_" else 0

                st.write(f"**Content** ({word_count} words):")
                st.markdown(raw_body)
