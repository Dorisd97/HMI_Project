import streamlit as st
import pandas as pd
import plotly.express as px
from streamlit_plotly_events import plotly_events
import json
from datetime import date
import openai
from typing import List, Dict
import re
from src.config.config import CLEANED_JSON_PATH

# ————— Page Config —————
st.set_page_config(layout="wide", page_title="Email Timeline & AI Summary")

# ————— Constants —————
JSON_PATH = CLEANED_JSON_PATH
PAGE_SIZE = 50
MIN_SELECTED = 1


# ————— Cleaning & Summarization Helpers —————
def clean_email_content(content: str) -> str:
    """Clean email content for better AI processing."""
    if not content:
        return ""
    # Collapse whitespace
    content = re.sub(r'\s+', ' ', content.strip())
    # Remove quoted replies/headers
    content = re.sub(r'On .* wrote:', '', content)
    content = re.sub(r'From:.*?Subject:', '', content, flags=re.DOTALL)
    content = re.sub(r'-----Original Message-----.*', '', content, flags=re.DOTALL)
    return content


# ————— Revised Summarization Functions —————


def summarize_emails_paragraph_openai(emails: List[Dict], api_key: str) -> str:
    """
    Given a list of email‑dicts, call OpenAI to produce a multi‑sentence paragraph
    that explains:
      1. Why these emails were sent (purpose/intent connecting them).
      2. The main topics/themes discussed.
      3. Any key action items or decisions.
      4. The tone or sentiment among participants.

    Returns a single cohesive paragraph (3–5 sentences).
    """
    try:
        if api_key:
            openai.api_key = api_key

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
Please write a detailed, multi‑sentence paragraph (approximately 4–6 sentences) that covers:
  1. Why these emails were exchanged—i.e., the overarching purpose or problem they address.
  2. The main topics or themes that emerge across them.
  3. Any concrete action items, decisions, or requests mentioned.
  4. The overall tone or sentiment among participants (e.g., collaborative, urgent, polite).

Do NOT list each email separately. Instead, weave everything into one cohesive paragraph.

{combined_block}

Summary:
"""
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=400,
            temperature=0.4
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"❌ Error generating AI summary: {str(e)}"


def summarize_emails_paragraph_simple(emails: List[Dict]) -> str:
    """
    Fallback summary (multi‑sentence paragraph):
      • Attempts to infer why these emails exist (e.g. coordinating a meeting,
        updating on a project, requesting approvals, etc.)
      • Identifies 2–3 key themes or topics found in the bodies.
      • Notes any clear action items or next steps.
      • Characterizes the tone (e.g. polite, urgent, informational).

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
    if "follow up" in all_text or "follow‑up" in all_text:
        themes.append("follow‑up tasks")
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

    # 6) Build a multi‑sentence paragraph
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
            + (" and other follow‑up tasks." if len(actions) > 2 else ".")
        )
        sentences.append(action_sentence)
    else:
        sentences.append("There are no explicit action items spelled out.")

    sentences.append(tone)

    # Join into one paragraph
    paragraph = " ".join(sentences).strip()
    return paragraph



def summarize_all_selected(
    emails: List[Dict],
    use_ai: bool,
    method: str,
    api_key: str = None
) -> str:
    """
    Wrapper that chooses AI vs. simple for the entire list of selected emails.
    Returns one multi‐sentence paragraph.
    """
    if use_ai and method == "OpenAI GPT" and api_key:
        return summarize_emails_paragraph_openai(emails, api_key)
    else:
        return summarize_emails_paragraph_simple(emails)


def summarize_all_selected(
    emails: List[Dict],
    use_ai: bool,
    method: str,
    api_key: str = None
) -> str:
    """
    Wrapper that chooses AI vs. simple for the entire list of selected emails.
    Returns a single paragraph string.
    """
    if use_ai and method == "OpenAI GPT" and api_key:
        return summarize_emails_paragraph_openai(emails, api_key)
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
    ["Simple Analysis", "OpenAI GPT", "Custom AI Service"],
    help="Choose how to generate the summary of all selected emails"
)

openai_api_key = None
if summary_method == "OpenAI GPT":
    openai_api_key = st.sidebar.text_input(
        "OpenAI API Key",
        type="password",
        help="Enter your OpenAI API key for AI‐powered summary"
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
        method=summary_method,
        api_key=openai_api_key
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


# ————— 6) Detailed Email View (List Only, No Per-Email Summary) —————
st.subheader(f"✉️ Email Details ({len(working)} selected)")
if working.empty:
    st.warning("Nothing to display.")
else:
    pages = (len(working) - 1) // PAGE_SIZE + 1
    page = st.sidebar.number_input("Page", 1, pages, 1)
    chunk = working.iloc[(page - 1) * PAGE_SIZE : page * PAGE_SIZE]

    display_cols = ["DateTime", "From", "Subject"]
    if "To" in chunk.columns:
        display_cols.append("To")

    st.dataframe(
        chunk[display_cols].style.format({"DateTime": "{:%Y-%m-%d %H:%M}"}),
        use_container_width=True
    )

    for _, row in chunk.iterrows():
        ts = row["DateTime"].strftime("%Y-%m-%d %H:%M")
        subject = row.get("Subject", "(no subject)")

        with st.expander(f"📧 {ts} — {subject}"):
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

            body = row.get("Body") or row.get("Content") or "_No content available_"
            word_count = len(body.split()) if body != "_No content available_" else 0
            st.write(f"**Content** ({word_count} words):")
            st.markdown(body)
