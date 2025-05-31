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


def summarize_emails_paragraph_openai(emails: List[Dict], api_key: str) -> str:
    """
    Given a list of email‐dicts, call OpenAI to produce one paragraph explaining:
      1. Why these emails were sent (purpose/intent)
      2. What insights or conclusions can be drawn
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
Below are {len(emails)} related emails. Please write a single, concise paragraph that explains:
  1. Why these emails were sent (the underlying purpose or intent connecting them)
  2. The most important insights, decisions, or conclusions that emerge from reading all of them

Combine everything into a fluid, human‐readable paragraph. Do not list each email separately—focus on the overarching reason and takeaways.

{combined_block}

Summary:
"""
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.4
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"❌ Error generating AI summary: {str(e)}"


def summarize_emails_paragraph_simple(emails: List[Dict]) -> str:
    """
    Fallback summary:
    - States why the emails likely exist (e.g., project updates, scheduling, approvals).
    - Provides a few key observations (e.g., most frequent sender, any urgent terms).
    All merged into one short paragraph.
    """
    if not emails:
        return "No emails to summarize."

    # Determine date range and senders
    dates = [e.get("DateTime") for e in emails if e.get("DateTime")]
    if dates:
        parsed = []
        for d in dates:
            if hasattr(d, "strftime"):
                parsed.append(d.strftime("%Y-%m-%d"))
            else:
                parsed.append(str(d))
        date_range = f"{min(parsed)} to {max(parsed)}"
    else:
        date_range = "unknown dates"

    senders = sorted({e.get("From", "Unknown") for e in emails})
    sender_summary = f"Emails from: {', '.join(senders)}."

    # Count how many mention “urgent” or “deadline”
    all_text = " ".join(
        clean_email_content(e.get("Body", "") or e.get("Content", ""))
        for e in emails
    )
    urgent_terms = []
    for term in ["urgent", "deadline", "important"]:
        if term in all_text.lower():
            urgent_terms.append(term)
    urgent_summary = (
        f"Appears urgency: {', '.join(urgent_terms)}."
        if urgent_terms
        else "No immediate urgency detected."
    )

    # Guess purpose by looking at common keywords
    purpose = "Likely project updates or coordination emails."
    if "meeting" in all_text.lower():
        purpose = "Primarily meeting scheduling and follow‐ups."
    elif "report" in all_text.lower():
        purpose = "Sharing status reports and results."

    # Build one paragraph
    paragraph = (
        f"Between {date_range}, {sender_summary} "
        f"{purpose} {urgent_summary}"
    )
    return paragraph


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
