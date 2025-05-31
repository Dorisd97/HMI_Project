import streamlit as st
import pandas as pd
import plotly.express as px
from streamlit_plotly_events import plotly_events
import json
from datetime import date
import openai  # or your preferred AI service
from typing import List, Dict
import re
from src.config.config import CLEANED_JSON_PATH

# ————— Page Config —————
st.set_page_config(layout="wide", page_title="Email Timeline & AI Summary")

# ————— Constants —————
JSON_PATH = CLEANED_JSON_PATH
PAGE_SIZE = 50
MIN_SELECTED = 1


# ————— AI Summary Functions —————
def clean_email_content(content: str) -> str:
    """Clean email content for better AI processing"""
    if not content:
        return ""

    # Remove excessive whitespace
    content = re.sub(r'\s+', ' ', content.strip())

    # Remove common email artifacts
    content = re.sub(r'On .* wrote:', '', content)
    content = re.sub(r'From:.*?Subject:', '', content, flags=re.DOTALL)
    content = re.sub(r'-----Original Message-----.*', '', content, flags=re.DOTALL)

    return content


def summarize_emails_openai(emails: List[Dict], api_key: str = None) -> str:
    """Summarize emails using OpenAI GPT"""
    try:
        if api_key:
            openai.api_key = api_key

        # Prepare email content for summarization
        email_texts = []
        for i, email in enumerate(emails, 1):
            subject = email.get('Subject', 'No Subject')
            sender = email.get('From', 'Unknown Sender')
            body = clean_email_content(email.get('Body', '') or email.get('Content', ''))
            timestamp = email.get('DateTime', 'Unknown Date')

            email_text = f"Email {i}:\nFrom: {sender}\nSubject: {subject}\nDate: {timestamp}\nContent: {body}\n"
            email_texts.append(email_text)

        combined_content = "\n---\n".join(email_texts)

        # Create summarization prompt
        prompt = f"""Analyze and summarize the following {len(emails)} email(s). Provide:

1. **Key Topics & Themes**: Main subjects discussed
2. **Important Actions/Decisions**: Any action items, decisions, or requests
3. **Key People**: Important individuals mentioned
4. **Timeline & Context**: Chronological flow and relationships between emails
5. **Sentiment & Tone**: Overall communication tone
6. **Summary**: Concise overview of the entire email thread/selection

Emails to analyze:
{combined_content}

Please provide a comprehensive but concise analysis."""

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000,
            temperature=0.3
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"❌ Error generating AI summary: {str(e)}"


def summarize_emails_simple(emails: List[Dict]) -> str:
    """Generate a simple rule-based summary"""
    if not emails:
        return "No emails to summarize."

    summary_parts = []

    # Basic statistics
    summary_parts.append(f"📊 **Email Analysis ({len(emails)} emails)**")

    # Date range
    dates = [email.get('DateTime') for email in emails if email.get('DateTime')]
    if dates:
        min_date = min(dates).strftime('%Y-%m-%d') if dates else 'Unknown'
        max_date = max(dates).strftime('%Y-%m-%d') if dates else 'Unknown'
        summary_parts.append(f"📅 **Date Range**: {min_date} to {max_date}")

    # Unique senders
    senders = set(email.get('From', 'Unknown') for email in emails)
    summary_parts.append(f"👥 **Senders**: {', '.join(senders)}")

    # Common subjects/topics
    subjects = [email.get('Subject', '') for email in emails if email.get('Subject')]
    summary_parts.append(f"📝 **Subjects**: {len(set(subjects))} unique subjects")

    # Content analysis
    all_content = ' '.join([
        clean_email_content(email.get('Body', '') or email.get('Content', ''))
        for email in emails
    ])

    word_count = len(all_content.split()) if all_content else 0
    summary_parts.append(f"📄 **Total Content**: ~{word_count} words")

    # Simple keyword extraction
    common_words = ['meeting', 'project', 'update', 'urgent', 'deadline', 'review', 'approval']
    found_keywords = [word for word in common_words if word.lower() in all_content.lower()]
    if found_keywords:
        summary_parts.append(f"🔍 **Key Terms Found**: {', '.join(found_keywords)}")

    return '\n\n'.join(summary_parts)


# ————— Load & Introspect JSON —————
@st.cache_data
def load_raw(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


raw = load_raw(JSON_PATH)
if not isinstance(raw, list) or len(raw) == 0:
    st.error("JSON is empty or not an array of emails.")
    st.stop()

# flatten into DataFrame
df = pd.json_normalize(raw)

# find date-like columns
candidates = [c for c in df.columns if any(k in c.lower() for k in ("date", "time", "timestamp"))]

if not candidates:
    st.error("No date-like field found in your JSON.")
    st.stop()

# pick the first candidate and parse it
date_col = candidates[0]
df["DateTime"] = pd.to_datetime(df[date_col], errors="coerce", dayfirst=True)

# drop rows where parse failed
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
    help="Choose how to generate email summaries"
)

if summary_method == "OpenAI GPT":
    openai_api_key = st.sidebar.text_input(
        "OpenAI API Key",
        type="password",
        help="Enter your OpenAI API key for AI-powered summaries"
    )

# Date and filter controls
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

# apply filters
mask = (
        df["DateTime"].dt.date.between(start_date, end_date) &
        df.get("From", "").isin(selected_senders)
)
if subject_kw:
    mask &= df.get("Subject", "").str.contains(subject_kw, case=False, na=False)

filtered = df[mask]

# ————— 1) Interactive Timeline —————
st.subheader("📈 Select Emails on the Timeline")
st.write("💡 **Tip**: Use the lasso or box select tool to choose emails for AI analysis")

fig = px.scatter(
    filtered,
    x="DateTime",
    y="From",
    hover_data=["Subject"],
    render_mode="webgl",
    title="Drag a box or lasso to select emails for analysis",
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
    st.success(f"✅ {len(idxs)} emails selected for analysis")
else:
    st.info(f"No selection—showing first {MIN_SELECTED} chronologically.")
    working = filtered.head(MIN_SELECTED)

# ————— AI SUMMARY SECTION —————
if use_ai_summary and not working.empty:
    st.subheader("🧠 AI Email Analysis")

    with st.expander("📋 **Email Summary & Analysis**", expanded=True):
        if st.button("🔄 Generate Fresh Analysis", type="primary"):
            with st.spinner("Analyzing emails..."):
                # Convert DataFrame rows to dictionaries for analysis
                email_data = working.to_dict('records')

                if summary_method == "OpenAI GPT" and 'openai_api_key' in locals() and openai_api_key:
                    summary = summarize_emails_openai(email_data, openai_api_key)
                elif summary_method == "Custom AI Service":
                    st.warning("Custom AI service not implemented. Using simple analysis.")
                    summary = summarize_emails_simple(email_data)
                else:
                    summary = summarize_emails_simple(email_data)

                st.markdown(summary)
        else:
            # Show cached analysis or prompt to generate
            st.info("👆 Click 'Generate Fresh Analysis' to analyze the selected emails")

# ————— 3) Charts and Visualizations —————
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

# ————— 4) Heatmap: Hour vs Day of Week —————
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

# ————— 5) Detailed Email View —————
st.subheader(f"✉️ Email Details ({len(working)} selected)")
if working.empty:
    st.warning("Nothing to display.")
else:
    pages = (len(working) - 1) // PAGE_SIZE + 1
    page = st.sidebar.number_input("Page", 1, pages, 1)
    chunk = working.iloc[(page - 1) * PAGE_SIZE: page * PAGE_SIZE]

    # Quick overview with enhanced metadata
    display_cols = ["DateTime", "From", "Subject"]
    if "To" in chunk.columns:
        display_cols.append("To")

    st.dataframe(
        chunk[display_cols].style.format({"DateTime": "{:%Y-%m-%d %H:%M}"}),
        use_container_width=True
    )

    # Enhanced email details with metadata
    for _, row in chunk.iterrows():
        ts = row["DateTime"].strftime("%Y-%m-%d %H:%M")
        subject = row.get('Subject', '(no subject)')

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
                if "MessageId" in row:
                    st.write(f"**Message ID:** {row['MessageId'][:50]}...")

            # Email body content
            body = row.get("Body") or row.get("Content") or "_No content available_"
            word_count = len(body.split()) if body != "_No content available_" else 0

            st.write(f"**Content** ({word_count} words):")
            st.markdown(body)