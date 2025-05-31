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


# ————— AI / Simple Paragraph Summary Functions —————
def clean_email_content(content: str) -> str:
    """Clean email content for better AI processing."""
    if not content:
        return ""
    # Remove excessive whitespace
    content = re.sub(r'\s+', ' ', content.strip())
    # Remove common “On ... wrote:” and forwarded chains
    content = re.sub(r'On .* wrote:', '', content)
    content = re.sub(r'From:.*?Subject:', '', content, flags=re.DOTALL)
    content = re.sub(r'-----Original Message-----.*', '', content, flags=re.DOTALL)
    return content


def summarize_single_email_openai(email: Dict, api_key: str) -> str:
    """
    Calls OpenAI to generate a single‐paragraph summary of exactly one email.
    """
    try:
        if api_key:
            openai.api_key = api_key

        sender = email.get("From", "Unknown Sender")
        subject = email.get("Subject", "No Subject")
        dt = email.get("DateTime", "Unknown Date")
        # If DateTime is a pandas Timestamp, convert to string
        if hasattr(dt, "strftime"):
            dt = dt.strftime("%Y-%m-%d %H:%M")
        raw_body = email.get("Body", "") or email.get("Content", "")
        body = clean_email_content(raw_body)

        # Build a simple prompt for one‐paragraph summary
        prompt = f"""
Below is an email. Please write a concise, one‐paragraph summary that captures:
  • key points and any requested actions  
  • overall tone / sentiment  

---  
From: {sender}  
Subject: {subject}  
Date: {dt}  

{body}

Summary:
"""
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=250,
            temperature=0.3
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"❌ Error generating AI summary: {str(e)}"


def summarize_single_email_simple(email: Dict) -> str:
    """
    Very basic fallback: grab sender/subject/date and first ~40 words of cleaned body.
    """
    sender = email.get("From", "Unknown Sender")
    subject = email.get("Subject", "No Subject")
    dt = email.get("DateTime", "Unknown Date")
    if hasattr(dt, "strftime"):
        dt = dt.strftime("%Y-%m-%d %H:%M")
    raw_body = email.get("Body", "") or email.get("Content", "")
    body = clean_email_content(raw_body)
    # Take first 40 words as “summary”
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
    method: str,
    api_key: str = None
) -> str:
    """
    Wrapper that picks AI vs. simple. Returns exactly one paragraph.
    """
    if use_ai and method == "OpenAI GPT" and api_key:
        return summarize_single_email_openai(email, api_key)
    else:
        return summarize_single_email_simple(email)


# ————— Load & Introspect JSON —————
@st.cache_data
def load_raw(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


raw = load_raw(JSON_PATH)
if not isinstance(raw, list) or len(raw) == 0:
    st.error("JSON is empty or not an array of emails.")
    st.stop()

df = pd.json_normalize(raw)

# Find date‐like columns
candidates = [c for c in df.columns if any(k in c.lower() for k in ("date", "time", "timestamp"))]
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
    help="Choose how to generate each email’s paragraph‐summary"
)

openai_api_key = None
if summary_method == "OpenAI GPT":
    openai_api_key = st.sidebar.text_input(
        "OpenAI API Key",
        type="password",
        help="Enter your OpenAI API key for AI‐powered summaries"
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

mask = (
    df["DateTime"].dt.date.between(start_date, end_date) &
    df.get("From", "").isin(selected_senders)
)
if subject_kw:
    mask &= df.get("Subject", "").str.contains(subject_kw, case=False, na=False)
filtered = df[mask]

# ————— 1) Interactive Timeline —————
st.subheader("📈 Select Emails on the Timeline")
st.write("💡 **Tip**: Use the lasso or box select tool to choose emails for analysis")

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

# ————— 5) Detailed Email View (auto‐summary) —————
st.subheader(f"✉️ Email Details ({len(working)} selected)")
if working.empty:
    st.warning("Nothing to display.")
else:
    pages = (len(working) - 1) // PAGE_SIZE + 1
    page = st.sidebar.number_input("Page", 1, pages, 1)
    chunk = working.iloc[(page - 1) * PAGE_SIZE: page * PAGE_SIZE]

    display_cols = ["DateTime", "From", "Subject"]
    if "To" in chunk.columns:
        display_cols.append("To")

    st.dataframe(
        chunk[display_cols].style.format({"DateTime": "{:%Y-%m-%d %H:%M}"}),
        use_container_width=True
    )

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
                if "MessageId" in row and pd.notna(row["MessageId"]):
                    st.write(f"**Message ID:** {row['MessageId'][:50]}...")

            body = row.get("Body") or row.get("Content") or "_No content available_"
            word_count = len(body.split()) if body != "_No content available_" else 0
            st.write(f"**Content** ({word_count} words):")
            st.markdown(body)

            # ——— Automatically generate a one‐paragraph summary ———
            email_dict = row.to_dict()
            # Convert DateTime to string if Timestamp
            if hasattr(email_dict["DateTime"], "strftime"):
                email_dict["DateTime"] = email_dict["DateTime"].strftime("%Y-%m-%d %H:%M")

            summary_text = summarize_single_email(
                email=email_dict,
                use_ai=use_ai_summary,
                method=summary_method,
                api_key=openai_api_key
            )

            st.markdown("**📝 Summary (one paragraph):**")
            st.markdown(summary_text)
