import streamlit as st
import pandas as pd
import plotly.express as px
from streamlit_plotly_events import plotly_events
import json
from datetime import date
from src.config.config import  CLEANED_JSON_PATH

# ————— Page Config —————
st.set_page_config(layout="wide", page_title="Email Timeline & Selection")

# ————— Constants —————
JSON_PATH = CLEANED_JSON_PATH
PAGE_SIZE = 50
MIN_SELECTED = 1

# ————— Load & Introspect JSON —————
@st.cache_data
def load_raw(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

raw = load_raw(JSON_PATH)
if not isinstance(raw, list) or len(raw) == 0:
    st.error("JSON is empty or not an array of emails.")
    st.stop()

# show user the keys in a sample record
st.sidebar.markdown("**JSON sample keys:**")
st.sidebar.write(list(raw[0].keys()))

# flatten into DataFrame
df = pd.json_normalize(raw)

# find date-like columns
candidates = [c for c in df.columns if any(k in c.lower() for k in ("date","time","timestamp"))]
st.sidebar.markdown("**Date-like columns:**")
st.sidebar.write(candidates)

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

# ————— Sidebar Filters —————
st.sidebar.header("🔍 Filters")
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
fig = px.scatter(
    filtered,
    x="DateTime",
    y="From",
    hover_data=["Subject"],
    render_mode="webgl",
    title="Drag a box or lasso to select",
    height=350,
)
fig.update_traces(marker={"size": 6})
fig.update_layout(
    xaxis_title="Date & Time",
    yaxis_title="Sender",
    yaxis={"categoryorder": "array", "categoryarray": senders},
)
selected = plotly_events(fig, select_event=True, override_height=350)
st.plotly_chart(fig, use_container_width=True)

# ————— 2) Build Working Set (≥ MIN_SELECTED) —————
if selected:
    idxs = [pt["pointIndex"] for pt in selected]
    if len(idxs) < MIN_SELECTED:
        remaining = [i for i in filtered.index if i not in idxs]
        idxs += remaining[: MIN_SELECTED - len(idxs)]
        st.warning(f"Only {len(selected)} selected—added oldest to reach {MIN_SELECTED}.")
    working = filtered.loc[idxs]
else:
    st.info(f"No selection—showing first {MIN_SELECTED} chronologically.")
    working = filtered.head(MIN_SELECTED)

# ————— 2) Daily Volume Line Chart —————
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
        height=250
    )
    fig_line.update_layout(xaxis_title="Date", yaxis_title="Email Count")
    st.plotly_chart(fig_line, use_container_width=True)

# ————— 3) Heatmap: Hour vs Day of Week —————
st.subheader("📋 Heatmap: Emails by Hour & Day of Week")
if not filtered.empty:
    heat = filtered.assign(
        DayOfWeek=filtered["DateTime"].dt.day_name(),
        Hour=filtered["DateTime"].dt.hour
    ).groupby(["DayOfWeek", "Hour"]).size().rename("Count").reset_index()
    days = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    heat["DayOfWeek"] = pd.Categorical(heat["DayOfWeek"], categories=days, ordered=True)
    fig_heat = px.density_heatmap(
        heat,
        x="Hour",
        y="DayOfWeek",
        z="Count",
        title="Email Traffic by Hour & Day",
        height=350,
        nbinsx=24
    )
    fig_heat.update_layout(xaxis_title="Hour of Day", yaxis_title="Day of Week")
    st.plotly_chart(fig_heat, use_container_width=True)

# ————— 3) Paginated View & Lazy-Load Body —————
st.subheader(f"✉️ Viewing {len(working)} Emails")
if working.empty:
    st.warning("Nothing to display.")
else:
    pages = (len(working)-1)//PAGE_SIZE + 1
    page = st.sidebar.number_input("Page", 1, pages, 1)
    chunk = working.iloc[(page-1)*PAGE_SIZE : page*PAGE_SIZE]

    # overview table
    st.dataframe(chunk[["DateTime", "From", "Subject"]], use_container_width=True)

    # full message on demand
    for _, row in chunk.iterrows():
        ts = row["DateTime"].strftime("%Y-%m-%d %H:%M")
        with st.expander(f"{ts} — {row.get('Subject','(no subject)')}"):
            st.write(f"**From:** {row.get('From','')}")
            if "To" in row: st.write(f"**To:** {row['To']}")
            body = row.get("Body") or row.get("Content") or "_No content available_"
            st.markdown(body)
