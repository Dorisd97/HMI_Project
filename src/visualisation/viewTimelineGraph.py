import streamlit as st
import pandas as pd
import plotly.express as px
from src.config.config import CLEANED_JSON_PATH_1

# ————— APP TITLE —————
st.set_page_config(layout="wide")
st.title("📧 Interactive Email Timeline")

# ————— CONFIG —————
JSON_PATH = CLEANED_JSON_PATH_1

# ————— DATA LOADING —————
@st.cache_data
def load_email_data(path):
    df = pd.read_json(path)
    # parse your timestamp field here; adjust format if needed
    df["DateTime"] = pd.to_datetime(df["Date"], format="%d.%m.%Y %H:%M:%S", errors="coerce")
    df = df.dropna(subset=["DateTime"]).sort_values("DateTime").reset_index(drop=True)
    return df

df = load_email_data(JSON_PATH)

# ————— SIDEBAR FILTERS —————
st.sidebar.header("🔍 Filters")

# date range
min_date = df["DateTime"].dt.date.min()
max_date = df["DateTime"].dt.date.max()
start_date, end_date = st.sidebar.date_input(
    "Date range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)

# sender multi-select
senders = sorted(df["From"].unique())
selected_senders = st.sidebar.multiselect("Sender", senders, default=senders)

# subject keyword search
subject_kw = st.sidebar.text_input("Subject contains")

# ————— APPLY FILTERS —————
mask = (
    df["DateTime"].dt.date.between(start_date, end_date)
    & df["From"].isin(selected_senders)
)
if subject_kw:
    mask &= df["Subject"].str.contains(subject_kw, case=False, na=False)

filtered = df[mask]

# ————— TIMELINE CHART —————
st.subheader("📈 Timeline of Emails")
fig = px.scatter(
    filtered,
    x="DateTime",
    y="From",
    hover_data=["Subject", "DateTime"],
    title="Emails Over Time",
)
fig.update_traces(marker=dict(size=8))
fig.update_layout(
    yaxis=dict(title="Sender", categoryorder="array", categoryarray=senders),
    xaxis_title="Date & Time",
    height=500,
)
st.plotly_chart(fig, use_container_width=True)

# ————— DETAILED VIEW —————
st.subheader(f"✉️ {len(filtered)} Emails Matching Filters")
for _, row in filtered.iterrows():
    ts = row["DateTime"].strftime("%Y-%m-%d %H:%M")
    with st.expander(f"{ts}  —  {row['Subject']}"):
        st.write(f"**From:** {row['From']}")
        if "To" in row:
            st.write(f"**To:** {row['To']}")
        if "Body" in row:
            st.markdown(row["Body"])
