import streamlit as st
import pandas as pd
import plotly.express as px
from src.config.config import CLEANED_JSON_PATH_1

# ————— Page Config —————
st.set_page_config(layout="wide", page_title="Interactive Email Timeline & Patterns")

# ————— Constants —————
JSON_PATH = CLEANED_JSON_PATH_1
PAGE_SIZE = 50  # emails per page in table

# ————— Data Loading & Caching —————
@st.cache_data
def load_email_data(path):
    df = pd.read_json(path)
    df["DateTime"] = pd.to_datetime(
        df["Date"],
        format="%d.%m.%Y %H:%M:%S",
        errors="coerce"
    )
    df = df.dropna(subset=["DateTime"]).sort_values("DateTime").reset_index(drop=True)
    return df

df = load_email_data(JSON_PATH)

# ————— Sidebar Filters —————
st.sidebar.header("🔍 Filters")
min_date = df["DateTime"].dt.date.min()
max_date = df["DateTime"].dt.date.max()
start_date, end_date = st.sidebar.date_input(
    "Date range", (min_date, max_date), min_value=min_date, max_value=max_date
)
senders = sorted(df["From"].unique())
selected_senders = st.sidebar.multiselect("Sender", senders, default=senders)
subject_kw = st.sidebar.text_input("Subject contains")

mask = (
    df["DateTime"].dt.date.between(start_date, end_date)
    & df["From"].isin(selected_senders)
)
if subject_kw:
    mask &= df["Subject"].str.contains(subject_kw, case=False, na=False)
filtered = df[mask]

# ————— 1) Timeline Scatter —————
st.subheader("📈 Email Timeline (Scatter)")
if filtered.empty:
    st.warning("No emails match your filters.")
else:
    fig_scatter = px.scatter(
        filtered,
        x="DateTime",
        y="From",
        hover_data=["Subject"],
        title="Emails Over Time by Sender",
        height=300,
        render_mode="webgl",
    )
    fig_scatter.update_traces(marker={"size": 6})
    fig_scatter.update_layout(
        xaxis_title="Date & Time",
        yaxis_title="Sender",
        yaxis={"categoryorder": "array", "categoryarray": senders},
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

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

# ————— 4) Paginated Table & Click‐to‐Reveal Body —————
st.subheader(f"✉️ Showing {len(filtered)} Emails")
if not filtered.empty:
    total_pages = (len(filtered) - 1) // PAGE_SIZE + 1
    page = st.sidebar.number_input("Page", 1, total_pages, 1)
    start_idx, end_idx = (page - 1) * PAGE_SIZE, page * PAGE_SIZE
    page_df = filtered.iloc[start_idx:end_idx]

    # Show just the overview table
    st.dataframe(page_df[["DateTime", "From", "Subject"]], use_container_width=True)

    # For each email on the page, render an expander that only loads when clicked
    for _, row in page_df.iterrows():
        ts = row["DateTime"].strftime("%Y-%m-%d %H:%M")
        with st.expander(f"{ts}  —  {row['Subject']}"):
            st.write(f"**From:** {row['From']}")
            if "To" in row:
                st.write(f"**To:** {row['To']}")
            body_field = "Body" if "Body" in row else "Content"
            st.markdown(row.get(body_field, "_No body/content available._"))
