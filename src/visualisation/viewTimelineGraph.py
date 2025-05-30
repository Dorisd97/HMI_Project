import streamlit as st
import pandas as pd
import plotly.express as px
from streamlit_plotly_events import plotly_events

# ————— Page Config —————
st.set_page_config(layout="wide", page_title="Interactive Email Timeline & Selection")

# ————— Constants —————
JSON_PATH = "/mnt/data/cleaned_json_50.json"
PAGE_SIZE = 50

# ————— Load & Cache —————
@st.cache_data
def load_data(path):
    df = pd.read_json(path)
    df["DateTime"] = pd.to_datetime(
        df["Date"],
        format="%d.%m.%Y %H:%M:%S",
        errors="coerce"
    )
    return df.dropna(subset=["DateTime"]).sort_values("DateTime").reset_index(drop=True)

df = load_data(JSON_PATH)

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
    df["DateTime"].dt.date.between(start_date, end_date) &
    df["From"].isin(selected_senders)
)
if subject_kw:
    mask &= df["Subject"].str.contains(subject_kw, case=False, na=False)
filtered = df[mask]

# ————— 1) Interactive Timeline with Selection —————
st.subheader("📈 Email Timeline (select with box/lasso)")
fig = px.scatter(
    filtered,
    x="DateTime",
    y="From",
    hover_data=["Subject"],
    render_mode="webgl",
    title="Emails Over Time by Sender",
    height=400
)
fig.update_traces(marker={"size": 6})
fig.update_layout(
    xaxis_title="Date & Time",
    yaxis_title="Sender",
    yaxis={"categoryorder": "array", "categoryarray": senders},
)
selected_points = plotly_events(fig, select_event=True, override_height=400)
st.plotly_chart(fig, use_container_width=True)

# Build working_df based on timeline selection or fallback to full filtered
if selected_points:
    idxs = [pt["pointIndex"] for pt in selected_points]
    working_df = filtered.iloc[idxs]
    st.info(f"{len(working_df)} point(s) selected on the timeline")
else:
    working_df = filtered

# ————— 2) Paginated View of the Current Selection —————
st.subheader(f"✉️ Viewing {len(working_df)} Emails")
if working_df.empty:
    st.warning("No emails to display.")
else:
    total_pages = (len(working_df) - 1) // PAGE_SIZE + 1
    page = st.sidebar.number_input("Page", 1, total_pages, 1, help="Navigate pages of the current selection")
    start_idx = (page - 1) * PAGE_SIZE
    end_idx = start_idx + PAGE_SIZE
    page_df = working_df.iloc[start_idx:end_idx]

    # Show overview
    st.dataframe(page_df[["DateTime", "From", "Subject"]], use_container_width=True)

    # Lazy-load full bodies on click
    for _, row in page_df.iterrows():
        ts = row["DateTime"].strftime("%Y-%m-%d %H:%M")
        with st.expander(f"{ts} — {row['Subject']}"):
            st.write(f"**From:** {row['From']}")
            if "To" in row:
                st.write(f"**To:** {row['To']}")
            body = row.get("Body") or row.get("Content") or "_No content_"
            st.markdown(body)
