# streamlit_app.py
import json
from datetime import datetime

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import plotly.express as px
from pyvis.network import Network

# ——— Helpers ———

@st.cache_data
def load_analysis(path="email_relationship_analysis.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def make_pyvis(nodes, edges, directed=False, height="600px"):
    """Build and return HTML for a PyVis network."""
    net = Network(height=height, width="100%", directed=directed)
    net.toggle_physics(True)

    # add nodes
    for n in nodes:
        if isinstance(n, dict):
            nid = n["id"]
            label = n.get("label", nid)
        else:
            nid, attrs = n[0], (n[1] if len(n)>1 else {})
            label = attrs.get("label", nid)
        net.add_node(nid, label=label)

    # add edges
    for e in edges:
        if isinstance(e, dict):
            src, tgt = e["source"], e["target"]
            w = e.get("weight", None)
        else:
            src, tgt = e[0], e[1]
            w = (e[2].get("weight") if len(e)>2 else None)
        if w is not None:
            net.add_edge(src, tgt, value=w)
        else:
            net.add_edge(src, tgt)

    # render to temporary HTML and return its content
    tmpfile = "temp_network.html"
    net.save_graph(tmpfile)
    with open(tmpfile, "r", encoding="utf-8") as f:
        return f.read()

# ——— Main ———

st.set_page_config(layout="wide")
st.title("📧 Email Relationship Analysis Dashboard")

data = load_analysis()

viz = st.sidebar.selectbox("Choose visualization", [
    "Summary Statistics",
    "Temporal Analysis",
    "Entity Overlaps",
    "Communication Patterns",
    "Network Graph",
    "Topic Progression",
])

if viz == "Summary Statistics":
    stats = data["summary_stats"]
    st.subheader("Overview Metrics")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Emails", stats["total_emails"])
    start = datetime.fromisoformat(stats["date_range"]["start"]).date()
    end   = datetime.fromisoformat(stats["date_range"]["end"]).date()
    c2.metric("Date Range", f"{start} ↔ {end}")
    c3.metric("Key Players", ", ".join(stats["key_players"]))

    # interactive bar charts with Plotly
    st.markdown("**Classifications**")
    fig1 = px.bar(
        x=list(stats["classification_counts"].keys()),
        y=list(stats["classification_counts"].values()),
        labels={"x":"Classification","y":"Count"}
    )
    st.plotly_chart(fig1, use_container_width=True)

    st.markdown("**Tones**")
    fig2 = px.bar(
        x=list(stats["tone_counts"].keys()),
        y=list(stats["tone_counts"].values()),
        labels={"x":"Tone","y":"Count"}
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("**Top Entities**")
    top_entities = pd.Series(stats["entity_counts"]).nlargest(10)
    fig3 = px.bar(
        x=top_entities.index,
        y=top_entities.values,
        labels={"x":"Entity","y":"Mentions"}
    )
    st.plotly_chart(fig3, use_container_width=True)

elif viz == "Temporal Analysis":
    ta = pd.DataFrame(data["temporal_analysis"])
    ta["date"] = pd.to_datetime(ta["date"])
    st.subheader("Emails Over Time")
    monthly = ta.set_index("date").resample("M").size().reset_index(name="count")
    fig4 = px.line(monthly, x="date", y="count", labels={"count":"Emails","date":"Date"})
    st.plotly_chart(fig4, use_container_width=True)

    st.subheader("Classification Trends")
    cls_ts = (
        ta.groupby([pd.Grouper(key="date", freq="M"), "classification"])
          .size()
          .reset_index(name="count")
    )
    fig5 = px.area(
        cls_ts,
        x="date", y="count", color="classification",
        labels={"count":"Emails","date":"Date"}
    )
    st.plotly_chart(fig5, use_container_width=True)

elif viz == "Entity Overlaps":
    eo = data["entity_overlaps"]
    pairs = [
        {"Email 1": a, "Email 2": b, "Overlap Count": v}
        for k, v in eo.items()
        for a, b in [k.split("-")]
    ]
    st.subheader("Entity Overlaps Between Email Pairs")
    st.dataframe(pd.DataFrame(pairs))

elif viz == "Communication Patterns":
    cp = data["communication_patterns"]
    st.subheader("Interactive Communication Network")
    html = make_pyvis(cp["nodes"], cp["edges"], directed=True)
    # only specify height as an int
    components.html(html, height=600)

elif viz == "Network Graph":
    ng = data["network_graph"]
    st.subheader("Interactive Global Network")
    html2 = make_pyvis(ng["nodes"], ng["edges"], directed=False)
    components.html(html2, height=600)

elif viz == "Topic Progression":
    tp = data["topic_progression"]
    st.subheader("Topics & Timeline")
    st.markdown("**Topics Identified**")
    st.write(tp["topics"])

    st.markdown("**Topic Connections**")
    st.write(pd.DataFrame(tp["connections"]))

    st.markdown("**Timeline of Topics**")
    df_tl = pd.DataFrame(tp["timeline"])
    df_tl["date"] = pd.to_datetime(df_tl["date"])
    # count events per date
    counts = df_tl.assign(cnt=1).groupby("date")["cnt"].sum().reset_index()
    fig6 = px.line(counts, x="date", y="cnt",
                   labels={"cnt":"Topic Events","date":"Date"})
    st.plotly_chart(fig6, use_container_width=True)

    st.markdown("**Key Insights**")
    for insight in tp["key_insights"]:
        st.write(f"- {insight}")
