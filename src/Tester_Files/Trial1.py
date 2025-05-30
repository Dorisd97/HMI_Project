import streamlit as st
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from collections import Counter
from datetime import datetime
import json

# Streamlit app title and introduction
st.title("Enron’s Unraveling: A Tale of Ambition and Collapse")
st.markdown("""
Enron, once an energy giant, collapsed in 2001 amid financial scandals and regulatory scrutiny. This app visualizes internal email communications from 1999–2002, revealing Enron’s corporate strategies, regulatory battles, and operational struggles. The story traces Enron’s journey from ambitious deal-making to its desperate merger with Dynegy and eventual bankruptcy. Explore the visualizations below to see the key players, events, and themes that defined Enron’s downfall.
""")


# Load data
@st.cache_data
def load_data():
    try:
        with open('/data/cleaned_json_50.json', 'r') as f:
            emails = json.load(f)
        df = pd.DataFrame(emails)

        # Parse dates
        def parse_date(date_str):
            try:
                return pd.to_datetime(date_str, format="%d.%m.%Y %H:%M:%S")
            except ValueError:
                try:
                    return pd.to_datetime(date_str, format="%d.%m.%Y")
                except ValueError:
                    return pd.NaT

        df['Date'] = df['Date'].apply(parse_date)
        return df
    except FileNotFoundError:
        st.error("Error: 'cleaned_json_50.json' not found. Please ensure the file is in the same directory.")
        return None


df = load_data()
if df is None:
    st.stop()

# Visualization 1: Network Graph
st.subheader("1. Communication Network")
st.markdown("""
This network graph shows email exchanges among Enron employees and external contacts. Nodes represent individuals, and edges indicate emails, labeled with subjects. Key players like Jeff Dasovich and Susan Mara emerge as central hubs, reflecting their roles in projects like Project Boomerang and the Dynegy merger.
""")

G = nx.DiGraph()
for _, row in df.iterrows():
    sender = row['From']
    recipients = [r.strip() for r in row['To'].split(',')]
    for recipient in recipients:
        G.add_edge(sender, recipient, subject=row['Subject'])

fig, ax = plt.subplots(figsize=(12, 8))
pos = nx.spring_layout(G, seed=42)
nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=1500, font_size=8, font_weight='bold',
        edge_color='gray', ax=ax)
nx.draw_networkx_edge_labels(G, pos, edge_labels={(u, v): d['subject'][:20] + '...' for u, v, d in G.edges(data=True)},
                             font_size=6)
plt.title("Enron Email Communication Network")
st.pyplot(fig)

# Visualization 2: Timeline Plot
st.subheader("2. Timeline of Key Events")
st.markdown("""
This timeline highlights pivotal moments in Enron’s collapse, from early projects like Nordic Power Documentation to the California refund demand and the Dynegy merger announcement. It shows the rapid escalation of crises leading to bankruptcy.
""")

events = [
    {"Date": "1999-12-07", "Event": "Nordic Power Documentation Project"},
    {"Date": "2000-04-14", "Event": "Project Boomerang Initiated"},
    {"Date": "2000-12-01", "Event": "Governor Davis Proposals Go Public"},
    {"Date": "2001-06-20", "Event": "Davis Demands $9B Refund"},
    {"Date": "2001-11-09", "Event": "Enron-Dynegy Merger Announced"},
    {"Date": "2002-01-09", "Event": "Auction of Enron Trading Assets"},
]

events_df = pd.DataFrame(events)
events_df['Date'] = pd.to_datetime(events_df['Date'])

fig = px.scatter(events_df, x="Date", y="Event", title="Timeline of Enron’s Key Events (1999–2002)",
                 labels={"Date": "Date", "Event": "Key Event"}, size_max=10)
fig.update_traces(marker=dict(size=12, color='red'), text=events_df['Event'], textposition='top center')
fig.update_layout(showlegend=False, yaxis_title="")
st.plotly_chart(fig)

# Visualization 3: Bar Plot
st.subheader("3. Key Topic Frequencies")
st.markdown("""
This bar plot shows the frequency of key topics in the emails, such as 'California,' 'Dynegy,' and 'FERC.' The dominance of California and regulatory terms underscores the external pressures Enron faced during the energy crisis and investigations.
""")

keywords = ['California', 'Dynegy', 'FERC', 'lawsuit', 'merger', 'auction']
keyword_counts = {kw: 0 for kw in keywords}

for _, row in df.iterrows():
    text = (row['Subject'] + ' ' + row['Body']).lower()
    for kw in keywords:
        if kw.lower() in text:
            keyword_counts[kw] += 1

counts_df = pd.DataFrame(list(keyword_counts.items()), columns=['Keyword', 'Count'])

fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(data=counts_df, x='Keyword', y='Count', palette='viridis', ax=ax)
plt.title("Frequency of Key Topics in Enron Emails")
plt.xlabel("Topic")
plt.ylabel("Number of Emails")
plt.xticks(rotation=45)
st.pyplot(fig)

# Visualization 4: Pie Chart
st.subheader("4. Email Distribution by Sender")
st.markdown("""
This pie chart shows the distribution of emails by sender, highlighting key communicators like Jeff Dasovich and Susan Mara. Their prominence reflects their critical roles in navigating Enron’s strategic and regulatory challenges.
""")

sender_counts = Counter(df['From'])
sender_df = pd.DataFrame(sender_counts.items(), columns=['Sender', 'Count'])

fig, ax = plt.subplots(figsize=(8, 8))
plt.pie(sender_df['Count'], labels=sender_df['Sender'], autopct='%1.1f%%', colors=sns.color_palette('pastel'))
plt.title("Distribution of Emails by Sender")
st.pyplot(fig)

# Conclusion
st.subheader("Conclusion")
st.markdown("""
The visualizations reveal Enron’s complex web of ambition and collapse. The network graph shows a tightly knit communication structure, with Jeff Dasovich and Susan Mara at its core. The timeline traces the escalating crises, from California’s refund demands to the failed Dynegy merger. The bar plot highlights the regulatory and strategic pressures, while the pie chart underscores key individuals’ roles. Together, they tell a story of a company undone by overreach, regulatory scrutiny, and operational chaos, culminating in one of the largest corporate failures in history.
""")