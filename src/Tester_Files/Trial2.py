import streamlit as st
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx
from datetime import datetime
import re
from collections import Counter, defaultdict
import numpy as np
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from textblob import TextBlob

# Configure Streamlit page
st.set_page_config(
    page_title="Enron Email Story: The Rise and Fall",
    page_icon="📧",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .section-header {
        font-size: 1.8rem;
        color: #ff7f0e;
        margin: 2rem 0 1rem 0;
        border-bottom: 2px solid #ff7f0e;
        padding-bottom: 0.5rem;
    }
    .insight-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .metric-container {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_and_process_data():
    """Load and process the email data from the actual JSON data"""

    # The actual JSON data from your provided dataset
    raw_data = [
        {
            "Message-ID": "<22289471.1075843076105.JavaMail.evans@thyme>",
            "Date": "14.04.2000 09:58:00",
            "From": "frank.vickers@enron.com",
            "To": "jeff.dasovich@enron.com",
            "Subject": "Re: Project Boomerang",
            "Mime-Version": "1.0",
            "Content-Type": "text/plain; charset=us-ascii",
            "Content-Transfer-Encoding": "7bit",
            "X-From": "Frank W Vickers",
            "X-To": "Jeff Dasovich",
            "X-cc": "",
            "X-bcc": "",
            "X-Folder": "\\Jeff_Dasovich_Dec2000\\Notes Folders\\Boomerang",
            "X-Origin": "DASOVICH-J",
            "X-FileName": "jdasovic.nsf",
            "Body": "Jeff, thanks for the response.  Consider yourself a member of Project \nBoomerang.  I suggest that we try to talk sometime Monday and I can fill in \nsome of the detail on this effort.  I did hear that the lawsuit was filed \nunder Section 5 of AB1890.  Does that make any sense to you ?  I will call \nyou Monday morning to discuss the project and the role that you can play to \nhelp the effort.\n\nThanks\n\n\nFrank\n\n\nJeff Dasovich@EES on 04/14/2000 04:36:35 PM\nTo: Frank W Vickers/HOU/ECT@ECT\ncc: Paul Kaufman@ECT \nSubject: Re: Project Boomerang  \n\nFrank:\n\nSorry I didn't get back to you sooner.  We're finalizing a deal to (among \nother things) have Socalgas unbundle its in-state transport, storage and \nbalancing, sell off the transport and storage rights in an open season and \ncreate a secondary market for each.  They will also permit trading of \nimbalances on their system.  The deal's getting finalized today. (Paul, \nyour pal Joe Karp has joined the deal.)\n\nThe deal is currently scheduled to take effect beginning April 1, 2000.  \nStephanie Miller's been kept in the loop.  May be something that folks might \nwant include in due diligence w.r.t. to any gas contracts tied to the QFs.\n\nYour questions:\n\n1.  I am aware that the PUC just filed suit at FERC against El Paso for \nselling about 1.2 Bcf/day of gas capacity into California to Merchant Energy, \nbut hadn't heard about a suit regarding the QFs.  I will find out and let you \nknow.\n\n2.  Yes, I am in a position to follow the debate and keep you informed.  I'm \nassuming that your are up to speed on the debate as of today.  If not, let me \nknow and I and/or Paul can fill you in.\n\nIf there's anything else, don't hesitate to holler.  (415.782.7822).  Good \nweekend to you both.\n\nBest,\nJeff\n\n\nFrank W Vickers@ECT\n04/14/2000 03:20 PM\nTo: Jeff Dasovich/SFO/EES@EES\ncc: Paul Kaufman \nSubject: Project Boomerang\n\nJeff, as you may know we are conducting due diligence on 11 california QF's \nthat El Paso Merchant Energy purchased from Dynegy on Feb 1 of this year.  we \nare considering the purchase of 50% of El Paso's interest.  In discussions \nwith Paul he suggested that I get you involved with the transaction.  Two \nthings come to immediate attention.\n\nI heard that the PUC has filed suit against El Paso relative to these 11 \nassets.  What do you know about this ?\nNOw that the CPUC is trying to implement terms of AB 1890's section 390 alot \nof questions and methods are surfacing about calculations of payments to \nQF's.  Are you in a postion to follow this debate and keep me informed during \nthe process ??\n\n\nThanks\n\nFrank\n\n503-464-3940",
            "SourceFile": "1. 2.txt"
        },
        {
            "Message-ID": "<33363905.1075841338554.JavaMail.evans@thyme>",
            "Date": "10.04.2002 09:52:45",
            "From": "jr..legal@enron.com",
            "To": "dl-ga-all_enron_worldwide1@enron.com",
            "Subject": "Document Retention",
            "Mime-Version": "1.0",
            "Content-Type": "text/plain; charset=us-ascii",
            "Content-Transfer-Encoding": "quoted-printable",
            "X-From": "Legal - Rob Walls, Jr. </O=ENRON/OU=NA/CN=RECIPIENTS/CN=MBX_ANNCLEGAL>",
            "X-To": "DL-GA-all_enron_worldwide1 </O=ENRON/OU=NA/CN=RECIPIENTS/CN=DL-GA-all_enron_worldwide1>",
            "X-cc": "",
            "X-bcc": "",
            "X-Folder": "\\ExMerge - Motley, Matt\\Inbox\\Broker Quotes",
            "X-Origin": "MOTLEY-M",
            "X-FileName": "matt motley 6-26-02.pst",
            "Body": "As stated in my earlier email, the Company is under an obligation to retain, among other documents identified in the Bankruptcy Court's February 15, 2002 Order, a copy of which you all have received by email, all documents that relate to a pending or threatened investigation or lawsuit. As I am sure you are aware from reading the newspapers and watching television, because of the number of investigations and lawsuits the universe of relevant documents in just this one category covered by the Order is very large. You must retain all relevant company-related documents until these actual or threatened lawsuits and investigations are over. No one can then second-guess whether you destroyed a relevant document. However, many of you have requested guidance on what must be retained in this category of documents. To that end, attached are the various subject matters of subpoenas and document requests that we have received. While you should be sure to take the time to review the subpoenas and document requests themselves, some of the topics covered are the following:\n1.  All special purpose entities (including, but not limited to, Whitewing, Marlin, Atlantic, Osprey, Braveheart, Yosemite, MEGS, Margaux, Backbone, Nahanni, Moose, Fishtail, and Blackhawk)\n2. All LJM entities\n3. Chewco\n4. JEDI I and II\n5. The Raptor structures\n6. Related party transactions\n7. Portland General acquisition\n8. Elektro acquisition\n9. Cuiaba project\n10. Nowa Sarzyna project\n11. Dabhol project\n12. The Dynegy merger\n13. All accounting records\n14. All structured finance documents\n15. Audit records\n16. All records relating to purchases or sales of Enron stock\n17. All records relating to Enron stock options\n18. All records relating to the Enron Savings Plan, Cash Balance Plan, ESOP, and any other employee benefit plans \n19. Communications with analysts\n20. Communications with investors\n21. Communications with credit rating agencies\n22. All documents relating to California\n23. All documents relating to Rio Piedras\n24. All documents relating to pipeline safety\n25. All corporate tax documents\n26. All structured finance documents\n27. ENA collateralized loan obligations\n28. All periodic reports to management (including, but not limited to, VAR Reports, Daily Position Reports, Capital Portfolio Statements, Merchant Portfolio Statements, and Earnings Flash Reports)\n29. All press releases and records of public statements\n30. All DASHs\n31. All policy manuals\n32. All records relating to political contributions\n33. All documents relating to or reflecting communications with the SEC, CFTC, FERC, or DOL\n34. All documents relating to Enron's dark fiber optic cable \n35. Mariner\n36. Matrix\n37. ECT Securities\n38. Enron Online\n39. All documents relating to the Enron PAC\n40. All documents reflecting any communication with any federal agency, Congress, or the Executive Office of the President\n41. All documents relating to Enron Broadband\n42. Drafts and non-identical duplicates relating to any of the foregoing.\nThough lengthy, this list is not inclusive. Please review the subpoenas and document requests for the precise topics covered. If you have any questions please contact Bob Williams, the Company's Litigation Manager, at (713) 345-2402 or email him at Robert.C.Williams@enron.com.\nAs always, thank you for your patience during this challenging time.",
            "SourceFile": "1. 3.txt"
        }
        # Continue with all the emails from your JSON data...
    ]

    # Process all the data from the complete JSON
    return process_email_data(raw_data)


def extract_companies_and_topics(df):
    """Extract company mentions and key topics from emails dynamically"""
    # Extract companies and topics from the actual email content
    all_text = ' '.join((df['subject'].fillna('') + ' ' + df['body'].fillna('')).astype(str))
    all_text_lower = all_text.lower()

    # Dynamically find company names from email domains and content
    companies = set()

    # Extract from email domains
    for domain in df['from_domain'].dropna().unique():
        if domain != 'unknown' and '.' in domain:
            company_name = domain.split('.')[0].title()
            if len(company_name) > 2:  # Filter out short domain parts
                companies.add(company_name)

    # Add manually identified companies from the content
    energy_companies = ['Enron', 'Dynegy', 'Duke', 'Reliant', 'PG&E', 'Williams', 'El Paso',
                        'Calpine', 'Southern', 'TXU', 'Sempra', 'Edison', 'FERC', 'BP',
                        'Chevron', 'Texaco', 'Mirant', 'AEP', 'Cinergy', 'Conoco', 'Shell']

    for company in energy_companies:
        if company.lower() in all_text_lower:
            companies.add(company)

    # Extract topics dynamically by finding frequent meaningful words
    import re
    words = re.findall(r'\w+', all_text_lower)
    word_freq = Counter(words)

    # Filter for meaningful topics (excluding common words)
    stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
                  'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before',
                  'after', 'above', 'below', 'between', 'among', 'is', 'are', 'was',
                  'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does',
                  'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must',
                  'can', 'this', 'that', 'these', 'those', 'a', 'an', 'as', 'if', 'each',
                  'we', 'you', 'they', 'them', 'their', 'our', 'your', 'my', 'his', 'her',
                  'please', 'thanks', 'thank', 'regards', 'best', 'sincerely', 'email',
                  'message', 'sent', 'received', 'subject', 'dear', 'hi', 'hello'}

    meaningful_words = {word: freq for word, freq in word_freq.items()
                        if len(word) > 3 and word not in stop_words and freq > 2}

    # Get top topics by frequency
    top_topics = sorted(meaningful_words.items(), key=lambda x: x[1], reverse=True)[:20]
    topics = [word.title() for word, freq in top_topics]

    # Count actual occurrences in the dataset
    company_counts = Counter()
    topic_counts = Counter()

    for _, row in df.iterrows():
        text = (str(row['subject']) + ' ' + str(row['body'])).lower()

        for company in companies:
            if company.lower() in text:
                company_counts[company] += 1

        for topic in topics:
            if topic.lower() in text:
                topic_counts[topic] += 1

    return company_counts, topic_counts


def create_network_graph(df):
    """Create a network graph of email communications"""
    G = nx.Graph()

    # Add edges between sender and recipients
    for _, row in df.iterrows():
        sender = row['from_domain']
        for recipient in row['to_domains']:
            if sender != recipient and sender != 'unknown' and recipient != 'unknown':
                if G.has_edge(sender, recipient):
                    G[sender][recipient]['weight'] += 1
                else:
                    G.add_edge(sender, recipient, weight=1)

    return G


@st.cache_data
def load_uploaded_data(uploaded_file):
    """Load data from uploaded JSON file"""
    try:
        raw_data = json.load(uploaded_file)
        return process_email_data(raw_data)
    except Exception as e:
        st.error(f"Error loading uploaded file: {e}")
        return pd.DataFrame()


def process_email_data(raw_data):
    """Process raw email data into DataFrame"""
    emails = []
    for email in raw_data:
        try:
            # Parse date
            date_str = email.get('Date', '')
            if date_str:
                try:
                    # Try different date formats
                    for fmt in ["%d.%m.%Y %H:%M:%S", "%d.%m.%Y %H:%M", "%Y-%m-%d %H:%M:%S"]:
                        try:
                            parsed_date = datetime.strptime(date_str, fmt)
                            break
                        except ValueError:
                            continue
                    else:
                        parsed_date = None
                except:
                    parsed_date = None
            else:
                parsed_date = None

            # Extract domain information
            from_email = email.get('From', '')
            from_domain = from_email.split('@')[-1] if '@' in from_email else 'unknown'

            # Process recipients
            to_emails_str = email.get('To', '')
            to_emails = [email.strip() for email in to_emails_str.split(',') if email.strip()]
            to_domains = [email.strip().split('@')[-1] if '@' in email.strip() else 'unknown'
                          for email in to_emails if email.strip()]

            emails.append({
                'message_id': email.get('Message-ID', ''),
                'date': parsed_date,
                'from_email': from_email,
                'from_domain': from_domain,
                'to_emails': to_emails,
                'to_domains': to_domains,
                'subject': email.get('Subject', ''),
                'body': email.get('Body', ''),
                'year': parsed_date.year if parsed_date else None,
                'month': parsed_date.month if parsed_date else None,
                'source_file': email.get('SourceFile', '')
            })
        except Exception as e:
            continue

    return pd.DataFrame(emails)


def main():
    # Header
    st.markdown('<h1 class="main-header">📧 The Enron Email Story: Rise, Crisis, and Fall</h1>',
                unsafe_allow_html=True)

    st.markdown("""
    <div class="insight-box">
    <strong>🔍 Dataset Overview:</strong> This visualization explores the Enron email corpus, 
    revealing the story of one of the largest corporate scandals in history through internal communications.
    The emails span from late 1999 to 2002, capturing the company's final years.
    </div>
    """, unsafe_allow_html=True)

    # Load data directly from the provided JSON
    with st.spinner("Loading and processing email data..."):
        df = load_and_process_data()

    if df.empty:
        st.error("No data could be loaded. Please check the JSON file.")
        return

    # Sidebar filters
    st.sidebar.markdown("## 🎛️ Filters")

    # Date range filter
    if not df['date'].isna().all():
        min_date = df['date'].min()
        max_date = df['date'].max()

        date_range = st.sidebar.date_input(
            "Select Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )

        # Filter dataframe by date
        if len(date_range) == 2:
            df_filtered = df[(df['date'] >= pd.Timestamp(date_range[0])) &
                             (df['date'] <= pd.Timestamp(date_range[1]))]
        else:
            df_filtered = df
    else:
        df_filtered = df

    # Main metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("📨 Total Emails", len(df_filtered))

    with col2:
        unique_senders = df_filtered['from_email'].nunique()
        st.metric("👥 Unique Senders", unique_senders)

    with col3:
        unique_domains = df_filtered['from_domain'].nunique()
        st.metric("🏢 Organizations", unique_domains)

    with col4:
        if not df_filtered['date'].isna().all():
            date_span = (df_filtered['date'].max() - df_filtered['date'].min()).days
            st.metric("📅 Time Span (Days)", date_span)
        else:
            st.metric("📅 Time Span", "Unknown")

    # Timeline Analysis
    st.markdown('<h2 class="section-header">📈 Timeline: The Unfolding Crisis</h2>',
                unsafe_allow_html=True)

    if not df_filtered['date'].isna().all():
        # Monthly email volume
        monthly_counts = df_filtered.groupby([df_filtered['date'].dt.year,
                                              df_filtered['date'].dt.month]).size().reset_index()
        monthly_counts['date'] = pd.to_datetime(monthly_counts[['date', 'month']].assign(day=1))
        monthly_counts = monthly_counts.rename(columns={0: 'count'})

        fig_timeline = px.line(monthly_counts, x='date', y='count',
                               title='Email Volume Over Time: Communication Intensity During Crisis',
                               labels={'count': 'Number of Emails', 'date': 'Date'})

        fig_timeline.add_annotation(
            x="2001-11-01", y=monthly_counts['count'].max() * 0.8,
            text="Enron-Dynegy Merger Announced",
            showarrow=True, arrowhead=2, arrowcolor="red"
        )

        fig_timeline.update_layout(height=400)
        st.plotly_chart(fig_timeline, use_container_width=True)

    # Company and Topic Analysis
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<h3 class="section-header">🏢 Key Players</h3>', unsafe_allow_html=True)
        company_counts, topic_counts = extract_companies_and_topics(df_filtered)

        if company_counts:
            companies_df = pd.DataFrame(list(company_counts.items()),
                                        columns=['Company', 'Mentions'])
            companies_df = companies_df.sort_values('Mentions', ascending=True)

            fig_companies = px.bar(companies_df, x='Mentions', y='Company',
                                   orientation='h',
                                   title='Company Mentions in Emails',
                                   color='Mentions',
                                   color_continuous_scale='viridis')
            fig_companies.update_layout(height=400)
            st.plotly_chart(fig_companies, use_container_width=True)

    with col2:
        st.markdown('<h3 class="section-header">🔑 Key Topics</h3>', unsafe_allow_html=True)

        if topic_counts:
            topics_df = pd.DataFrame(list(topic_counts.items()),
                                     columns=['Topic', 'Mentions'])
            topics_df = topics_df.sort_values('Mentions', ascending=True)

            fig_topics = px.bar(topics_df, x='Mentions', y='Topic',
                                orientation='h',
                                title='Topic Frequency in Emails',
                                color='Mentions',
                                color_continuous_scale='plasma')
            fig_topics.update_layout(height=400)
            st.plotly_chart(fig_topics, use_container_width=True)

    # Communication Network
    st.markdown('<h2 class="section-header">🕸️ Communication Network</h2>',
                unsafe_allow_html=True)

    G = create_network_graph(df_filtered)

    if G.nodes():
        # Calculate network metrics
        degrees = dict(G.degree())
        top_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:10]

        # Create network visualization data
        pos = nx.spring_layout(G, k=1, iterations=50)

        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        node_x = []
        node_y = []
        node_text = []
        node_size = []

        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(f"{node}<br>Connections: {degrees[node]}")
            node_size.append(degrees[node] * 3 + 10)

        fig_network = go.Figure()

        # Add edges
        fig_network.add_trace(go.Scatter(x=edge_x, y=edge_y,
                                         line=dict(width=0.5, color='#888'),
                                         hoverinfo='none',
                                         mode='lines'))

        # Add nodes
        fig_network.add_trace(go.Scatter(x=node_x, y=node_y,
                                         mode='markers+text',
                                         hoverinfo='text',
                                         text=[node.split('.')[0] for node in G.nodes()],
                                         textposition="middle center",
                                         hovertext=node_text,
                                         marker=dict(size=node_size,
                                                     color='lightblue',
                                                     line=dict(width=2, color='darkblue'))))

        fig_network.update_layout(title='Email Communication Network: Who Talks to Whom',
                                  showlegend=False,
                                  hovermode='closest',
                                  margin=dict(b=20, l=5, r=5, t=40),
                                  annotations=[dict(
                                      text="Node size represents communication frequency",
                                      showarrow=False,
                                      xref="paper", yref="paper",
                                      x=0.005, y=-0.002,
                                      xanchor='left', yanchor='bottom',
                                      font=dict(size=12)
                                  )],
                                  xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                  yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                  height=500)

        st.plotly_chart(fig_network, use_container_width=True)

        # Top communicators
        st.markdown("### 🔝 Most Connected Organizations")
        top_orgs_df = pd.DataFrame(top_nodes[:5], columns=['Organization', 'Connections'])
        st.dataframe(top_orgs_df, use_container_width=True)

    # Email Content Analysis
    st.markdown('<h2 class="section-header">📝 Content Analysis</h2>',
                unsafe_allow_html=True)

    # Word cloud
    if not df_filtered.empty:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### ☁️ Subject Line Word Cloud")
            all_subjects = ' '.join(df_filtered['subject'].dropna().astype(str))

            if all_subjects.strip():
                try:
                    wordcloud = WordCloud(width=400, height=300,
                                          background_color='white',
                                          colormap='viridis').generate(all_subjects)

                    fig_wc, ax = plt.subplots(figsize=(8, 6))
                    ax.imshow(wordcloud, interpolation='bilinear')
                    ax.axis('off')
                    st.pyplot(fig_wc)
                except Exception as e:
                    st.error(f"Could not generate word cloud: {e}")

        with col2:
            st.markdown("### 📊 Email Length Distribution")
            df_filtered['body_length'] = df_filtered['body'].astype(str).str.len()

            fig_length = px.histogram(df_filtered, x='body_length',
                                      title='Distribution of Email Body Lengths',
                                      labels={'body_length': 'Email Length (characters)',
                                              'count': 'Number of Emails'},
                                      nbins=30)
            fig_length.update_layout(height=300)
            st.plotly_chart(fig_length, use_container_width=True)

    # Key Events Timeline - Dynamic based on actual email content
    st.markdown('<h2 class="section-header">⏰ Key Events Timeline</h2>',
                unsafe_allow_html=True)

    # Extract key events from the actual email data
    key_events = []

    # Find significant dates and events from the email subjects and content
    for _, row in df_filtered.iterrows():
        if pd.isna(row['date']):
            continue

        subject = str(row['subject']).lower()
        body = str(row['body']).lower()
        text = subject + ' ' + body

        # Identify key event types based on content
        event_type = "Internal"
        event_desc = row['subject'][:50] + "..." if len(str(row['subject'])) > 50 else str(row['subject'])

        if any(word in text for word in ['merger', 'dynegy', 'acquisition']):
            event_type = "Corporate"
            if 'dynegy' in text and 'merger' in text:
                event_desc = "Enron-Dynegy merger discussions"
        elif any(word in text for word in ['lawsuit', 'legal', 'sec', 'investigation', 'bankruptcy']):
            event_type = "Legal"
            if 'document retention' in text:
                event_desc = "Document retention orders"
            elif 'sec' in text:
                event_desc = "SEC-related communications"
        elif any(word in text for word in ['california', 'crisis', 'energy', 'power']):
            event_type = "Market"
            if 'california' in text:
                event_desc = "California energy crisis"
        elif 'chairman.ken@enron.com' in str(row['from_email']):
            event_type = "Executive"
            event_desc = "Executive communication: " + event_desc

        key_events.append({
            "date": row['date'].strftime("%Y-%m-%d"),
            "event": event_desc,
            "type": event_type
        })

    # Get unique events and sort by date
    unique_events = {}
    for event in key_events:
        key = event['date'] + event['type']
        if key not in unique_events:
            unique_events[key] = event

    events_list = list(unique_events.values())
    events_list = sorted(events_list, key=lambda x: x['date'])[:15]  # Show top 15 events

    if events_list:
        events_df = pd.DataFrame(events_list)
        events_df['date'] = pd.to_datetime(events_df['date'])

        fig_events = px.scatter(events_df, x='date', y='type',
                                color='type', size=[1] * len(events_df),
                                hover_data=['event'],
                                title='Key Events Timeline from Email Communications')

        for i, row in events_df.iterrows():
            fig_events.add_annotation(
                x=row['date'], y=row['type'],
                text=row['event'][:25] + "..." if len(row['event']) > 25 else row['event'],
                showarrow=True, arrowhead=2,
                yshift=20, font=dict(size=9)
            )

        fig_events.update_layout(height=400)
        st.plotly_chart(fig_events, use_container_width=True)

    # Recent Emails Table
    st.markdown('<h2 class="section-header">📋 Sample Email Details</h2>',
                unsafe_allow_html=True)

    # Display sample emails
    display_cols = ['date', 'from_email', 'subject']
    sample_emails = df_filtered[display_cols].dropna().head(10)

    if not sample_emails.empty:
        st.dataframe(sample_emails, use_container_width=True)

    # Key Insights - Dynamic based on data analysis
    st.markdown('<h2 class="section-header">🔍 Key Insights</h2>',
                unsafe_allow_html=True)

    # Generate insights dynamically from the data
    insights = []

    # Timeline insights
    if not df_filtered['date'].isna().all():
        date_range = (df_filtered['date'].max() - df_filtered['date'].min()).days
        peak_month = df_filtered.groupby(df_filtered['date'].dt.to_period('M')).size().idxmax()
        peak_count = df_filtered.groupby(df_filtered['date'].dt.to_period('M')).size().max()
        insights.append(
            f"📈 **Timeline Analysis**: Email communications span {date_range} days, with peak activity in {peak_month} ({peak_count} emails).")

    # Company analysis insights
    if company_counts:
        top_company = company_counts.most_common(1)[0]
        insights.append(
            f"🏢 **Corporate Focus**: {top_company[0]} is the most mentioned entity with {top_company[1]} references, indicating its central role in communications.")

    # Communication network insights
    unique_domains = df_filtered['from_domain'].nunique()
    total_emails = len(df_filtered)
    avg_recipients = df_filtered['to_domains'].apply(len).mean()
    insights.append(
        f"🌐 **Network Scope**: Communications involve {unique_domains} different organizations across {total_emails} emails, with an average of {avg_recipients:.1f} recipients per email.")

    # Content insights
    if topic_counts:
        top_topic = topic_counts.most_common(1)[0]
        insights.append(
            f"🔑 **Key Themes**: '{top_topic[0]}' appears {top_topic[1]} times, highlighting its importance in the communications.")

    # Temporal insights
    if not df_filtered['date'].isna().all():
        early_emails = df_filtered[df_filtered['date'] < df_filtered['date'].quantile(0.3)]
        late_emails = df_filtered[df_filtered['date'] > df_filtered['date'].quantile(0.7)]

        if len(early_emails) > 0 and len(late_emails) > 0:
            early_topics = ' '.join(
                (early_emails['subject'].fillna('') + ' ' + early_emails['body'].fillna('')).astype(str)).lower()
            late_topics = ' '.join(
                (late_emails['subject'].fillna('') + ' ' + late_emails['body'].fillna('')).astype(str)).lower()

            if 'legal' in late_topics and 'legal' not in early_topics:
                insights.append(
                    "⚖️ **Crisis Evolution**: Legal and compliance themes emerge in later communications, indicating escalating regulatory concerns.")

            if 'dynegy' in late_topics:
                insights.append(
                    "🤝 **Strategic Shifts**: References to Dynegy increase in later emails, reflecting merger discussions during crisis period.")

    # Email patterns
    avg_length = df_filtered['body'].astype(str).str.len().mean()
    insights.append(
        f"📊 **Communication Patterns**: Average email length is {avg_length:.0f} characters, indicating {'detailed' if avg_length > 1000 else 'concise'} communication style.")

    # Executive communication insight
    executive_emails = df_filtered[
        df_filtered['from_email'].str.contains('chairman|ceo|president', case=False, na=False)]
    if len(executive_emails) > 0:
        insights.append(
            f"👔 **Executive Communications**: {len(executive_emails)} emails from executive level, showing leadership involvement in crisis management.")

    for insight in insights:
        st.markdown(f'<div class="insight-box">{insight}</div>', unsafe_allow_html=True)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.9em; margin-top: 2rem;'>
    📧 Enron Email Corpus Analysis | Data visualization reveals the human story behind corporate communications
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()