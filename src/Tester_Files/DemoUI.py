import streamlit as st
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import re
from collections import Counter

# Set page config
st.set_page_config(
    page_title="Enron Email Analysis",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Global styles */
    .main {
        padding-top: 1rem;
    }

    /* Custom font */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem 1rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }

    .main-header h1 {
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }

    .main-header p {
        font-size: 1.2rem;
        opacity: 0.9;
        font-weight: 300;
    }

    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #f8fafc 0%, #e2e8f0 100%);
    }

    /* Metric cards */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        border-left: 4px solid #667eea;
        margin: 1rem 0;
        transition: transform 0.2s ease;
    }

    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
    }

    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: #667eea;
        margin-bottom: 0.5rem;
    }

    .metric-label {
        font-size: 1rem;
        color: #64748b;
        font-weight: 500;
    }

    /* Theme cards */
    .theme-card {
        background: white;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 5px 20px rgba(0,0,0,0.08);
        border: 1px solid #e2e8f0;
        transition: all 0.3s ease;
    }

    .theme-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 30px rgba(0,0,0,0.12);
        border-color: #667eea;
    }

    .theme-title {
        font-size: 1.25rem;
        font-weight: 600;
        color: #1e293b;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
    }

    .theme-icon {
        margin-right: 0.5rem;
        font-size: 1.5rem;
    }

    .theme-content {
        color: #475569;
        line-height: 1.6;
    }

    /* Email cards */
    .email-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 3px 15px rgba(0,0,0,0.06);
        border-left: 4px solid #10b981;
        transition: all 0.3s ease;
    }

    .email-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
    }

    .email-header {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        margin-bottom: 1rem;
    }

    .email-subject {
        font-size: 1.1rem;
        font-weight: 600;
        color: #1e293b;
        margin-bottom: 0.5rem;
    }

    .email-meta {
        font-size: 0.875rem;
        color: #64748b;
    }

    .email-tags {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        margin-top: 1rem;
    }

    .tag {
        background: #f1f5f9;
        color: #475569;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 500;
    }

    .tag-classification {
        background: #dbeafe;
        color: #1d4ed8;
    }

    .tag-tone {
        background: #ecfdf5;
        color: #059669;
    }

    /* Phase timeline styling */
    .timeline-phase {
        background: white;
        border-radius: 15px;
        padding: 2rem;
        margin: 2rem 0;
        position: relative;
        box-shadow: 0 5px 20px rgba(0,0,0,0.08);
        border-left: 5px solid #667eea;
    }

    .phase-number {
        position: absolute;
        top: -15px;
        left: 20px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        width: 40px;
        height: 40px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        font-size: 1.2rem;
    }

    .phase-title {
        font-size: 1.5rem;
        font-weight: 600;
        color: #1e293b;
        margin-bottom: 1rem;
        margin-top: 1rem;
    }

    /* Chart containers */
    .chart-container {
        background: white;
        border-radius: 15px;
        padding: 1.5rem;
        box-shadow: 0 5px 20px rgba(0,0,0,0.08);
        margin: 1rem 0;
    }

    /* Filter section */
    .filter-section {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 3px 15px rgba(0,0,0,0.06);
        margin-bottom: 2rem;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 500;
        transition: all 0.3s ease;
    }

    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }

    /* Selectbox styling */
    .stSelectbox > div > div {
        border-radius: 8px;
        border: 2px solid #e2e8f0;
    }

    /* Success/Info boxes */
    .stSuccess {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        border: none;
        border-radius: 12px;
    }

    .stInfo {
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
        border: none;
        border-radius: 12px;
    }

    /* Hide streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Custom spacing */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# File path - update this to your actual file path
JSON_FILE_PATH = "enron_thematic_analysis_output1.json"


@st.cache_data
def load_data():
    """Load and parse the JSON data"""
    try:
        with open(JSON_FILE_PATH, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        st.error(f"File not found: {JSON_FILE_PATH}")
        st.stop()
    except json.JSONDecodeError:
        st.error("Invalid JSON file format")
        st.stop()


def parse_date(date_str):
    """Parse date string to datetime object"""
    try:
        if '.' in date_str:
            return datetime.strptime(date_str, "%d.%m.%Y %H:%M:%S")
        else:
            return datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
    except:
        return None


def create_modern_timeline_chart(emails_df):
    """Create modern timeline visualization"""
    emails_df['parsed_date'] = emails_df['date'].apply(parse_date)
    emails_df = emails_df.dropna(subset=['parsed_date'])
    emails_df['year_month'] = emails_df['parsed_date'].dt.to_period('M')

    timeline_data = emails_df.groupby(['year_month', 'classification']).size().reset_index(name='count')
    timeline_data['year_month_str'] = timeline_data['year_month'].astype(str)

    fig = px.bar(timeline_data,
                 x='year_month_str',
                 y='count',
                 color='classification',
                 title="📈 Email Volume Timeline",
                 color_discrete_sequence=px.colors.qualitative.Set3)

    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_family="Inter",
        title_font_size=20,
        title_font_color="#1e293b",
        xaxis_tickangle=-45,
        xaxis=dict(showgrid=True, gridcolor='#f1f5f9'),
        yaxis=dict(showgrid=True, gridcolor='#f1f5f9'),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    return fig


def create_modern_donut_chart(emails_df, column, title, colors=None):
    """Create modern donut chart"""
    value_counts = emails_df[column].value_counts()

    fig = go.Figure(data=[go.Pie(
        labels=value_counts.index,
        values=value_counts.values,
        hole=.6,
        textinfo='label+percent',
        textposition='outside',
        marker_colors=colors or px.colors.qualitative.Set3
    )])

    fig.update_layout(
        title=title,
        title_font_size=20,
        title_font_color="#1e293b",
        font_family="Inter",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        showlegend=False,
        annotations=[dict(text=f'{len(emails_df)}<br>Total', x=0.5, y=0.5, font_size=20, showarrow=False)]
    )
    return fig


def create_entity_chart(entity_data, title, color):
    """Create horizontal bar chart for entities"""
    if not entity_data:
        return None

    df = pd.DataFrame(entity_data, columns=['Entity', 'Count'])

    fig = px.bar(df,
                 x='Count',
                 y='Entity',
                 orientation='h',
                 title=title,
                 color_discrete_sequence=[color])

    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_family="Inter",
        title_font_size=18,
        title_font_color="#1e293b",
        yaxis={'categoryorder': 'total ascending'},
        xaxis=dict(showgrid=True, gridcolor='#f1f5f9'),
        yaxis_title="",
        height=400
    )
    return fig


def main():
    # Main header
    st.markdown("""
    <div class="main-header">
        <h1>⚡ Enron Email Analysis</h1>
        <p>Deep dive into the corporate scandal that shocked the world</p>
    </div>
    """, unsafe_allow_html=True)

    # Load data
    data = load_data()
    emails_df = pd.DataFrame(data['email_summaries'])
    metadata = data['analysis_metadata']

    # Sidebar with enhanced metrics
    with st.sidebar:
        st.markdown("### 📊 **Analysis Overview**")

        # Custom metric cards
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{metadata['total_emails']}</div>
            <div class="metric-label">Total Emails Analyzed</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{metadata['model_used'].title()}</div>
            <div class="metric-label">AI Model Used</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{len(emails_df['classification'].unique())}</div>
            <div class="metric-label">Email Categories</div>
        </div>
        """, unsafe_allow_html=True)

        # Quick stats
        st.markdown("### 📈 **Quick Stats**")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Crisis Emails", len(emails_df[emails_df['tone'] == 'Concerned/Critical']))
        with col2:
            st.metric("Business Emails", len(emails_df[emails_df['classification'].str.contains('Business', na=False)]))

        # Analysis date
        st.markdown(f"**📅 Analysis Date:** {metadata['analysis_timestamp'][:10]}")

    # Enhanced tabs with icons
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🎯 **Themes**",
        "📊 **Analytics**",
        "📧 **Emails**",
        "🌐 **Entities**",
        "📖 **Timeline**"
    ])

    with tab1:
        st.markdown("## 🎯 **Major Themes in the Enron Scandal**")
        st.markdown("*Click on each theme to explore the key findings and evidence*")

        # Theme data with icons
        themes_data = [
            ("🤝", "The Dynegy Merger Saga", "The failed merger that could have saved Enron"),
            ("⚡", "California Energy Crisis", "Market manipulation and price fixing schemes"),
            ("💰", "Financial Engineering", "Special Purpose Entities and off-balance-sheet transactions"),
            ("📉", "The Unraveling", "The beginning of the end and bankruptcy filing"),
            ("⚖️", "Legal Battles", "Regulatory investigations and criminal charges"),
            ("📈", "Energy Trading", "The core business that became a house of cards"),
            ("🏢", "Corporate Culture", "Internal dynamics that enabled the fraud")
        ]

        themes_text = data['themes_analysis']
        theme_sections = re.split(r'\d+\.\s+', themes_text)[1:]

        for i, ((icon, title, subtitle), content) in enumerate(zip(themes_data, theme_sections)):
            with st.expander(f"{icon} **{title}**", expanded=False):
                st.markdown(f"*{subtitle}*")
                st.markdown("---")
                st.write(content.strip())

    with tab2:
        st.markdown("## 📊 **Email Analytics Dashboard**")

        # Top row - main charts
        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            timeline_fig = create_modern_timeline_chart(emails_df)
            st.plotly_chart(timeline_fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            tone_fig = create_modern_donut_chart(emails_df, 'tone', '🎭 Email Tone Distribution')
            st.plotly_chart(tone_fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # Bottom row - classification chart
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown("### 📁 **Email Classification Breakdown**")
        class_counts = emails_df['classification'].value_counts()
        class_fig = px.bar(
            x=class_counts.values,
            y=class_counts.index,
            orientation='h',
            title="Distribution by Email Type",
            color_discrete_sequence=['#667eea']
        )
        class_fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_family="Inter",
            yaxis={'categoryorder': 'total ascending'},
            height=500
        )
        st.plotly_chart(class_fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with tab3:
        st.markdown("## 📧 **Email Explorer**")

        # Enhanced filter section
        st.markdown('<div class="filter-section">', unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            selected_classification = st.selectbox(
                "📁 Classification",
                ["All"] + list(emails_df['classification'].unique())
            )

        with col2:
            selected_tone = st.selectbox(
                "🎭 Tone",
                ["All"] + list(emails_df['tone'].unique())
            )

        with col3:
            search_term = st.text_input("🔍 Search", placeholder="Enter keywords...")

        with col4:
            sort_by = st.selectbox("📊 Sort by", ["Date", "ID", "Classification"])

        st.markdown('</div>', unsafe_allow_html=True)

        # Apply filters
        filtered_df = emails_df.copy()
        if selected_classification != "All":
            filtered_df = filtered_df[filtered_df['classification'] == selected_classification]
        if selected_tone != "All":
            filtered_df = filtered_df[filtered_df['tone'] == selected_tone]
        if search_term:
            filtered_df = filtered_df[filtered_df['summary'].str.contains(search_term, case=False, na=False)]

        # Results summary
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info(f"📊 **{len(filtered_df)}** emails found")
        with col2:
            if len(filtered_df) > 0:
                avg_length = filtered_df['summary'].str.len().mean()
                st.success(f"📝 **{avg_length:.0f}** avg chars")
        with col3:
            if len(filtered_df) > 0:
                crisis_pct = (len(filtered_df[filtered_df['tone'] == 'Concerned/Critical']) / len(filtered_df) * 100)
                st.warning(f"⚠️ **{crisis_pct:.0f}%** critical tone")

        # Display emails with modern cards
        for _, email in filtered_df.head(20).iterrows():  # Limit to 20 for performance
            st.markdown(f"""
            <div class="email-card">
                <div class="email-header">
                    <div>
                        <div class="email-subject">📧 {email['subject']}</div>
                        <div class="email-meta">
                            📅 {email['date']} | 🆔 ID: {email['id']}
                        </div>
                    </div>
                </div>
                <div class="email-content">
                    <p>{email['summary']}</p>
                </div>
                <div class="email-tags">
                    <span class="tag tag-classification">{email['classification']}</span>
                    <span class="tag tag-tone">{email['tone']}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

    with tab4:
        st.markdown("## 🌐 **Entity Analysis**")
        st.markdown("*Key players, organizations, and locations in the Enron scandal*")

        # Calculate entity data
        all_orgs, all_people, all_locations = [], [], []
        for _, row in emails_df.iterrows():
            entities = row['entities']
            all_orgs.extend(entities.get('organizations', []))
            all_people.extend(entities.get('people', []))
            all_locations.extend(entities.get('locations', []))

        entity_data = {
            'organizations': Counter(all_orgs).most_common(10),
            'people': Counter(all_people).most_common(10),
            'locations': Counter(all_locations).most_common(10)
        }

        # Three columns for entity charts
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            org_fig = create_entity_chart(entity_data['organizations'], '🏢 Top Organizations', '#667eea')
            if org_fig:
                st.plotly_chart(org_fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            people_fig = create_entity_chart(entity_data['people'], '👥 Key People', '#10b981')
            if people_fig:
                st.plotly_chart(people_fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col3:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            loc_fig = create_entity_chart(entity_data['locations'], '📍 Locations', '#f59e0b')
            if loc_fig:
                st.plotly_chart(loc_fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

    with tab5:
        st.markdown("## 📖 **The Enron Timeline**")
        st.markdown("*From corporate success to spectacular collapse*")

        timeline_text = data['narrative_timeline']
        phases = re.split(r'\*\*\d+\.', timeline_text)[1:]

        phase_data = [
            ("📈", "Early Operations", "January 1992 - August 2000", "#10b981"),
            ("⚠️", "Growing Problems", "September 2000 - December 2001", "#f59e0b"),
            ("🚨", "Crisis Phase", "January 2002 - March 2002", "#ef4444"),
            ("💥", "Collapse", "April 2002 - December 2006", "#7c3aed")
        ]

        for i, ((icon, title, period, color), content) in enumerate(zip(phase_data, phases)):
            st.markdown(f"""
            <div class="timeline-phase">
                <div class="phase-number">{i + 1}</div>
                <div class="phase-title">{icon} {title}</div>
                <div style="color: #64748b; margin-bottom: 1rem; font-weight: 500;">{period}</div>
            """, unsafe_allow_html=True)

            # Clean and display content
            content = content.replace('**', '').strip()
            sections = content.split('- Description:')

            if len(sections) >= 2:
                description = sections[1].split('- Connection to the broader narrative:')[0].strip()
                st.write(description)

                if '- Connection to the broader narrative:' in sections[1]:
                    connection = sections[1].split('- Connection to the broader narrative:')[1].strip()
                    st.markdown(f"**🔗 Connection:** {connection}")
            else:
                st.write(content)

            st.markdown('</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
