import streamlit as st
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import networkx as nx
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from langchain.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from collections import defaultdict, Counter
import re
from typing import List, Dict, Tuple
import numpy as np
import os
import hashlib

from src.config.config import PROCESSED_JSON_OUTPUT_100


# Initialize Ollama with Mistral
@st.cache_resource
def init_llm():
    return Ollama(model="mistral", temperature=0.7)


class EnronEmailAnalyzer:
    def __init__(self, email_data: List[Dict], data_path: str):
        self.emails = email_data
        self.data_path = data_path
        self.llm = init_llm()
        self.df = pd.DataFrame(email_data)
        self.df['date'] = pd.to_datetime(self.df['date'], format='%d.%m.%Y %H:%M:%S', errors='coerce')

    def analyze_individual_email(self, email: Dict) -> str:
        """Analyze a single email using LLM - no assumptions"""
        prompt = PromptTemplate(
            input_variables=["sender", "recipient", "date", "subject", "summary", "tone", "classification", "entities"],
            template="""
            You are a detective analyzing corporate emails. Examine this email without any preconceptions:

            From: {sender}
            To: {recipient}
            Date: {date}
            Subject: {subject}
            Summary: {summary}
            Tone: {tone}
            Classification: {classification}
            Mentioned Entities: {entities}

            Analyze this email like you're discovering it for the first time:
            1. What seems significant or unusual about this communication?
            2. What relationships or power dynamics are revealed?
            3. Are there any warning signs, red flags, or interesting patterns?
            4. What questions does this email raise?
            5. How might this connect to other events (without assuming what those are)?

            Be investigative and curious. Point out anything noteworthy.
            Keep analysis to 3-4 sentences.
            """
        )

        chain = LLMChain(llm=self.llm, prompt=prompt)

        # Prepare entities string
        entities = email.get('entities', {})
        entities_str = f"People: {', '.join(entities.get('people', [])[:5])}, " \
                       f"Orgs: {', '.join(entities.get('organizations', [])[:5])}, " \
                       f"Projects: {', '.join(entities.get('projects', [])[:3])}"

        try:
            analysis = chain.run(
                sender=email.get('from', 'Unknown'),
                recipient=email.get('to', 'Unknown'),
                date=email.get('date', 'Unknown'),
                subject=email.get('subject', 'No subject'),
                summary=email.get('summary', 'No summary'),
                tone=email.get('tone_analysis', 'Unknown'),
                classification=email.get('classification', 'Unknown'),
                entities=entities_str
            )
            return analysis
        except Exception as e:
            return f"Analysis failed: {str(e)}"

    def generate_narrative(self, email_analyses: List[Tuple[Dict, str]]) -> str:
        """Generate overall narrative using LLM - completely data-driven"""
        # First, let LLM analyze the overall pattern
        pattern_prompt = PromptTemplate(
            input_variables=["email_analyses"],
            template="""
            You are analyzing emails from Enron's final period. Based on these individual email analyses, 
            identify the overall story arc and key patterns:

            {email_analyses}

            Identify:
            1. The major story arc (beginning, crisis points, resolution)
            2. Key players and their roles
            3. Critical turning points
            4. Cause and effect relationships
            5. The human drama elements

            Provide a structured analysis of what story these emails tell collectively.
            """
        )

        # Prepare email analyses text
        analyses_text = "\n\n".join([
            f"Email {i + 1} ({email.get('date', 'Unknown date')}): {email.get('subject', 'No subject')}\n"
            f"Analysis: {analysis}"
            for i, (email, analysis) in enumerate(email_analyses)
        ])

        pattern_chain = LLMChain(llm=self.llm, prompt=pattern_prompt)
        pattern_analysis = pattern_chain.run(email_analyses=analyses_text[:3000])  # Limit tokens

        # Now generate the final narrative
        narrative_prompt = PromptTemplate(
            input_variables=["pattern_analysis", "timeline_data"],
            template="""
            Based on your analysis of Enron emails, create a compelling documentary-style narrative.

            Pattern Analysis:
            {pattern_analysis}

            Timeline Data:
            {timeline_data}

            Write a comprehensive story that:
            1. Tells the complete story found in the emails
            2. Highlights discoveries and surprises
            3. Connects all the dots between different events
            4. Explains the significance of what happened
            5. Captures both the business and human elements

            Make it engaging and let the data tell its own story. Don't assume any prior knowledge.
            Structure it like chapters in a documentary, with clear progression.
            """
        )

        # Group emails by time period for timeline
        timeline_groups = self.group_emails_by_period()
        timeline_text = ""
        for period, emails in timeline_groups.items():
            timeline_text += f"\n{period}:\n"
            for email in emails[:3]:
                timeline_text += f"- {email['subject']}: {email['summary'][:100]}...\n"

        chain = LLMChain(llm=self.llm, prompt=narrative_prompt)

        narrative = chain.run(
            pattern_analysis=pattern_analysis,
            timeline_data=timeline_text
        )

        return narrative

    def save_story_to_json(self, story: str, analyses: List[Tuple[Dict, str]]):
        """Save the generated story back to the JSON file"""
        # Create a story object
        story_data = {
            "generated_story": story,
            "generation_timestamp": datetime.now().isoformat(),
            "email_analyses": [
                {
                    "email_id": email.get("email_id"),
                    "analysis": analysis
                }
                for email, analysis in analyses
            ],
            "metadata": {
                "total_emails_analyzed": len(analyses),
                "model_used": "mistral",
                "version": "1.0"
            }
        }

        # Read existing data
        with open(self.data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Add story to the data
        if isinstance(data, dict):
            data['enron_story'] = story_data
        else:
            # If it's a list of emails, convert to dict
            data = {
                'emails': data,
                'enron_story': story_data
            }

        # Save back to file
        with open(self.data_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        return story_data

    def load_existing_story(self):
        """Check if story already exists in JSON"""
        try:
            with open(self.data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if isinstance(data, dict) and 'enron_story' in data:
                return data['enron_story']
        except:
            pass
        return None

    def group_emails_by_period(self) -> Dict:
        """Group emails by time period"""
        periods = defaultdict(list)

        for _, email in self.df.iterrows():
            if pd.notna(email['date']):
                year_month = email['date'].strftime('%Y-%m')
                periods[year_month].append(email.to_dict())

        return dict(sorted(periods.items()))

    def extract_key_themes(self) -> List[str]:
        """Let LLM extract key themes from emails"""
        # Prepare email summaries for analysis
        email_summaries = "\n".join([
            f"- {email.get('date', 'Unknown date')}: {email.get('subject', 'No subject')} - {email.get('summary', '')[:200]}"
            for email in self.emails[:30]  # Sample for theme extraction
        ])

        theme_prompt = PromptTemplate(
            input_variables=["email_summaries"],
            template="""
            Analyze these Enron email summaries and identify the major themes and patterns:

            {email_summaries}

            Extract and list the 8-10 most significant recurring themes, events, or topics.
            Focus on identifying patterns, crises, business relationships, and major developments.
            List them in order of importance, separated by semicolons.
            """
        )

        chain = LLMChain(llm=self.llm, prompt=theme_prompt)

        try:
            themes_text = chain.run(email_summaries=email_summaries)
            # Split themes by semicolon and clean them
            themes = [theme.strip() for theme in themes_text.split(';') if theme.strip()]
            return themes[:10]  # Return top 10 themes
        except:
            return ["Email analysis", "Business communications", "Corporate events"]

    def create_timeline_visualization(self):
        """Create timeline visualization"""
        timeline_df = self.df[self.df['date'].notna()].copy()
        timeline_df = timeline_df.sort_values('date')

        # Add event importance based on classification
        importance_map = {
            'Regulatory Alert / Crisis Communication': 3,
            'Legal/Compliance Matter': 3,
            'Business Project Coordination': 2,
            'Internal Communication': 1,
            'Market/Trading Information': 2,
            'Strategic Planning': 2
        }

        timeline_df['importance'] = timeline_df['classification'].map(
            lambda x: importance_map.get(x, 1)
        )

        fig = px.scatter(timeline_df,
                         x='date',
                         y='importance',
                         color='classification',
                         size='importance',
                         hover_data=['subject', 'summary'],
                         title='Enron Email Timeline',
                         labels={'importance': 'Event Importance', 'date': 'Date'})

    def extract_key_events(self) -> List[Dict]:
        """Let LLM identify key events from the emails"""
        # Get emails sorted by date
        sorted_emails = self.df[self.df['date'].notna()].sort_values('date')

        # Prepare data for LLM analysis
        email_timeline = "\n".join([
                                       f"{row['date'].strftime('%Y-%m-%d')}: {row['subject']} ({row['classification']}) - {row['summary'][:150]}..."
                                       for _, row in sorted_emails.iterrows()
                                   ][:50])  # Limit to prevent token overflow

        events_prompt = PromptTemplate(
            input_variables=["email_timeline"],
            template="""
            Analyze this chronological list of Enron emails and identify the most critical events:

            {email_timeline}

            Identify the 5-7 most significant events or turning points in Enron's story based on these emails.
            For each event, provide:
            - Date (YYYY-MM-DD format)
            - Event name (brief description)
            - Significance level (1-3, where 3 is most critical)

            Format: Date|Event|Significance
            One event per line.
            """
        )

        chain = LLMChain(llm=self.llm, prompt=events_prompt)

        try:
            events_text = chain.run(email_timeline=email_timeline)
            events = []

            for line in events_text.strip().split('\n'):
                if '|' in line:
                    parts = line.split('|')
                    if len(parts) >= 3:
                        events.append({
                            'date': parts[0].strip(),
                            'event': parts[1].strip(),
                            'significance': float(parts[2].strip()) if parts[2].strip().replace('.',
                                                                                                '').isdigit() else 2
                        })

            return events
        except Exception as e:
            return []

        return fig

    def create_network_graph(self):
        """Create network visualization of email communications"""
        G = nx.Graph()

        # Add edges based on email communications
        for email in self.emails:
            sender = email.get('from', '').split('@')[0]
            recipients = email.get('to', '').split(',')

            if sender:
                for recipient in recipients[:5]:  # Limit recipients
                    rec_name = recipient.strip().split('@')[0]
                    if rec_name and rec_name != sender:
                        G.add_edge(sender, rec_name)

        # Calculate centrality
        centrality = nx.degree_centrality(G)

        # Create positions
        pos = nx.spring_layout(G, k=1, iterations=50)

        # Create edge trace
        edge_trace = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_trace.append(go.Scatter(x=[x0, x1, None],
                                         y=[y0, y1, None],
                                         mode='lines',
                                         line=dict(width=0.5, color='#888'),
                                         hoverinfo='none'))

        # Create node trace
        node_trace = go.Scatter(
            x=[pos[node][0] for node in G.nodes()],
            y=[pos[node][1] for node in G.nodes()],
            mode='markers+text',
            hoverinfo='text',
            marker=dict(
                showscale=True,
                colorscale='RdBu',
                size=[20 + centrality[node] * 50 for node in G.nodes()],
                color=[centrality[node] for node in G.nodes()],
                colorbar=dict(
                    thickness=15,
                    title='Node Connections',
                    xanchor='left',
                    titleside='right'
                )
            ),
            text=[node for node in G.nodes()],
            textposition="top center"
        )

        fig = go.Figure(data=edge_trace + [node_trace],
                        layout=go.Layout(
                            title='Email Communication Network',
                            showlegend=False,
                            hovermode='closest',
                            margin=dict(b=0, l=0, r=0, t=40),
                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                        ))

        return fig

    def create_wordcloud(self):
        """Create word cloud from email content"""
        # Combine all text
        all_text = " ".join([
            f"{email.get('subject', '')} {email.get('summary', '')}"
            for email in self.emails
        ])

        # Remove common words and email-specific terms
        stopwords = set(['enron', 'email', 'com', 'sent', 'subject', 'date', 'will', 'would'])

        # Generate word cloud
        wordcloud = WordCloud(width=800, height=400,
                              background_color='white',
                              stopwords=stopwords,
                              colormap='viridis').generate(all_text)

        return wordcloud

    def analyze_sentiment_timeline(self):
        """Analyze sentiment over time"""
        sentiment_df = self.df[self.df['date'].notna()].copy()

        # Map tone to sentiment score
        tone_map = {
            'Collaborative/Professional': 1,
            'Neutral/Informational': 0,
            'Concerned/Critical': -1
        }

        sentiment_df['sentiment'] = sentiment_df['tone_analysis'].map(
            lambda x: tone_map.get(x, 0)
        )

        # Group by month
        sentiment_df['month'] = sentiment_df['date'].dt.to_period('M')
        monthly_sentiment = sentiment_df.groupby('month')['sentiment'].agg(['mean', 'count'])
        monthly_sentiment.index = monthly_sentiment.index.to_timestamp()

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=monthly_sentiment.index,
            y=monthly_sentiment['mean'],
            mode='lines+markers',
            name='Average Sentiment',
            line=dict(color='blue', width=2)
        ))

        fig.add_trace(go.Bar(
            x=monthly_sentiment.index,
            y=monthly_sentiment['count'],
            name='Email Count',
            yaxis='y2',
            opacity=0.3
        ))

        fig.update_layout(
            title='Email Sentiment Over Time',
            xaxis_title='Date',
            yaxis=dict(title='Average Sentiment', side='left'),
            yaxis2=dict(title='Email Count', overlaying='y', side='right'),
            hovermode='x unified'
        )

        return fig

    def extract_key_entities(self):
        """Extract and visualize key entities"""
        all_people = []
        all_orgs = []
        all_projects = []

        for email in self.emails:
            entities = email.get('entities', {})
            all_people.extend(entities.get('people', []))
            all_orgs.extend(entities.get('organizations', []))
            all_projects.extend(entities.get('projects', []))

        # Count frequencies
        people_counts = Counter(all_people).most_common(15)
        org_counts = Counter(all_orgs).most_common(15)
        project_counts = Counter(all_projects).most_common(10)

        return people_counts, org_counts, project_counts


def find_json_file():
    """Automatically find the JSON file in subfolders"""
    # Look for common locations
    possible_paths = [
        'enron_emails.json',
        'data/enron_emails.json',
        'emails/enron_emails.json',
        'json/enron_emails.json',
        'input/enron_emails.json'
    ]

    # Also search in subdirectories
    for root, dirs, files in os.walk('.'):
        for file in files:
            if file == 'enron_emails.json':
                return os.path.join(root, file)

    # Check predefined paths
    for path in possible_paths:
        if os.path.exists(path):
            return path

    return None


def load_email_data(file_path):
    """Load email data from JSON file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Handle both formats: list of emails or dict with 'emails' key
    if isinstance(data, list):
        return data
    elif isinstance(data, dict) and 'emails' in data:
        return data['emails']
    else:
        raise ValueError("Invalid JSON format")


def main():
    st.set_page_config(page_title="Enron Email Analysis", layout="wide")

    st.title("🏢 Enron Email Analysis & Story Generator")
    st.markdown("### Analyzing the collapse of Enron through email communications")

    # Find JSON file automatically
    json_path = PROCESSED_JSON_OUTPUT_100

    if json_path is None:
        st.error("""
        ❌ Could not find 'enron_emails.json' file!

        Please ensure the file is placed in one of these locations:
        - Current directory
        - ./data/
        - ./emails/
        - ./json/
        - ./input/
        """)
        return

    st.sidebar.success(f"✅ Found JSON file at: {json_path}")

    # Load email data
    try:
        email_data = load_email_data(json_path)
        st.sidebar.info(f"📧 Loaded {len(email_data)} emails")
    except Exception as e:
        st.error(f"Error loading JSON file: {str(e)}")
        return

    analyzer = EnronEmailAnalyzer(email_data, json_path)

    # Check if story already exists
    existing_story = analyzer.load_existing_story()

    if existing_story:
        st.sidebar.success("✅ Story already generated!")
        st.sidebar.write(f"Generated on: {existing_story['generation_timestamp']}")

    # Sidebar options
    st.sidebar.header("Analysis Options")
    analysis_type = st.sidebar.selectbox(
        "Choose Analysis",
        ["Generated Story", "Overview", "Timeline Analysis", "Network Analysis", "Entity Analysis", "Regenerate Story"]
    )

    if analysis_type == "Generated Story":
        if existing_story:
            st.header("📖 The Enron Story")
            st.info(f"Story generated on: {existing_story['generation_timestamp']}")
            st.markdown("---")
            st.markdown(existing_story['generated_story'])

            # Show some stats
            with st.expander("📊 Generation Statistics"):
                st.write(f"- Emails analyzed: {existing_story['metadata']['total_emails_analyzed']}")
                st.write(f"- Model used: {existing_story['metadata']['model_used']}")
                st.write(f"- Version: {existing_story['metadata']['version']}")

            # Download button
            st.download_button(
                label="Download Story",
                data=existing_story['generated_story'],
                file_name="enron_story.txt",
                mime="text/plain"
            )
        else:
            st.warning("No story generated yet. Click 'Regenerate Story' to generate one.")

    elif analysis_type == "Overview":
        st.header("📊 Dataset Overview")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Emails", len(email_data))
        with col2:
            st.metric("Date Range",
                      f"{analyzer.df['date'].min().strftime('%Y-%m-%d') if not analyzer.df['date'].isna().all() else 'N/A'} to "
                      f"{analyzer.df['date'].max().strftime('%Y-%m-%d') if not analyzer.df['date'].isna().all() else 'N/A'}")
        with col3:
            st.metric("Unique Senders", analyzer.df['from'].nunique())
        with col4:
            st.metric("Classifications", analyzer.df['classification'].nunique())

        # Display word cloud
        st.subheader("📝 Word Cloud")
        wordcloud = analyzer.create_wordcloud()

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)

        # Classification distribution
        st.subheader("📧 Email Classifications")
        class_counts = analyzer.df['classification'].value_counts()
        fig = px.pie(values=class_counts.values, names=class_counts.index,
                     title="Distribution of Email Types")
        st.plotly_chart(fig)

    elif analysis_type == "Timeline Analysis":
        st.header("📅 Timeline Analysis")

        # Timeline visualization
        timeline_fig = analyzer.create_timeline_visualization()
        st.plotly_chart(timeline_fig, use_container_width=True)

        # Sentiment timeline
        st.subheader("😊😐😟 Sentiment Over Time")
        sentiment_fig = analyzer.analyze_sentiment_timeline()
        st.plotly_chart(sentiment_fig, use_container_width=True)

        # Key events
        st.subheader("🔑 Key Events (AI-Identified)")
        with st.spinner("Analyzing emails to identify key events..."):
            key_events = analyzer.extract_key_events()
            if key_events:
                key_events_df = pd.DataFrame(key_events)
                st.table(key_events_df)
                st.caption("These events were automatically identified by AI analysis of the email patterns.")
            else:
                st.info("No key events could be extracted. Try regenerating the story.")

    elif analysis_type == "Network Analysis":
        st.header("🕸️ Communication Network")

        network_fig = analyzer.create_network_graph()
        st.plotly_chart(network_fig, use_container_width=True)

        st.info(
            "Node size represents the number of connections. Larger nodes are more central to the communication network.")

    elif analysis_type == "Entity Analysis":
        st.header("👥 Key Entities")

        people, orgs, projects = analyzer.extract_key_entities()

        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("Top People")
            people_df = pd.DataFrame(people, columns=['Person', 'Mentions'])
            st.dataframe(people_df)

        with col2:
            st.subheader("Top Organizations")
            orgs_df = pd.DataFrame(orgs, columns=['Organization', 'Mentions'])
            st.dataframe(orgs_df)

        with col3:
            st.subheader("Key Projects")
            projects_df = pd.DataFrame(projects, columns=['Project', 'Mentions'])
            st.dataframe(projects_df)

    elif analysis_type == "Regenerate Story":
        st.header("🔄 Regenerate Story")

        if existing_story:
            st.warning("⚠️ A story already exists. Regenerating will overwrite it.")

        if st.button("Generate New Story", type="primary"):
            with st.spinner("Analyzing emails and generating story... This may take a few minutes."):
                # Analyze emails
                progress_bar = st.progress(0)
                email_analyses = []

                # Analyze a subset of emails to save time
                sample_size = min(20, len(email_data))

                # Get most important emails based on date and classification
                important_emails = []

                # Add critical emails from November-December 2001
                critical_period = analyzer.df[
                    (analyzer.df['date'] >= '2001-11-01') &
                    (analyzer.df['date'] <= '2001-12-31')
                    ]
                if not critical_period.empty:
                    important_emails.extend(critical_period.to_dict('records'))

                # Add remaining emails by importance
                if len(important_emails) < sample_size:
                    remaining = analyzer.df[~analyzer.df.index.isin(critical_period.index)]
                    if not remaining.empty:
                        remaining_sorted = remaining.sort_values(
                            'date', ascending=False
                        ).head(sample_size - len(important_emails))
                        important_emails.extend(remaining_sorted.to_dict('records'))

                sample_emails = important_emails[:sample_size]

                for i, email in enumerate(sample_emails):
                    analysis = analyzer.analyze_individual_email(email)
                    email_analyses.append((email, analysis))
                    progress_bar.progress((i + 1) / sample_size)

                # Generate narrative
                narrative = analyzer.generate_narrative(email_analyses)

                # Save to JSON
                story_data = analyzer.save_story_to_json(narrative, email_analyses)

                st.success("✅ Story generated and saved successfully!")
                st.balloons()

                # Display the story
                st.markdown("---")
                st.markdown(narrative)

                # Download button
                st.download_button(
                    label="Download Story",
                    data=narrative,
                    file_name="enron_story.txt",
                    mime="text/plain"
                )


if __name__ == "__main__":
    main()