import streamlit as st
import json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from datetime import datetime
import io
import base64
from collections import Counter
import warnings
import asyncio
import time
from typing import List, Dict, Any
import requests
import json as json_lib
from concurrent.futures import ThreadPoolExecutor
import threading

# Langchain imports for Ollama
from langchain.llms import Ollama
from langchain.schema import HumanMessage, SystemMessage
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.callbacks import StreamlitCallbackHandler

warnings.filterwarnings('ignore')

# Import the config
from src.config.config import PROCESSED_JSON_OUTPUT

# Set page configuration
st.set_page_config(
    page_title="Ollama-Enhanced Kohonen SOM Analyzer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)


class OllamaLLMAnalyzer:
    """Langchain-based email cluster analyzer using Ollama with Mistral model"""

    def __init__(self, model_name="mistral", base_url="http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url

        # Initialize Langchain with Ollama
        try:
            self.llm = Ollama(
                model=model_name,
                base_url=base_url,
                temperature=0.3,
                num_predict=2000
            )
            self._setup_chains()
            self.is_available = self._test_connection()
        except Exception as e:
            st.error(f"Failed to initialize Ollama LLM: {e}")
            self.llm = None
            self.is_available = False

    def _test_connection(self):
        """Test if Ollama is running and model is available"""
        try:
            # Test basic connection
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [model['name'] for model in models]

                # Check if our model is available
                if any(self.model_name in name for name in model_names):
                    return True
                else:
                    st.warning(f"Model '{self.model_name}' not found. Available models: {model_names}")
                    return False
            return False
        except Exception as e:
            st.error(f"Cannot connect to Ollama at {self.base_url}: {e}")
            return False

    def _setup_chains(self):
        """Setup Langchain chains for different analysis tasks"""

        # Cluster analysis chain
        cluster_analysis_template = """You are an expert business analyst specializing in email communication analysis and organizational insights.

Analyze this email cluster from a Kohonen Self-Organizing Map:

CLUSTER INFORMATION:
- Cluster ID: {cluster_id}
- Total emails in cluster: {total_count}
- Sample emails analyzed: {sample_count}

SAMPLE EMAILS:
{emails_text}

Please provide a comprehensive analysis with the following sections:

1. **CLUSTER THEME**: What is the main theme/topic that unites these emails?

2. **COMMUNICATION PATTERNS**:
   - Primary communication type (internal/external, formal/informal)
   - Common tone and style
   - Typical participants or roles involved

3. **BUSINESS CONTEXT**:
   - What business processes or activities do these emails represent?
   - Key projects, departments, or initiatives mentioned
   - Priority level and urgency patterns

4. **KEY INSIGHTS**:
   - Most important findings about this cluster
   - Patterns or anomalies identified
   - Potential action items or business implications

5. **CLUSTER SUMMARY**: Provide a 2-3 sentence executive summary.

Format your response clearly with markdown headers and bullet points."""

        self.cluster_prompt = PromptTemplate(
            template=cluster_analysis_template,
            input_variables=["cluster_id", "total_count", "sample_count", "emails_text"]
        )
        self.cluster_chain = LLMChain(llm=self.llm, prompt=self.cluster_prompt, verbose=False)

        # Overview analysis chain
        overview_template = """You are a senior business intelligence analyst. Provide an executive summary of this Kohonen Self-Organizing Map analysis.

ANALYSIS DATA:
- Total Emails Analyzed: {total_emails:,}
- Active Clusters: {total_clusters}
- SOM Grid Size: {som_width}x{som_height}
- SOM Utilization: {utilization:.1f}%
- Average Cluster Size: {avg_cluster_size:.1f}

TOP CLUSTER INSIGHTS:
{cluster_summaries}

CLUSTER SIZE DISTRIBUTION:
{size_distribution}

Please provide:

1. **COMMUNICATION LANDSCAPE OVERVIEW**
   - Overall patterns in organizational communication
   - Diversity of communication types and topics

2. **DOMINANT THEMES AND PATTERNS**
   - Most prominent communication themes
   - Recurring patterns across clusters

3. **ORGANIZATIONAL INSIGHTS**
   - What this reveals about business processes
   - Communication efficiency and effectiveness
   - Cross-functional collaboration patterns

4. **STRATEGIC RECOMMENDATIONS**
   - Actionable insights for management
   - Process improvement suggestions
   - Communication strategy recommendations

5. **EXECUTIVE SUMMARY**
   - 3-4 sentence high-level summary for decision makers

Focus on actionable business insights and strategic implications."""

        self.overview_prompt = PromptTemplate(
            template=overview_template,
            input_variables=["total_emails", "total_clusters", "som_width", "som_height",
                             "utilization", "avg_cluster_size", "cluster_summaries", "size_distribution"]
        )
        self.overview_chain = LLMChain(llm=self.llm, prompt=self.overview_prompt, verbose=False)

    def analyze_cluster_batch(self, cluster_emails: List[Dict], cluster_id: str, max_samples: int = 15) -> Dict:
        """Analyze a cluster of emails using Ollama Mistral"""

        if not self.is_available:
            return self._generate_demo_analysis(cluster_id, len(cluster_emails))

        # Sample emails if cluster is too large
        if len(cluster_emails) > max_samples:
            sample_emails = np.random.choice(cluster_emails, max_samples, replace=False).tolist()
        else:
            sample_emails = cluster_emails

        # Prepare email data for analysis
        emails_text = self._format_emails_for_analysis(sample_emails)

        try:
            # Use Langchain chain for analysis
            response = self.cluster_chain.run(
                cluster_id=cluster_id,
                total_count=len(cluster_emails),
                sample_count=len(sample_emails),
                emails_text=emails_text
            )

            return {
                'cluster_id': cluster_id,
                'total_emails': len(cluster_emails),
                'sampled_emails': len(sample_emails),
                'analysis': response,
                'sample_subjects': [email.get('subject', '')[:50] for email in sample_emails[:5]],
                'success': True
            }
        except Exception as e:
            st.warning(f"Error analyzing cluster {cluster_id}: {str(e)}")
            return {
                'cluster_id': cluster_id,
                'total_emails': len(cluster_emails),
                'sampled_emails': len(sample_emails),
                'analysis': f"Analysis failed: {str(e)}",
                'sample_subjects': [email.get('subject', '')[:50] for email in sample_emails[:5]],
                'success': False
            }

    def _format_emails_for_analysis(self, emails: List[Dict]) -> str:
        """Format emails for LLM analysis"""
        formatted_emails = []

        for i, email in enumerate(emails):
            email_text = f"""
Email {i + 1}:
  Subject: {email.get('subject', 'No Subject')[:100]}
  Summary: {email.get('summary', 'No Summary')[:400]}
  Classification: {email.get('classification', 'Unknown')}
  Tone: {email.get('tone_analysis', 'Unknown')}
  Date: {email.get('date', 'Unknown')}
  Recipients: {len(email.get('to', '').split(',')) if email.get('to') else 0}
  Key Entities: {self._format_entities(email.get('entities', {}))}
            """
            formatted_emails.append(email_text.strip())

        return "\n\n".join(formatted_emails)

    def _format_entities(self, entities: Dict) -> str:
        """Format entities for display"""
        if not entities:
            return "None"

        formatted = []
        for entity_type, entity_list in entities.items():
            if entity_list:
                formatted.append(f"{entity_type}: {', '.join(entity_list[:3])}")

        return "; ".join(formatted) if formatted else "None"

    def analyze_som_overview(self, cluster_analyses: List[Dict], som_stats: Dict, som_width: int,
                             som_height: int) -> str:
        """Generate overall SOM analysis summary using Ollama"""

        if not self.is_available:
            return self._generate_demo_overview()

        # Prepare cluster summaries
        successful_analyses = [a for a in cluster_analyses if a.get('success', True)]
        cluster_summaries = []

        for analysis in successful_analyses[:15]:  # Top 15 clusters
            summary_line = f"- Cluster {analysis['cluster_id']}: {analysis['total_emails']} emails"

            # Extract theme from analysis if available
            analysis_text = analysis.get('analysis', '')
            if 'CLUSTER THEME' in analysis_text:
                try:
                    theme_start = analysis_text.find('CLUSTER THEME')
                    theme_end = analysis_text.find('\n\n', theme_start)
                    if theme_end == -1:
                        theme_end = analysis_text.find('\n2.', theme_start)
                    theme = analysis_text[theme_start:theme_end].replace('**CLUSTER THEME**:', '').strip()[:100]
                    if theme:
                        summary_line += f" - {theme}"
                except:
                    pass

            cluster_summaries.append(summary_line)

        # Create size distribution summary
        cluster_sizes = [a['total_emails'] for a in cluster_analyses]
        size_distribution = f"""
- Largest cluster: {max(cluster_sizes)} emails
- Smallest cluster: {min(cluster_sizes)} emails
- Clusters with >100 emails: {len([s for s in cluster_sizes if s > 100])}
- Clusters with >50 emails: {len([s for s in cluster_sizes if s > 50])}
- Single-email clusters: {len([s for s in cluster_sizes if s == 1])}
        """

        try:
            response = self.overview_chain.run(
                total_emails=som_stats.get('total_emails', 0),
                total_clusters=som_stats.get('total_clusters', 0),
                som_width=som_width,
                som_height=som_height,
                utilization=som_stats.get('utilization', 0),
                avg_cluster_size=som_stats.get('avg_cluster_size', 0),
                cluster_summaries='\n'.join(cluster_summaries),
                size_distribution=size_distribution
            )
            return response
        except Exception as e:
            st.error(f"Error generating overview analysis: {str(e)}")
            return f"Overview analysis failed: {str(e)}"

    def _generate_demo_analysis(self, cluster_id: str, email_count: int) -> Dict:
        """Generate demo analysis when Ollama is not available"""

        themes = [
            "Project Management & Coordination",
            "Customer Support & Service Issues",
            "Internal Team Communications",
            "Vendor & Supplier Relations",
            "Executive Decision Making",
            "Marketing & Sales Activities",
            "Technical Support & Troubleshooting",
            "Financial & Administrative Tasks",
            "HR & Personnel Management",
            "Product Development Updates"
        ]

        theme = np.random.choice(themes)

        analysis = f"""
## **CLUSTER THEME**
{theme}

## **COMMUNICATION PATTERNS**
- **Type**: Mixed internal and external communications
- **Tone**: Professional and task-oriented
- **Participants**: Cross-functional teams and stakeholders

## **BUSINESS CONTEXT**
- **Process**: {theme.lower()} workflows and coordination
- **Priority**: Medium to high business impact
- **Scope**: Involves multiple departments and stakeholders

## **KEY INSIGHTS**
- Strong evidence of structured business processes
- Clear action-oriented communication patterns
- Good cross-functional collaboration
- Opportunities for process optimization

## **CLUSTER SUMMARY**
This cluster represents {theme.lower()} with {email_count} emails showing structured business communication patterns. The emails demonstrate active collaboration and clear business processes with opportunities for enhanced efficiency.

*Note: This is a demonstration analysis. Install and run Ollama with Mistral for AI-powered insights.*
        """

        return {
            'cluster_id': cluster_id,
            'total_emails': email_count,
            'sampled_emails': min(email_count, 15),
            'analysis': analysis,
            'sample_subjects': [f"Sample email subject {i + 1}..." for i in range(min(5, email_count))],
            'success': True
        }

    def _generate_demo_overview(self) -> str:
        """Generate demo overview analysis"""
        return """
## **COMMUNICATION LANDSCAPE OVERVIEW**
The email dataset reveals a diverse organizational communication ecosystem with well-structured business processes and active cross-functional collaboration.

## **DOMINANT THEMES AND PATTERNS** 
- Project management and coordination activities dominate
- Strong customer service and support communications
- Regular internal team coordination and updates
- Active vendor and supplier relationship management

## **ORGANIZATIONAL INSIGHTS**
- Evidence of mature business processes across departments
- Good communication flow between internal and external stakeholders
- Clear action-oriented communication culture
- Strong collaboration patterns across functional areas

## **STRATEGIC RECOMMENDATIONS**
- Implement communication templates for recurring processes
- Consider automation for routine coordination tasks
- Enhance cross-functional collaboration tools
- Establish communication efficiency metrics

## **EXECUTIVE SUMMARY**
Analysis of the email dataset reveals a well-functioning organizational communication system with diverse business processes and strong collaboration patterns. The clustering identifies clear workflow categories and presents opportunities for process optimization and communication efficiency improvements.

*Note: This is a demonstration analysis. Install and run Ollama with Mistral for AI-powered insights.*
        """


class KohonenSOM:
    def __init__(self, width, height, input_dim, learning_rate=0.1, sigma=1.0):
        self.width = width
        self.height = height
        self.input_dim = input_dim
        self.learning_rate = learning_rate
        self.sigma = sigma

        # Initialize weights randomly
        self.weights = np.random.random((height, width, input_dim))

        # Create coordinate arrays
        self.locations = np.array([[i, j] for i in range(height) for j in range(width)])
        self.locations = self.locations.reshape((height, width, 2))

    def _find_best_matching_unit(self, x):
        distances = np.sum((self.weights - x) ** 2, axis=2)
        bmu_idx = np.unravel_index(np.argmin(distances), distances.shape)
        return bmu_idx

    def _update_weights(self, x, bmu_idx, iteration, total_iterations):
        current_learning_rate = self.learning_rate * np.exp(-iteration / total_iterations)
        current_sigma = self.sigma * np.exp(-iteration / total_iterations)

        bmu_location = np.array([bmu_idx[0], bmu_idx[1]])
        distances_to_bmu = np.sum((self.locations - bmu_location) ** 2, axis=2)
        neighborhood = np.exp(-distances_to_bmu / (2 * current_sigma ** 2))

        for i in range(self.height):
            for j in range(self.width):
                self.weights[i, j] += (current_learning_rate * neighborhood[i, j] *
                                       (x - self.weights[i, j]))

    def train(self, data, epochs=1000, progress_callback=None):
        for epoch in range(epochs):
            idx = np.random.randint(0, len(data))
            x = data[idx]

            bmu_idx = self._find_best_matching_unit(x)
            self._update_weights(x, bmu_idx, epoch, epochs)

            if progress_callback and (epoch + 1) % 50 == 0:
                progress_callback(epoch + 1, epochs)

    def map_data(self, data):
        mapped_data = []
        for x in data:
            bmu_idx = self._find_best_matching_unit(x)
            mapped_data.append(bmu_idx)
        return np.array(mapped_data)

    def get_u_matrix(self):
        u_matrix = np.zeros((self.height, self.width))

        for i in range(self.height):
            for j in range(self.width):
                neighbors = []
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        if di == 0 and dj == 0:
                            continue
                        ni, nj = i + di, j + dj
                        if 0 <= ni < self.height and 0 <= nj < self.width:
                            neighbors.append(self.weights[ni, nj])

                if neighbors:
                    distances = [np.linalg.norm(self.weights[i, j] - neighbor)
                                 for neighbor in neighbors]
                    u_matrix[i, j] = np.mean(distances)

        return u_matrix


class EmailFeatureExtractor:
    def __init__(self):
        self.tfidf_vectorizer = TfidfVectorizer(max_features=50, stop_words='english')
        self.label_encoders = {}
        self.scaler = StandardScaler()

    def extract_features(self, emails_data):
        features = []

        for email in emails_data:
            email_features = {}

            # Text-based features
            summary = email.get('summary', '')
            subject = email.get('subject', '')

            email_features['summary_length'] = len(summary)
            email_features['subject_length'] = len(subject)
            email_features['word_count'] = len(summary.split())

            # Recipient features
            to_field = email.get('to', '')
            email_features['recipient_count'] = len(to_field.split(',')) if to_field else 0

            # Date features
            date_str = email.get('date', '')
            email_features.update(self._extract_date_features(date_str))

            # Classification and tone features
            email_features['classification'] = email.get('classification', 'Unknown')
            email_features['tone_analysis'] = email.get('tone_analysis', 'Unknown')

            # Entity features
            entities = email.get('entities', {})
            email_features['people_count'] = len(entities.get('people', []))
            email_features['organizations_count'] = len(entities.get('organizations', []))
            email_features['locations_count'] = len(entities.get('locations', []))
            email_features['projects_count'] = len(entities.get('projects', []))

            features.append(email_features)

        df = pd.DataFrame(features)

        # TF-IDF features
        text_content = [email.get('summary', '') + ' ' + email.get('subject', '')
                        for email in emails_data]

        if text_content and any(text_content):
            tfidf_features = self.tfidf_vectorizer.fit_transform(text_content).toarray()
            tfidf_df = pd.DataFrame(tfidf_features,
                                    columns=[f'tfidf_{i}' for i in range(tfidf_features.shape[1])])
            df = pd.concat([df, tfidf_df], axis=1)

        # Encode categorical features
        categorical_columns = ['classification', 'tone_analysis']
        for col in categorical_columns:
            if col in df.columns:
                le = LabelEncoder()
                df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le

        # Select numerical features
        numerical_features = df.select_dtypes(include=[np.number]).fillna(0)

        # Normalize features
        normalized_features = self.scaler.fit_transform(numerical_features)

        return normalized_features, df

    def _extract_date_features(self, date_str):
        features = {'year': 2000, 'month': 1, 'day': 1, 'hour': 0, 'minute': 0}

        try:
            if '.' in date_str and ' ' in date_str:
                date_part, time_part = date_str.split(' ')
                day, month, year = map(int, date_part.split('.'))
                hour, minute, _ = map(int, time_part.split(':'))
                features.update({'year': year, 'month': month, 'day': day, 'hour': hour, 'minute': minute})
        except:
            pass

        return features


def load_json_data_from_config():
    """Load JSON data from the configured path"""
    try:
        with open(PROCESSED_JSON_OUTPUT, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if isinstance(data, list):
            emails = data
        elif isinstance(data, dict) and 'emails' in data:
            emails = data['emails']
        elif isinstance(data, dict) and 'data' in data:
            emails = data['data']
        else:
            emails = [data] if not isinstance(list(data.values())[0], list) else list(data.values())[0]

        return emails
    except FileNotFoundError:
        st.error(f"❌ JSON file not found at: {PROCESSED_JSON_OUTPUT}")
        st.info("Please ensure the JSON file exists at the configured path.")
        return None
    except json.JSONDecodeError as e:
        st.error(f"❌ Error parsing JSON file: {str(e)}")
        return None
    except Exception as e:
        st.error(f"❌ Error loading JSON: {str(e)}")
        return None


def create_som_visualizations(som, mapped_data, original_data, features_df):
    """Create interactive Plotly visualizations"""

    # 1. U-Matrix
    u_matrix = som.get_u_matrix()

    fig_u_matrix = go.Figure(data=go.Heatmap(
        z=u_matrix,
        colorscale='Viridis',
        showscale=True,
        colorbar=dict(title="Distance")
    ))
    fig_u_matrix.update_layout(
        title="U-Matrix (Unified Distance Matrix)",
        xaxis_title="SOM Width",
        yaxis_title="SOM Height",
        height=400
    )

    # 2. Data Points Distribution
    scatter_data = pd.DataFrame({
        'x': mapped_data[:, 1],
        'y': mapped_data[:, 0],
        'email_id': range(len(mapped_data)),
        'subject': [email.get('subject', 'N/A')[:50] + '...' for email in original_data],
        'classification': [email.get('classification', 'Unknown') for email in original_data]
    })

    fig_scatter = px.scatter(
        scatter_data,
        x='x', y='y',
        color='classification',
        hover_data=['email_id', 'subject'],
        title="Email Distribution on SOM Grid",
        labels={'x': 'SOM Width', 'y': 'SOM Height'}
    )
    fig_scatter.update_layout(height=400)

    # 3. Density Heatmap
    density_map = np.zeros((som.height, som.width))
    for point in mapped_data:
        density_map[point[0], point[1]] += 1

    fig_density = go.Figure(data=go.Heatmap(
        z=density_map,
        colorscale='Reds',
        showscale=True,
        colorbar=dict(title="Email Count")
    ))
    fig_density.update_layout(
        title="Email Density Map",
        xaxis_title="SOM Width",
        yaxis_title="SOM Height",
        height=400
    )

    # 4. Feature Importance (average weights per dimension)
    feature_importance = np.mean(np.abs(som.weights), axis=(0, 1))

    fig_features = go.Figure(data=go.Bar(
        x=list(range(len(feature_importance))),
        y=feature_importance,
        text=[f"F{i}" for i in range(len(feature_importance))],
        textposition='auto'
    ))
    fig_features.update_layout(
        title="Feature Importance (Average Absolute Weights)",
        xaxis_title="Feature Index",
        yaxis_title="Average Weight",
        height=400
    )

    return fig_u_matrix, fig_scatter, fig_density, fig_features


def organize_emails_by_clusters(mapped_data, original_data):
    """Organize emails by their SOM clusters"""
    clusters = {}

    for idx, point in enumerate(mapped_data):
        cluster_key = f"({point[0]}, {point[1]})"

        if cluster_key not in clusters:
            clusters[cluster_key] = []

        clusters[cluster_key].append(original_data[idx])

    # Sort clusters by size (largest first)
    sorted_clusters = dict(sorted(clusters.items(), key=lambda x: len(x[1]), reverse=True))

    return sorted_clusters


def run_ollama_cluster_analysis(clusters, llm_analyzer, max_clusters=25):
    """Run Ollama Mistral analysis on clusters with enhanced progress tracking"""

    # Limit to top N clusters for analysis
    cluster_items = list(clusters.items())[:max_clusters]

    progress_bar = st.progress(0)
    status_text = st.empty()
    analysis_info = st.empty()

    cluster_analyses = []

    st.info(f"🤖 Analyzing {len(cluster_items)} largest clusters with Ollama Mistral via Langchain...")

    for i, (cluster_id, emails) in enumerate(cluster_items):
        status_text.text(f"🔍 Analyzing cluster {i + 1}/{len(cluster_items)}: {cluster_id}")
        analysis_info.text(f"📊 Cluster contains {len(emails)} emails • Processing with local Mistral...")

        # Use Ollama analyzer
        analysis = llm_analyzer.analyze_cluster_batch(emails, cluster_id, max_samples=20)
        cluster_analyses.append(analysis)

        # Update progress
        progress_bar.progress((i + 1) / len(cluster_items))

        # Show success/failure
        if analysis.get('success', True):
            analysis_info.text(f"✅ Successfully analyzed cluster {cluster_id}")
        else:
            analysis_info.text(f"❌ Error analyzing cluster {cluster_id}")

        # Small delay for UI updates (no rate limiting needed for local Ollama)
        time.sleep(0.1)

    status_text.text("✅ Ollama Mistral cluster analysis completed!")
    analysis_info.text(f"🎉 Processed {len(cluster_analyses)} clusters with AI insights")
    progress_bar.progress(1.0)

    return cluster_analyses


def main():
    st.title("🧠🦙 Ollama-Enhanced Kohonen SOM Analyzer")
    st.markdown("### Local AI-Powered Email Clustering Analysis for Large Datasets")
    st.info(f"📁 Loading data from: `{PROCESSED_JSON_OUTPUT}`")

    # Sidebar for configuration
    st.sidebar.header("🔧 Configuration")

    # Ollama Configuration
    st.sidebar.subheader("🦙 Ollama Configuration")

    st.sidebar.info("💡 Using Ollama with Mistral for local AI analysis - no API costs!")

    ollama_url = st.sidebar.text_input(
        "Ollama URL",
        value="http://localhost:11434",
        help="URL where Ollama is running"
    )

    ollama_model = st.sidebar.selectbox(
        "Ollama Model",
        ["mistral", "mistral:7b", "mistral:latest", "llama2", "codellama"],
        help="Choose the Ollama model for analysis"
    )

    # Test Ollama connection
    test_connection = st.sidebar.button("🔍 Test Ollama Connection")

    if test_connection:
        try:
            response = requests.get(f"{ollama_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [model['name'] for model in models]
                st.sidebar.success(f"✅ Connected! Available models: {len(model_names)}")

                if any(ollama_model in name for name in model_names):
                    st.sidebar.success(f"✅ Model '{ollama_model}' is available")
                else:
                    st.sidebar.warning(f"⚠️ Model '{ollama_model}' not found")
                    st.sidebar.info(f"Available: {', '.join(model_names[:3])}...")
            else:
                st.sidebar.error("❌ Failed to connect to Ollama")
        except Exception as e:
            st.sidebar.error(f"❌ Connection error: {str(e)}")
            st.sidebar.info("Make sure Ollama is running: `ollama serve`")

    # Initialize Ollama Analyzer
    llm_analyzer = OllamaLLMAnalyzer(ollama_model, ollama_url)

    if llm_analyzer.is_available:
        st.sidebar.success("✅ Ollama connected via Langchain")
    else:
        st.sidebar.warning("⚠️ Ollama not available - using demo mode")

    # Load data automatically
    if st.sidebar.button("🔄 Load All Data", type="primary"):
        with st.spinner("Loading complete dataset..."):
            emails_data = load_json_data_from_config()

        if emails_data is not None:
            st.session_state.emails_data = emails_data
            st.success(f"✅ Loaded {len(emails_data):,} emails from dataset")

            # Show dataset statistics
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("📧 Total Emails", f"{len(emails_data):,}")

            with col2:
                classifications = [email.get('classification', 'Unknown') for email in emails_data]
                unique_classifications = len(set(classifications))
                st.metric("🏷️ Classifications", unique_classifications)

            with col3:
                avg_length = np.mean([len(email.get('summary', '')) for email in emails_data])
                st.metric("📏 Avg Summary", f"{avg_length:.0f} chars")

            with col4:
                date_range = len(set([email.get('date', '')[:7] for email in emails_data if email.get('date')]))
                st.metric("📅 Time Span", f"{date_range} months")

    # Check if data is loaded
    if hasattr(st.session_state, 'emails_data'):
        emails_data = st.session_state.emails_data

        # SOM Configuration for large datasets
        st.sidebar.subheader("🗺️ SOM Parameters")
        som_width = st.sidebar.slider("SOM Width", 8, 25, 15, help="Larger grid for more detailed clustering")
        som_height = st.sidebar.slider("SOM Height", 8, 20, 12, help="Larger grid for more detailed clustering")
        epochs = st.sidebar.slider("Training Epochs", 500, 3000, 1000, help="More epochs for better convergence")
        learning_rate = st.sidebar.slider("Learning Rate", 0.01, 0.3, 0.08, help="Lower rate for large datasets")

        # Processing options
        st.sidebar.subheader("⚙️ Processing Options")
        max_emails_som = st.sidebar.slider(
            "Max Emails for SOM Training",
            1000, min(10000, len(emails_data)),
            min(5000, len(emails_data)),
            help="Balance between accuracy and processing time"
        )

        max_clusters_llm = st.sidebar.slider(
            "Max Clusters for LLM Analysis",
            10, 50, 25,
            help="Number of top clusters to analyze with Ollama"
        )

        if st.sidebar.button("🚀 Run Complete Analysis", type="primary"):
            # Step 1: Feature extraction
            st.subheader("📊 Step 1: Feature Extraction")
            with st.spinner("Extracting features from all emails..."):
                feature_extractor = EmailFeatureExtractor()
                all_features, all_features_df = feature_extractor.extract_features(emails_data)

            st.success(f"✅ Extracted {all_features.shape[1]} features from {len(emails_data):,} emails")

            # Step 2: SOM Training (on subset for performance)
            st.subheader("🧠 Step 2: SOM Training")

            # Use subset for SOM training but analyze all data
            training_data = emails_data[:max_emails_som]
            training_features = all_features[:max_emails_som]

            st.info(f"Training SOM on {len(training_data):,} emails (subset for performance)")

            progress_bar = st.progress(0)
            status_text = st.empty()

            def update_progress(current_epoch, total_epochs):
                progress = current_epoch / total_epochs
                progress_bar.progress(progress)
                status_text.text(f"Training SOM: {current_epoch}/{total_epochs} epochs ({progress:.1%})")

            som = KohonenSOM(som_width, som_height, all_features.shape[1],
                             learning_rate=learning_rate, sigma=max(som_width, som_height) / 2)

            som.train(training_features, epochs=epochs, progress_callback=update_progress)

            status_text.text("✅ SOM training completed!")

            # Step 3: Map all emails to SOM
            st.subheader("🗺️ Step 3: Mapping All Emails")
            with st.spinner("Mapping all emails to SOM clusters..."):
                all_mapped_data = som.map_data(all_features)

            st.success(f"✅ Mapped {len(emails_data):,} emails to SOM clusters")

            # Step 4: Organize emails by clusters
            st.subheader("📋 Step 4: Cluster Organization")
            with st.spinner("Organizing emails by clusters..."):
                clusters = organize_emails_by_clusters(all_mapped_data, emails_data)

            st.success(f"✅ Organized emails into {len(clusters)} active clusters")

            # Display cluster statistics
            cluster_sizes = [len(emails) for emails in clusters.values()]

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("🏘️ Active Clusters", len(clusters))
            with col2:
                st.metric("📊 Largest Cluster", max(cluster_sizes))
            with col3:
                st.metric("📈 Average Size", f"{np.mean(cluster_sizes):.1f}")
            with col4:
                st.metric("🎯 Utilization", f"{len(clusters) / (som_width * som_height) * 100:.1f}%")

            # Step 5: Ollama LLM Analysis
            st.subheader("🦙 Step 5: Ollama Mistral Analysis")

            st.info("🧠 Using local Ollama with Mistral for intelligent cluster analysis...")
            cluster_analyses = run_ollama_cluster_analysis(clusters, llm_analyzer, max_clusters_llm)

            # Step 6: Generate overall analysis
            st.subheader("📋 Step 6: Comprehensive Analysis Report")
            with st.spinner("Generating AI-powered insights..."):
                som_stats = {
                    'total_emails': len(emails_data),
                    'total_clusters': len(clusters),
                    'utilization': len(clusters) / (som_width * som_height) * 100,
                    'avg_cluster_size': np.mean(cluster_sizes)
                }

                overall_analysis = llm_analyzer.analyze_som_overview(
                    cluster_analyses, som_stats, som_width, som_height
                )

            # Store results
            st.session_state.som = som
            st.session_state.all_mapped_data = all_mapped_data
            st.session_state.all_features_df = all_features_df
            st.session_state.clusters = clusters
            st.session_state.cluster_analyses = cluster_analyses
            st.session_state.overall_analysis = overall_analysis
            st.session_state.som_stats = som_stats
            st.session_state.ollama_model = ollama_model

            st.success("🎉 Complete AI-powered analysis finished!")

    # Display results if available
    if hasattr(st.session_state, 'overall_analysis'):
        st.header("🦙 Ollama Mistral Analysis Results")

        # Show analysis quality indicator
        successful_analyses = len([a for a in st.session_state.cluster_analyses if a.get('success', True)])
        total_analyses = len(st.session_state.cluster_analyses)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("🎯 Analysis Success Rate", f"{(successful_analyses / total_analyses) * 100:.1f}%")
        with col2:
            st.metric("🤖 Clusters Analyzed", f"{total_analyses}")
        with col3:
            st.metric("📊 Total Emails", f"{st.session_state.som_stats['total_emails']:,}")
        with col4:
            st.metric("🦙 Model Used", st.session_state.get('ollama_model', 'mistral'))

        # Overall Analysis
        st.subheader("🎯 Executive Summary (Powered by Ollama Mistral)")
        st.markdown(st.session_state.overall_analysis)

        # Add Ollama workflow visualization
        with st.expander("🔍 Local AI Analysis Workflow", expanded=False):
            st.markdown(f"""
            **Ollama + Langchain Processing Pipeline:**

            1. **🔄 Local Processing**: All analysis runs locally - no data leaves your machine
            2. **📊 Data Preprocessing**: Emails formatted and structured for AI analysis
            3. **🧠 Cluster Analysis Chain**: Each cluster analyzed using local Mistral model
            4. **🔍 Pattern Recognition**: AI identifies communication themes and business processes
            5. **💡 Insight Generation**: Strategic recommendations generated from patterns
            6. **📋 Executive Summary**: High-level insights synthesized for decision makers

            **Model Used**: `{st.session_state.get('ollama_model', 'mistral')}` via Ollama + Langchain
            **Benefits**: Private, fast, cost-free analysis with enterprise-grade insights
            **Performance**: Processes {len(st.session_state.cluster_analyses)} clusters in minutes
            """)

        # Detailed cluster analyses
        st.subheader("🔍 Detailed Cluster Analysis")

        # Filter successful analyses for display
        successful_analyses = [a for a in st.session_state.cluster_analyses if a.get('success', True)]
        failed_analyses = [a for a in st.session_state.cluster_analyses if not a.get('success', True)]

        if failed_analyses:
            st.warning(
                f"⚠️ {len(failed_analyses)} cluster(s) had analysis errors. Showing {len(successful_analyses)} successful analyses.")

        # Create tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs(
            ["📊 Top Clusters", "📈 Cluster Insights", "🔬 Analysis Details", "🦙 Ollama Info"])

        with tab1:
            # Display top 10 clusters in a more structured way
            for i, analysis in enumerate(successful_analyses[:10]):
                with st.expander(f"📁 Cluster {analysis['cluster_id']} - {analysis['total_emails']} emails",
                                 expanded=i < 3):

                    # Cluster metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Emails", analysis['total_emails'])
                    with col2:
                        st.metric("Analyzed Sample", analysis['sampled_emails'])
                    with col3:
                        percentage = (analysis['total_emails'] / st.session_state.som_stats['total_emails']) * 100
                        st.metric("% of Dataset", f"{percentage:.1f}%")
                    with col4:
                        rank = i + 1
                        st.metric("Cluster Rank", f"#{rank}")

                    # Sample subjects
                    st.write("**📧 Sample Email Subjects:**")
                    for j, subject in enumerate(analysis['sample_subjects']):
                        st.write(f"{j + 1}. {subject}...")

                    # AI Analysis
                    st.write("**🦙 Ollama Mistral Analysis:**")
                    st.markdown(analysis['analysis'])

        with tab2:
            # Cluster insights visualization
            st.write("**📊 Cluster Size Distribution**")

            cluster_sizes = [a['total_emails'] for a in successful_analyses]
            cluster_ids = [a['cluster_id'] for a in successful_analyses]

            # Create bar chart of cluster sizes
            fig_cluster_sizes = px.bar(
                x=cluster_ids[:15],  # Top 15 clusters
                y=cluster_sizes[:15],
                title="Top 15 Cluster Sizes",
                labels={'x': 'Cluster ID', 'y': 'Number of Emails'},
                color=cluster_sizes[:15],
                color_continuous_scale='viridis'
            )
            fig_cluster_sizes.update_layout(height=400)
            st.plotly_chart(fig_cluster_sizes, use_container_width=True)

            # Summary statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Largest Cluster", f"{max(cluster_sizes)} emails")
            with col2:
                st.metric("Average Cluster", f"{np.mean(cluster_sizes):.1f} emails")
            with col3:
                st.metric("Median Cluster", f"{np.median(cluster_sizes):.0f} emails")

        with tab3:
            # Analysis quality and details
            st.write("**🔬 Analysis Quality Metrics**")

            # Calculate analysis quality metrics
            total_emails_analyzed = sum(a['sampled_emails'] for a in successful_analyses)
            total_emails_in_top_clusters = sum(a['total_emails'] for a in successful_analyses)

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Success Rate",
                          f"{(len(successful_analyses) / len(st.session_state.cluster_analyses)) * 100:.1f}%")
            with col2:
                st.metric("Emails Sampled", f"{total_emails_analyzed:,}")
            with col3:
                st.metric("Coverage",
                          f"{(total_emails_in_top_clusters / st.session_state.som_stats['total_emails']) * 100:.1f}%")
            with col4:
                avg_analysis_length = np.mean([len(a['analysis']) for a in successful_analyses])
                st.metric("Avg Analysis Length", f"{avg_analysis_length:.0f} chars")

            # Show failed analyses if any
            if failed_analyses:
                st.write("**❌ Failed Analyses:**")
                for analysis in failed_analyses:
                    st.error(f"Cluster {analysis['cluster_id']}: {analysis['analysis']}")

        with tab4:
            # Ollama-specific information
            st.write("**🦙 Ollama Performance & Setup**")

            col1, col2 = st.columns(2)

            with col1:
                st.write("**Connection Status:**")
                if llm_analyzer.is_available:
                    st.success("✅ Ollama Connected")
                    st.info(f"📍 URL: {ollama_url}")
                    st.info(f"🤖 Model: {st.session_state.get('ollama_model', 'mistral')}")
                else:
                    st.error("❌ Ollama Not Available")
                    st.info("Run `ollama serve` to start Ollama")

            with col2:
                st.write("**Setup Instructions:**")
                st.code("""
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Start Ollama service
ollama serve

# Pull Mistral model
ollama pull mistral

# Verify installation
ollama list
                """)

            st.write("**🚀 Benefits of Local Analysis:**")
            benefits = [
                "🔒 **Complete Privacy**: Your emails never leave your machine",
                "💰 **Zero API Costs**: No charges for processing thousands of emails",
                "⚡ **High Speed**: Local processing without network delays",
                "🔄 **Unlimited Usage**: Process as many emails as you want",
                "🛡️ **Data Security**: Enterprise-grade privacy and security",
                "🌐 **Offline Capability**: Works without internet connection"
            ]

            for benefit in benefits:
                st.markdown(f"- {benefit}")

        # SOM Visualization Results
        if hasattr(st.session_state, 'som'):
            st.subheader("📈 SOM Visualization Results")

            # Create visualizations using training subset but show full dataset stats
            training_data = st.session_state.emails_data[:max_emails_som]
            training_mapped = st.session_state.som.map_data(
                st.session_state.all_features_df.select_dtypes(include=[np.number]).fillna(0).values[:max_emails_som]
            )

            fig_u_matrix, fig_scatter, fig_density, fig_features = create_som_visualizations(
                st.session_state.som,
                training_mapped,
                training_data,
                st.session_state.all_features_df[:max_emails_som]
            )

            # Display plots in tabs
            viz_tab1, viz_tab2, viz_tab3, viz_tab4 = st.tabs(
                ["🗺️ U-Matrix", "📍 Email Distribution", "🔥 Density Map", "⚖️ Feature Importance"])

            with viz_tab1:
                st.plotly_chart(fig_u_matrix, use_container_width=True)
                st.info("💡 The U-Matrix shows cluster boundaries. Darker areas indicate cluster separations.")

            with viz_tab2:
                st.plotly_chart(fig_scatter, use_container_width=True)
                st.info("💡 Each point represents an email, colored by classification. Hover for details!")

            with viz_tab3:
                st.plotly_chart(fig_density, use_container_width=True)
                st.info("💡 Red areas show high email concentration. These are the main clusters.")

            with viz_tab4:
                st.plotly_chart(fig_features, use_container_width=True)
                st.info("💡 Higher bars indicate more influential features in the clustering.")

        # Enhanced Download Section
        st.subheader("💾 Export Analysis Results")

        col1, col2, col3 = st.columns(3)

        with col1:
            # Create comprehensive results CSV
            results_data = []
            for i, (cluster_id, emails) in enumerate(st.session_state.clusters.items()):
                # Get analysis for this cluster if available
                cluster_analysis = next((a for a in st.session_state.cluster_analyses if a['cluster_id'] == cluster_id),
                                        None)

                for email in emails:
                    results_data.append({
                        'cluster_id': cluster_id,
                        'cluster_size': len(emails),
                        'cluster_rank': i + 1,
                        'analysis_success': cluster_analysis.get('success', False) if cluster_analysis else False,
                        'subject': email.get('subject', ''),
                        'classification': email.get('classification', ''),
                        'tone_analysis': email.get('tone_analysis', ''),
                        'date': email.get('date', ''),
                        'summary': email.get('summary', '')[:200] + '...' if len(
                            email.get('summary', '')) > 200 else email.get('summary', ''),
                        'recipient_count': len(email.get('to', '').split(',')) if email.get('to') else 0
                    })

            results_df = pd.DataFrame(results_data)
            csv_buffer = io.StringIO()
            results_df.to_csv(csv_buffer, index=False)

            st.download_button(
                label="📊 Download Detailed Results",
                data=csv_buffer.getvalue(),
                file_name="ollama_som_analysis_results.csv",
                mime="text/csv",
                help="Complete dataset with cluster assignments and analysis status"
            )

        with col2:
            # Analysis summary report
            report_content = f"""# Ollama Mistral-Enhanced SOM Analysis Report

## Dataset Overview
- **Total Emails**: {st.session_state.som_stats['total_emails']:,}
- **Analysis Date**: {datetime.now().strftime('%Y-%m-%d %H:%M')}
- **SOM Configuration**: {som_width}x{som_height} grid
- **Active Clusters**: {st.session_state.som_stats['total_clusters']}
- **SOM Utilization**: {st.session_state.som_stats['utilization']:.1f}%

## AI Analysis Summary
- **Model Used**: {st.session_state.get('ollama_model', 'mistral')} (Ollama Local)
- **Clusters Analyzed**: {len(successful_analyses)}
- **Success Rate**: {(len(successful_analyses) / len(st.session_state.cluster_analyses)) * 100:.1f}%
- **Framework**: Langchain + Ollama
- **Privacy**: Complete local processing

## Executive Summary
{st.session_state.overall_analysis}

## Top Cluster Summaries
"""

            for i, analysis in enumerate(successful_analyses[:10]):
                report_content += f"""
### Cluster {analysis['cluster_id']} ({analysis['total_emails']} emails)
{analysis['analysis'][:500]}...

"""

            report_content += f"""
## Technical Details
- **Feature Dimensions**: {st.session_state.som.input_dim}
- **Training Epochs**: {epochs}
- **Learning Rate**: {learning_rate}
- **Ollama Model**: {st.session_state.get('ollama_model', 'mistral')}
- **Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

            st.download_button(
                label="📄 Download Analysis Report",
                data=report_content,
                file_name="ollama_som_analysis_report.md",
                mime="text/markdown",
                help="Comprehensive analysis report in Markdown format"
            )

        with col3:
            # Cluster analysis results (JSON format for further processing)
            analysis_results = {
                'metadata': {
                    'analysis_date': datetime.now().isoformat(),
                    'total_emails': st.session_state.som_stats['total_emails'],
                    'model_used': st.session_state.get('ollama_model', 'mistral'),
                    'framework': 'ollama_langchain',
                    'som_config': {
                        'width': som_width,
                        'height': som_height,
                        'epochs': epochs,
                        'learning_rate': learning_rate
                    }
                },
                'cluster_analyses': st.session_state.cluster_analyses,
                'overall_analysis': st.session_state.overall_analysis,
                'som_stats': st.session_state.som_stats
            }

            json_buffer = io.StringIO()
            json_lib.dump(analysis_results, json_buffer, indent=2, default=str)

            st.download_button(
                label="🔬 Download Analysis Data",
                data=json_buffer.getvalue(),
                file_name="ollama_som_analysis_data.json",
                mime="application/json",
                help="Complete analysis results in JSON format for further processing"
            )

    elif hasattr(st.session_state, 'som'):
        # Show SOM results even without LLM analysis
        st.subheader("📈 SOM Results (Without AI Analysis)")
        st.info("SOM clustering completed. Configure Ollama for detailed cluster analysis.")

        # Basic cluster information
        clusters = st.session_state.clusters
        cluster_sizes = [len(emails) for emails in clusters.values()]

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Active Clusters", len(clusters))
        with col2:
            st.metric("Largest Cluster", max(cluster_sizes))
        with col3:
            st.metric("Average Cluster Size", f"{np.mean(cluster_sizes):.1f}")

    else:
        if not hasattr(st.session_state, 'emails_data'):
            st.info("👆 Please click 'Load All Data' to load your 7000+ emails and start the analysis.")
        else:
            st.info(
                "👆 Configure parameters and click 'Run Complete Analysis' to process all emails with Ollama Mistral.")

        # Show Ollama + Langchain benefits
        with st.expander("🚀 Why Ollama + Mistral for Email Analysis?", expanded=False):
            st.markdown("""
            **🦙 Ollama Framework Benefits:**
            - **🔒 Complete Privacy**: All processing happens locally on your machine
            - **💰 Zero API Costs**: No charges for processing thousands of emails
            - **⚡ High Performance**: Optimized local inference without network delays
            - **🔄 Unlimited Usage**: Process as many emails as you want, anytime
            - **🛡️ Enterprise Security**: Your sensitive emails never leave your environment

            **🧠 Mistral Model Advantages:**
            - **🎯 Business Context**: Excellent understanding of organizational communication
            - **🌐 Multilingual**: Handles emails in multiple languages seamlessly
            - **⚖️ Balanced Performance**: Great results with reasonable resource requirements
            - **🔍 Detailed Analysis**: Provides deep insights into communication patterns

            **📊 Perfect for 7000+ Emails:**
            - **🏃‍♂️ Scalable Processing**: Efficiently handles large datasets
            - **🧩 Pattern Recognition**: Identifies complex business communication patterns
            - **📈 Strategic Insights**: Provides actionable recommendations for decision makers
            - **🔄 Consistent Analysis**: Maintains quality across all cluster analyses

            **🔧 Easy Setup:**
            ```bash
            # Install Ollama
            curl -fsSL https://ollama.ai/install.sh | sh

            # Start Ollama
            ollama serve

            # Pull Mistral model
            ollama pull mistral
            ```
            """)

        # Show example analysis format
        with st.expander("📋 Sample Ollama Mistral Analysis Output", expanded=False):
            st.markdown("""
            **Example Local AI Cluster Analysis:**

            ## **CLUSTER THEME**
            Customer Escalation & Resolution Communications

            ## **COMMUNICATION PATTERNS**
            - **Type**: Mixed internal coordination and customer-facing communications
            - **Tone**: Professional but urgent, solution-oriented approach
            - **Participants**: Support managers, technical specialists, account managers, customers

            ## **BUSINESS CONTEXT**
            - **Process**: Customer issue escalation and resolution workflows
            - **Priority**: High urgency items requiring immediate cross-team coordination
            - **Departments**: Customer Success, Engineering, Product Management, Sales

            ## **KEY INSIGHTS**
            - Clear escalation procedures with well-defined handoff points
            - Strong collaboration between support and technical teams
            - Average resolution time tracked and optimized
            - Customer satisfaction metrics actively monitored
            - Opportunity to automate routine escalation notifications
            - Evidence of proactive communication with affected customers

            ## **CLUSTER SUMMARY**
            This cluster represents critical customer escalation communications with 47 emails showing structured problem-solving workflows. The communications demonstrate effective cross-functional coordination with clear accountability and customer-first mindset.

            *This detailed analysis is generated locally for each cluster - completely private and cost-free.*
            """)

        # Show Ollama installation guide
        with st.expander("🛠️ Ollama Installation & Setup Guide", expanded=False):
            st.markdown("""
            **Step-by-Step Setup:**

            **1. Install Ollama**
            ```bash
            # On macOS/Linux
            curl -fsSL https://ollama.ai/install.sh | sh

            # On Windows
            # Download from https://ollama.ai/download/windows
            ```

            **2. Start Ollama Service**
            ```bash
            ollama serve
            ```

            **3. Pull Mistral Model**
            ```bash
            # Standard Mistral model
            ollama pull mistral

            # Or specific version
            ollama pull mistral:7b
            ```

            **4. Verify Installation**
            ```bash
            # List available models
            ollama list

            # Test model
            ollama run mistral "Hello, how are you?"
            ```

            **5. Configure in App**
            - Ensure Ollama is running (`ollama serve`)
            - Use default URL: `http://localhost:11434`
            - Select your installed model (e.g., `mistral`)
            - Click "Test Ollama Connection"

            **System Requirements:**
            - **RAM**: 8GB minimum (16GB recommended for better performance)
            - **Storage**: 4-7GB for Mistral model
            - **CPU**: Modern multi-core processor (GPU optional but beneficial)
            """)


if __name__ == "__main__":
    main()