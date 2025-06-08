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

warnings.filterwarnings('ignore')

# Import the config
from src.config.config import PROCESSED_JSON_OUTPUT

# Set page configuration
st.set_page_config(
    page_title="Kohonen SOM Analyzer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)


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
        self.tfidf_vectorizer = TfidfVectorizer(max_features=30, stop_words='english')
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
    feature_names = features_df.select_dtypes(include=[np.number]).columns

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


def analyze_clusters_detailed(som, mapped_data, original_data):
    """Detailed cluster analysis"""
    node_data = {}

    for idx, point in enumerate(mapped_data):
        node = (point[0], point[1])
        if node not in node_data:
            node_data[node] = []
        node_data[node].append(idx)

    cluster_info = []
    for node, email_indices in node_data.items():
        if len(email_indices) > 0:
            # Get email characteristics for this cluster
            node_emails = [original_data[i] for i in email_indices]

            classifications = [email.get('classification', 'Unknown') for email in node_emails]
            tones = [email.get('tone_analysis', 'Unknown') for email in node_emails]

            cluster_info.append({
                'Node': f"({node[0]}, {node[1]})",
                'Email Count': len(email_indices),
                'Primary Classification': Counter(classifications).most_common(1)[0][0],
                'Primary Tone': Counter(tones).most_common(1)[0][0],
                'Sample Subject': node_emails[0].get('subject', 'N/A')[:50] + '...'
            })

    return pd.DataFrame(cluster_info).sort_values('Email Count', ascending=False)


def main():
    st.title("🧠 Kohonen Self-Organizing Map Analyzer")
    st.markdown("### Interactive Email Dataset Clustering and Visualization")
    st.info(f"📁 Loading data from: `{PROCESSED_JSON_OUTPUT}`")

    # Sidebar for configuration
    st.sidebar.header("Configuration")

    # Load data automatically
    if st.sidebar.button("🔄 Load Data", type="primary"):
        with st.spinner("Loading dataset..."):
            emails_data = load_json_data_from_config()

        if emails_data is not None:
            st.session_state.emails_data = emails_data
            st.success(f"✅ Loaded {len(emails_data)} emails from dataset")

    # Check if data is loaded
    if hasattr(st.session_state, 'emails_data'):
        emails_data = st.session_state.emails_data

        # Data preview
        with st.expander("📊 Dataset Preview", expanded=False):
            if len(emails_data) > 0:
                sample_email = emails_data[0]
                st.json(sample_email)

                # Basic statistics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Emails", len(emails_data))
                with col2:
                    classifications = [email.get('classification', 'Unknown') for email in emails_data]
                    st.metric("Unique Classifications", len(set(classifications)))
                with col3:
                    avg_summary_length = np.mean([len(email.get('summary', '')) for email in emails_data])
                    st.metric("Avg Summary Length", f"{avg_summary_length:.0f}")

        # SOM Configuration
        st.sidebar.subheader("SOM Parameters")
        som_width = st.sidebar.slider("SOM Width", 5, 15, 10)
        som_height = st.sidebar.slider("SOM Height", 5, 15, 8)
        epochs = st.sidebar.slider("Training Epochs", 100, 2000, 500)
        learning_rate = st.sidebar.slider("Learning Rate", 0.01, 0.5, 0.1)

        # Data size limit
        max_emails = st.sidebar.slider("Max Emails to Process", 100, min(2000, len(emails_data)),
                                       min(1000, len(emails_data)))

        if st.sidebar.button("🚀 Train SOM", type="primary"):
            # Limit data size for performance
            working_data = emails_data[:max_emails]

            # Feature extraction
            with st.spinner("Extracting features..."):
                feature_extractor = EmailFeatureExtractor()
                features, features_df = feature_extractor.extract_features(working_data)

            st.success(f"✅ Extracted {features.shape[1]} features")

            # Train SOM
            st.subheader("🔄 Training Self-Organizing Map")
            progress_bar = st.progress(0)
            status_text = st.empty()

            def update_progress(current_epoch, total_epochs):
                progress = current_epoch / total_epochs
                progress_bar.progress(progress)
                status_text.text(f"Training: {current_epoch}/{total_epochs} epochs ({progress:.1%})")

            som = KohonenSOM(som_width, som_height, features.shape[1],
                             learning_rate=learning_rate, sigma=max(som_width, som_height) / 2)

            som.train(features, epochs=epochs, progress_callback=update_progress)

            progress_bar.progress(1.0)
            status_text.text("✅ Training completed!")

            # Map data to SOM
            mapped_data = som.map_data(features)

            # Store results in session state
            st.session_state.som = som
            st.session_state.mapped_data = mapped_data
            st.session_state.working_data = working_data
            st.session_state.features_df = features_df
            st.session_state.features = features

    # Display results if available
    if hasattr(st.session_state, 'som'):
        st.subheader("📈 SOM Visualization Results")

        # Brief Summary
        with st.container():
            st.markdown("### 📋 Analysis Summary")

            # Calculate summary statistics
            som = st.session_state.som
            mapped_data = st.session_state.mapped_data
            working_data = st.session_state.working_data

            # Basic metrics
            total_emails = len(working_data)
            unique_nodes = len(set(map(tuple, mapped_data)))
            som_size = som.width * som.height
            utilization = (unique_nodes / som_size) * 100

            # Cluster analysis for summary
            node_counts = {}
            for point in mapped_data:
                node = tuple(point)
                node_counts[node] = node_counts.get(node, 0) + 1

            largest_cluster_size = max(node_counts.values()) if node_counts else 0
            smallest_cluster_size = min(node_counts.values()) if node_counts else 0
            avg_cluster_size = total_emails / unique_nodes if unique_nodes > 0 else 0

            # Classification distribution
            classifications = [email.get('classification', 'Unknown') for email in working_data]
            classification_counts = Counter(classifications)
            most_common_class = classification_counts.most_common(1)[0] if classification_counts else ('Unknown', 0)

            # Display summary in columns
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(
                    label="📧 Emails Processed",
                    value=f"{total_emails:,}",
                    help="Total number of emails analyzed"
                )
                st.metric(
                    label="🎯 SOM Utilization",
                    value=f"{utilization:.1f}%",
                    help=f"Percentage of SOM nodes with emails ({unique_nodes}/{som_size})"
                )

            with col2:
                st.metric(
                    label="🏘️ Active Clusters",
                    value=f"{unique_nodes}",
                    help="Number of SOM nodes containing emails"
                )
                st.metric(
                    label="📊 Avg Cluster Size",
                    value=f"{avg_cluster_size:.1f}",
                    help="Average number of emails per cluster"
                )

            with col3:
                st.metric(
                    label="🔝 Largest Cluster",
                    value=f"{largest_cluster_size}",
                    help="Size of the largest email cluster"
                )
                st.metric(
                    label="🔻 Smallest Cluster",
                    value=f"{smallest_cluster_size}",
                    help="Size of the smallest email cluster"
                )

            with col4:
                st.metric(
                    label="🏷️ Top Classification",
                    value=f"{most_common_class[0][:15]}...",
                    delta=f"{most_common_class[1]} emails",
                    help=f"Most common email classification: {most_common_class[0]}"
                )
                st.metric(
                    label="📏 Feature Dimensions",
                    value=f"{som.input_dim}",
                    help="Number of features used for clustering"
                )

            # Quick insights
            st.markdown("#### 🔍 Quick Insights")
            insights = []

            if utilization < 50:
                insights.append(
                    f"🎯 **Low SOM utilization ({utilization:.1f}%)** - Consider using a smaller SOM grid for better clustering")
            elif utilization > 90:
                insights.append(
                    f"🎯 **High SOM utilization ({utilization:.1f}%)** - Well-distributed clustering across the grid")

            if largest_cluster_size > avg_cluster_size * 3:
                insights.append(
                    f"📊 **Dominant cluster detected** - One cluster contains {largest_cluster_size} emails ({largest_cluster_size / total_emails * 100:.1f}% of data)")

            if len(classification_counts) > 1:
                second_most_common = classification_counts.most_common(2)[1][1] if len(classification_counts) > 1 else 0
                if most_common_class[1] > second_most_common * 2:
                    insights.append(
                        f"🏷️ **Classification imbalance** - '{most_common_class[0]}' dominates with {most_common_class[1] / total_emails * 100:.1f}% of emails")

            if unique_nodes < 10:
                insights.append(
                    f"🏘️ **Few active clusters ({unique_nodes})** - Emails are grouping into very distinct categories")
            elif unique_nodes > som_size * 0.8:
                insights.append(f"🏘️ **Many small clusters** - Emails are quite diverse with little grouping")

            if insights:
                for insight in insights:
                    st.markdown(f"- {insight}")
            else:
                st.markdown("- ✅ **Balanced clustering** - Good distribution of emails across clusters")

            st.markdown("---")

        # Create visualizations
        fig_u_matrix, fig_scatter, fig_density, fig_features = create_som_visualizations(
            st.session_state.som,
            st.session_state.mapped_data,
            st.session_state.working_data,
            st.session_state.features_df
        )

        # Display plots in tabs
        tab1, tab2, tab3, tab4 = st.tabs(
            ["🗺️ U-Matrix", "📍 Email Distribution", "🔥 Density Map", "⚖️ Feature Importance"])

        with tab1:
            st.plotly_chart(fig_u_matrix, use_container_width=True)
            st.info("💡 The U-Matrix shows cluster boundaries. Darker areas indicate cluster separations.")

        with tab2:
            st.plotly_chart(fig_scatter, use_container_width=True)
            st.info("💡 Each point represents an email, colored by classification. Hover for details!")

        with tab3:
            st.plotly_chart(fig_density, use_container_width=True)
            st.info("💡 Red areas show high email concentration. These are the main clusters.")

        with tab4:
            st.plotly_chart(fig_features, use_container_width=True)
            st.info("💡 Higher bars indicate more influential features in the clustering.")

        # Cluster Analysis
        st.subheader("🔍 Cluster Analysis")
        cluster_df = analyze_clusters_detailed(
            st.session_state.som,
            st.session_state.mapped_data,
            st.session_state.working_data
        )

        st.dataframe(cluster_df, use_container_width=True)

        # Download results
        st.subheader("💾 Download Results")
        col1, col2 = st.columns(2)

        with col1:
            # Prepare results DataFrame
            results_df = st.session_state.features_df.copy()
            results_df['som_x'] = st.session_state.mapped_data[:, 1]
            results_df['som_y'] = st.session_state.mapped_data[:, 0]

            csv_buffer = io.StringIO()
            results_df.to_csv(csv_buffer, index=False)

            st.download_button(
                label="📊 Download Results CSV",
                data=csv_buffer.getvalue(),
                file_name="som_results.csv",
                mime="text/csv"
            )

        with col2:
            # Save SOM weights
            weights_buffer = io.BytesIO()
            np.save(weights_buffer, st.session_state.som.weights)

            st.download_button(
                label="🧠 Download SOM Weights",
                data=weights_buffer.getvalue(),
                file_name="som_weights.npy",
                mime="application/octet-stream"
            )

    else:
        if not hasattr(st.session_state, 'emails_data'):
            st.info("👆 Please click 'Load Data' to load the JSON file and then train the SOM to see visualizations.")
        else:
            st.info("👆 Please configure SOM parameters and click 'Train SOM' to see visualizations.")

        # Show example JSON format
        with st.expander("📋 Expected JSON Format Examples", expanded=False):
            st.code('''
// Format 1: Array of email objects
[
    {
        "subject": "Project Update",
        "summary": "Discussion about project progress...",
        "classification": "Internal Communication",
        "tone_analysis": "Professional",
        "date": "14.04.2000 09:58:00",
        "to": "john@company.com",
        "from": "jane@company.com",
        "entities": {
            "people": ["John", "Jane"],
            "organizations": ["Company"],
            "locations": ["Office"]
        }
    }
]

// Format 2: Object with emails array
{
    "emails": [
        {...email objects...}
    ]
}
            ''', language='json')


if __name__ == "__main__":
    main()