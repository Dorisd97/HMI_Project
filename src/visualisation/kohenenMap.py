import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from datetime import datetime
import re
from collections import Counter
import warnings

from src.config.config import PROCESSED_JSON_OUTPUT

warnings.filterwarnings('ignore')


class KohonenSOM:
    def __init__(self, width, height, input_dim, learning_rate=0.1, sigma=1.0):
        """
        Initialize the Kohonen Self-Organizing Map

        Args:
            width (int): Width of the SOM grid
            height (int): Height of the SOM grid
            input_dim (int): Number of input features
            learning_rate (float): Initial learning rate
            sigma (float): Initial neighborhood radius
        """
        self.width = width
        self.height = height
        self.input_dim = input_dim
        self.learning_rate = learning_rate
        self.sigma = sigma

        # Initialize weights randomly
        self.weights = np.random.random((height, width, input_dim))

        # Create coordinate arrays for distance calculations
        self.locations = np.array([[i, j] for i in range(height) for j in range(width)])
        self.locations = self.locations.reshape((height, width, 2))

    def _find_best_matching_unit(self, x):
        """Find the Best Matching Unit (BMU) for input vector x"""
        distances = np.sum((self.weights - x) ** 2, axis=2)
        bmu_idx = np.unravel_index(np.argmin(distances), distances.shape)
        return bmu_idx

    def _update_weights(self, x, bmu_idx, iteration, total_iterations):
        """Update weights based on BMU and neighborhood function"""
        # Decay learning rate and neighborhood radius
        current_learning_rate = self.learning_rate * np.exp(-iteration / total_iterations)
        current_sigma = self.sigma * np.exp(-iteration / total_iterations)

        # Calculate distances from BMU
        bmu_location = np.array([bmu_idx[0], bmu_idx[1]])
        distances_to_bmu = np.sum((self.locations - bmu_location) ** 2, axis=2)

        # Calculate neighborhood function (Gaussian)
        neighborhood = np.exp(-distances_to_bmu / (2 * current_sigma ** 2))

        # Update weights
        for i in range(self.height):
            for j in range(self.width):
                self.weights[i, j] += (current_learning_rate * neighborhood[i, j] *
                                       (x - self.weights[i, j]))

    def train(self, data, epochs=1000):
        """Train the SOM with the given data"""
        print(f"Training SOM with {len(data)} samples for {epochs} epochs...")

        for epoch in range(epochs):
            # Randomly select a training sample
            idx = np.random.randint(0, len(data))
            x = data[idx]

            # Find BMU and update weights
            bmu_idx = self._find_best_matching_unit(x)
            self._update_weights(x, bmu_idx, epoch, epochs)

            if (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch + 1}/{epochs} completed")

    def map_data(self, data):
        """Map data points to their BMUs"""
        mapped_data = []
        for x in data:
            bmu_idx = self._find_best_matching_unit(x)
            mapped_data.append(bmu_idx)
        return np.array(mapped_data)

    def get_u_matrix(self):
        """Calculate U-matrix (unified distance matrix)"""
        u_matrix = np.zeros((self.height, self.width))

        for i in range(self.height):
            for j in range(self.width):
                neighbors = []
                # Check all 8 neighbors
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        if di == 0 and dj == 0:
                            continue
                        ni, nj = i + di, j + dj
                        if 0 <= ni < self.height and 0 <= nj < self.width:
                            neighbors.append(self.weights[ni, nj])

                if neighbors:
                    # Calculate average distance to neighbors
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
        """Extract comprehensive features from email data"""
        print("Extracting features from email data...")

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

        # Convert to DataFrame for easier processing
        df = pd.DataFrame(features)

        # Handle TF-IDF features for text content
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

        # Select numerical features for SOM
        numerical_features = df.select_dtypes(include=[np.number]).fillna(0)

        # Normalize features
        normalized_features = self.scaler.fit_transform(numerical_features)

        print(f"Extracted {normalized_features.shape[1]} features from {len(emails_data)} emails")
        return normalized_features, df

    def _extract_date_features(self, date_str):
        """Extract features from date string"""
        features = {
            'year': 2000,
            'month': 1,
            'day': 1,
            'hour': 0,
            'minute': 0
        }

        try:
            # Try to parse different date formats
            if '.' in date_str:
                # Format: "14.04.2000 09:58:00"
                date_part, time_part = date_str.split(' ')
                day, month, year = map(int, date_part.split('.'))
                hour, minute, _ = map(int, time_part.split(':'))
            else:
                # Add other date format parsers as needed
                return features

            features.update({
                'year': year,
                'month': month,
                'day': day,
                'hour': hour,
                'minute': minute
            })
        except:
            pass  # Use default values if parsing fails

        return features


def load_json_data(file_path, chunk_size=None):
    """
    Load JSON data from file, with optional chunking for large files

    Args:
        file_path (str): Path to JSON file
        chunk_size (int): If specified, process data in chunks
    """
    print(f"Loading data from {file_path}...")

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)

        # Handle different JSON structures
        if isinstance(data, list):
            emails = data
        elif isinstance(data, dict) and 'emails' in data:
            emails = data['emails']
        elif isinstance(data, dict) and 'data' in data:
            emails = data['data']
        else:
            # Assume the entire dict is one email or convert dict values to list
            emails = [data] if not isinstance(list(data.values())[0], list) else list(data.values())[0]

        print(f"Loaded {len(emails)} email records")

        # Apply chunking if specified
        if chunk_size and len(emails) > chunk_size:
            print(f"Using chunk size: {chunk_size}")
            emails = emails[:chunk_size]

        return emails

    except FileNotFoundError:
        print(f"Error: File {file_path} not found")
        return None
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        return None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None


def visualize_som_results(som, mapped_data, original_data, feature_names=None):
    """Visualize SOM results with multiple plots"""

    # Create subplot figure
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Kohonen Self-Organizing Map Results', fontsize=16)

    # 1. U-matrix (Unified Distance Matrix)
    u_matrix = som.get_u_matrix()
    im1 = axes[0, 0].imshow(u_matrix, cmap='viridis', interpolation='nearest')
    axes[0, 0].set_title('U-Matrix (Distance Map)')
    axes[0, 0].set_xlabel('SOM Width')
    axes[0, 0].set_ylabel('SOM Height')
    plt.colorbar(im1, ax=axes[0, 0])

    # 2. Data point distribution
    scatter_plot = axes[0, 1].scatter(mapped_data[:, 1], mapped_data[:, 0],
                                      c=range(len(mapped_data)), cmap='tab10', alpha=0.7)
    axes[0, 1].set_title('Data Points on SOM Grid')
    axes[0, 1].set_xlabel('SOM Width')
    axes[0, 1].set_ylabel('SOM Height')
    axes[0, 1].set_xlim(-0.5, som.width - 0.5)
    axes[0, 1].set_ylim(-0.5, som.height - 0.5)

    # 3. Density map
    density_map = np.zeros((som.height, som.width))
    for point in mapped_data:
        density_map[point[0], point[1]] += 1

    im3 = axes[1, 0].imshow(density_map, cmap='Reds', interpolation='nearest')
    axes[1, 0].set_title('Data Point Density')
    axes[1, 0].set_xlabel('SOM Width')
    axes[1, 0].set_ylabel('SOM Height')
    plt.colorbar(im3, ax=axes[1, 0])

    # 4. Classification distribution (if available)
    if len(original_data) > 0 and 'classification' in original_data[0]:
        classifications = [email.get('classification', 'Unknown') for email in original_data]
        unique_classes = list(set(classifications))
        class_colors = plt.cm.Set3(np.linspace(0, 1, len(unique_classes)))

        for i, (point, classification) in enumerate(zip(mapped_data, classifications)):
            class_idx = unique_classes.index(classification)
            axes[1, 1].scatter(point[1], point[0], c=[class_colors[class_idx]],
                               label=classification if i == classifications.index(classification) else "",
                               alpha=0.7)

        axes[1, 1].set_title('Classification Distribution')
        axes[1, 1].set_xlabel('SOM Width')
        axes[1, 1].set_ylabel('SOM Height')
        axes[1, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[1, 1].set_xlim(-0.5, som.width - 0.5)
        axes[1, 1].set_ylim(-0.5, som.height - 0.5)
    else:
        axes[1, 1].text(0.5, 0.5, 'Classification data\nnot available',
                        ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Classification Distribution')

    plt.tight_layout()
    return fig


def analyze_clusters(som, mapped_data, original_data, features_df):
    """Analyze and print cluster information"""
    print("\n" + "=" * 50)
    print("CLUSTER ANALYSIS")
    print("=" * 50)

    # Count emails per SOM node
    node_counts = {}
    for point in mapped_data:
        node = (point[0], point[1])
        node_counts[node] = node_counts.get(node, 0) + 1

    # Find most populated nodes
    top_nodes = sorted(node_counts.items(), key=lambda x: x[1], reverse=True)[:5]

    print(f"\nTop 5 most populated SOM nodes:")
    for (i, j), count in top_nodes:
        print(f"Node ({i}, {j}): {count} emails")

        # Find emails in this node
        node_emails = []
        for idx, point in enumerate(mapped_data):
            if point[0] == i and point[1] == j:
                node_emails.append(idx)

        # Show sample characteristics
        if node_emails and len(original_data) > 0:
            print(f"  Sample characteristics:")
            sample_idx = node_emails[0]
            email = original_data[sample_idx]
            print(f"    Subject: {email.get('subject', 'N/A')[:60]}...")
            print(f"    Classification: {email.get('classification', 'N/A')}")
            print(f"    Tone: {email.get('tone_analysis', 'N/A')}")
        print()


def main():
    """Main function to demonstrate SOM with large JSON dataset"""

    # Configuration
    JSON_FILE_PATH = PROCESSED_JSON_OUTPUT# Replace with your file path
    CHUNK_SIZE = 1000  # Process first 1000 emails for demo, set to None for all data
    SOM_WIDTH = 10
    SOM_HEIGHT = 8
    EPOCHS = 500

    print("Kohonen Self-Organizing Map for Large Email Dataset")
    print("=" * 60)

    # Step 1: Load JSON data
    emails_data = load_json_data(JSON_FILE_PATH, chunk_size=CHUNK_SIZE)
    if emails_data is None:
        print("Failed to load data. Please check your file path and format.")
        return

    # Step 2: Extract features
    feature_extractor = EmailFeatureExtractor()
    features, features_df = feature_extractor.extract_features(emails_data)

    # Step 3: Initialize and train SOM
    input_dim = features.shape[1]
    som = KohonenSOM(SOM_WIDTH, SOM_HEIGHT, input_dim,
                     learning_rate=0.1, sigma=max(SOM_WIDTH, SOM_HEIGHT) / 2)

    som.train(features, epochs=EPOCHS)

    # Step 4: Map data to SOM
    mapped_data = som.map_data(features)

    # Step 5: Visualize results
    fig = visualize_som_results(som, mapped_data, emails_data)
    plt.show()

    # Step 6: Analyze clusters
    analyze_clusters(som, mapped_data, emails_data, features_df)

    # Step 7: Save results (optional)
    save_results = input("\nSave results to files? (y/n): ").lower() == 'y'
    if save_results:
        # Save SOM weights
        np.save('som_weights.npy', som.weights)

        # Save mapped data
        results_df = features_df.copy()
        results_df['som_x'] = mapped_data[:, 1]
        results_df['som_y'] = mapped_data[:, 0]
        results_df.to_csv('som_results.csv', index=False)

        # Save visualization
        fig.savefig('som_visualization.png', dpi=300, bbox_inches='tight')

        print("Results saved!")
        print("- som_weights.npy: SOM weight matrix")
        print("- som_results.csv: Features and SOM coordinates")
        print("- som_visualization.png: Visualization plots")


if __name__ == "__main__":
    main()

# Example usage for different JSON formats:
"""
# Format 1: Array of email objects
[
    {"subject": "...", "summary": "...", "classification": "..."},
    {"subject": "...", "summary": "...", "classification": "..."}
]

# Format 2: Object with emails array
{
    "emails": [
        {"subject": "...", "summary": "...", "classification": "..."}
    ]
}

# Format 3: Object with data array
{
    "data": [
        {"subject": "...", "summary": "...", "classification": "..."}
    ]
}
"""