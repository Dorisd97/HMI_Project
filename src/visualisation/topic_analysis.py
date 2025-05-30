import json
import re
import os
from collections import Counter
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# 📍 Step 1: Load data
data_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'cleaned_enron.json')

with open(data_path, "r", encoding="utf-8") as f:
    emails = json.load(f)

# 📍 Step 2: Extract and clean all email bodies
bodies = [email.get("Body", "") for email in emails]
text = " ".join(bodies).lower()
text = re.sub(r'[^a-z\s]', '', text)

# 📍 Step 3: Define target topics/entities
topics = ["dynegy", "merger", "california", "crisis", "lawsuit", "energy", "enron", "deal", "ferc", "ab1890"]

# 📍 Step 4: Count topic mentions
topic_counts = Counter({topic: text.count(topic) for topic in topics})
df_topics = pd.DataFrame(topic_counts.items(), columns=["Topic", "Frequency"]).sort_values(by="Frequency", ascending=False)

# 📍 Step 5: Plot as bar chart
plt.figure(figsize=(10, 6))
sns.barplot(data=df_topics, x="Frequency", y="Topic", palette="viridis")
plt.title("Most Discussed Topics in Emails")
plt.xlabel("Mentions")
plt.ylabel("Topics")
plt.tight_layout()
plt.show()
