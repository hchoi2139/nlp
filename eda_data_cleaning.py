import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os

# Setup
os.makedirs('eda_plots', exist_ok=True)

# Load Binary Data
filepath_pcl = 'dontpatronizeme_pcl.tsv'
df_pcl = pd.read_csv(filepath_pcl, sep='\t', skiprows=4, header=None, 
                     names=['par_id', 'art_id', 'keyword', 'country', 'text', 'label'])
df_pcl = df_pcl.dropna(subset=['text', 'label'])

# ==============================================================================
# 1. RIGOROUS DUPLICATE ANALYSIS (Data Leakage Risk)
# ==============================================================================
print("[1/3] Analyzing Duplicates...")

# Create a normalized version of the text for a stricter duplicate check
# 1. Lowercase everything
# 2. Strip leading/trailing whitespaces
# 3. Replace multiple spaces/newlines inside the text with a single space
df_pcl['normalized_text'] = df_pcl['text'].str.lower().str.strip()
df_pcl['normalized_text'] = df_pcl['normalized_text'].apply(lambda x: re.sub(r'\s+', ' ', str(x)))

# Now check for duplicates based on the normalized text
duplicates = df_pcl[df_pcl.duplicated(subset=['normalized_text'], keep=False)]

num_duplicates = len(duplicates)
unique_dupes = duplicates['normalized_text'].nunique()

print(f"Total Duplicate Rows (Exact & Near-matches): {num_duplicates}")
print(f"Unique Duplicate Sentences: {unique_dupes}\n")

# Optional: Print a few examples of the duplicates if any are found
if num_duplicates > 0:
    print("--- EXAMPLES OF HIDDEN DUPLICATES FOUND ---")
    # Sort by normalized text so the duplicates appear next to each other
    dupe_examples = duplicates.sort_values(by='normalized_text').head(6)
    for idx, row in dupe_examples.iterrows():
        print(f"Row {idx} | Label {row['label']}: {row['text'][:80]}...")
    print("-------------------------------------------\n")

# ==============================================================================
# 2. SEQUENCE LENGTH / OUTLIERS
# ==============================================================================
print("[2/3] Analyzing Outliers...")
df_pcl['word_count'] = df_pcl['text'].str.split().str.len()

plt.figure(figsize=(10, 6))
sns.boxplot(x='label', y='word_count', hue='label', data=df_pcl, palette='Set2', legend=False)

# Updated annotations based on our 7-word pivot finding
plt.annotate('n=23 (<7 words)', xy=(0, 7), xytext=(0.5, 300),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5),
             fontsize=11, weight='bold')
plt.annotate('n=7 (>250 words)', xy=(0, 250), xytext=(2, 350),
             arrowprops=dict(facecolor='red', shrink=0.05, width=1, headwidth=5),
             fontsize=11, weight='bold')

plt.title('Sequence Length Outliers by Label')
plt.ylabel('Word Count')
plt.xlabel('PCL Label (0-4)')
plt.savefig('eda_plots/length_outliers.png')
plt.close()

# ==============================================================================
# 3. COMPREHENSIVE ARTIFACT & NOISE DETECTION
# ==============================================================================
print("[3/3] Identifying Tokenizer Noise...")

def count_artifacts(text):
    artifacts = {
        # Catches both named (&amp;) and numeric (&#39;) entities
        'html_entities': len(re.findall(r'&[a-zA-Z0-9#]+;', text)), 
        
        # Catches raw HTML tags like <h>, <p>, <br>
        'html_tags': len(re.findall(r'<[^>]+>', text)), 
        
        # Catches @mentions and #hashtags
        'social_media_tags': len(re.findall(r'[@#]\w+', text)), 
        
        # Catches tabs, multiple spaces, and weird whitespace like \xa0
        'extra_whitespace': len(re.findall(r'\s{2,}', text)), 
        
        # Catches URLs starting with http or www
        'urls': len(re.findall(r'(?:http[s]?://|www\.)\S+', text))
    }
    return pd.Series(artifacts)

# Apply the counting function
artifact_counts = df_pcl['text'].apply(count_artifacts)
df_artifacts = pd.concat([df_pcl['label'], artifact_counts], axis=1)

# Group by label to see the distribution
avg_artifacts = df_artifacts.groupby('label').mean()

# Plotting the stacked bar chart
plt.figure(figsize=(12, 6))
avg_artifacts.plot(kind='bar', stacked=True, colormap='viridis', figsize=(10,6))

plt.title('Average Noise Artifacts per Label')
plt.ylabel('Average Count per Paragraph')
plt.xlabel('PCL Label (0-4)')
plt.legend(title='Artifact Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout() # Ensures the legend doesn't get cut off
plt.savefig('eda_plots/artifact_distribution.png')
plt.close()

print("Artifact analysis complete. Check 'eda_plots/artifact_distribution.png'")