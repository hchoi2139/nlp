import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
import spacy
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE
import os

# ==========================================
# 0. Setup and Data Loading
# ==========================================
os.makedirs('eda_plots', exist_ok=True)
print("Loading spaCy model...")
nlp = spacy.load("en_core_web_sm")

print("Loading Data...")
# --- Load Binary Data ---
# Note: Ensure these file paths match your local setup
filepath_pcl = 'dontpatronizeme_pcl.tsv'
df_pcl = pd.read_csv(filepath_pcl, sep='\t', skiprows=4, header=None, 
                     names=['par_id', 'art_id', 'keyword', 'country', 'text', 'label'])
df_pcl = df_pcl.dropna(subset=['text', 'label'])
df_pcl['par_id'] = df_pcl['par_id'].astype(str)

# Load official splits
df_train_split = pd.read_csv('practice-splits/train_semeval_parids-labels.csv')
df_dev_split = pd.read_csv('practice-splits/dev_semeval_parids-labels.csv')
official_ids = set(df_train_split['par_id'].astype(str).tolist() + df_dev_split['par_id'].astype(str).tolist())

# Filter for official split & Apply Label 2+3+4 mapping
df_official = df_pcl[df_pcl['par_id'].isin(official_ids)].copy()
df_official['label'] = pd.to_numeric(df_official['label'], errors='coerce')
df_official['binary_label'] = df_official['label'].apply(lambda x: 1 if x >= 2 else 0)

# --- Load Category Data for Semantic Analysis ---
filepath_cat = 'dontpatronizeme_categories.tsv'
df_cat = pd.read_csv(filepath_cat, sep='\t', skiprows=3, header=None,
                     names=['par_id', 'art_id', 'text', 'keyword', 'country', 
                            'span_start', 'span_end', 'span_text', 'category', 'num_annotators'])
df_cat['par_id'] = df_cat['par_id'].astype(str)
df_cat_official = df_cat[df_cat['par_id'].isin(official_ids)].copy()

# Separate PCL (1) and Non-PCL (0) texts for comparison
pcl_texts = df_official[df_official['binary_label'] == 1]['text'].tolist()
non_pcl_texts = df_official[df_official['binary_label'] == 0]['text'].tolist()

print(f"Data Loaded. Binary Rows: {len(df_official)}")

# ==============================================================================
# ANALYSIS 1: LEXICAL PROFILING (Log-Odds Keyness)
# Objective: Find words statistically unique to PCL (not just frequent ones)
# ==============================================================================
print("\n[1/3] Running Lexical Analysis (Keyness)...")

def get_word_counts(texts):
    counts = Counter()
    # Disable heavy pipeline components for speed
    for doc in nlp.pipe(texts, disable=["ner", "parser", "tagger"]):
        for token in doc:
            if not token.is_stop and token.is_alpha:
                counts[token.lemma_.lower()] += 1
    return counts

pcl_counts = get_word_counts(pcl_texts)
non_pcl_counts = get_word_counts(non_pcl_texts)
total_pcl = sum(pcl_counts.values())
total_non_pcl = sum(non_pcl_counts.values())

def calculate_log_odds(count_a, count_b, total_a, total_b):
    # Log-Odds Ratio with Dirichlet prior smoothing
    return np.log((count_a + 1) / (total_a + 1)) - np.log((count_b + 1) / (total_b + 1))

keyness = {}
all_words = set(pcl_counts.keys()).union(set(non_pcl_counts.keys()))

for word in all_words:
    if pcl_counts[word] > 10: # Filter low-frequency noise
        keyness[word] = calculate_log_odds(pcl_counts[word], non_pcl_counts[word], total_pcl, total_non_pcl)

# Get top 20 distinctive words
keyness_df = pd.Series(keyness).sort_values(ascending=False).head(20)

plt.figure(figsize=(10, 8))
sns.barplot(x=keyness_df.values, y=keyness_df.index, palette='rocket')
plt.title('Lexical Profile: Words Most Distinctive to PCL (Log-Odds)')
plt.xlabel('Log-Odds Ratio (Higher = More Specific to PCL)')
plt.savefig('eda_plots/lexical_keyness.png', bbox_inches='tight')
plt.close()

# ==============================================================================
# ANALYSIS 2: SEMANTIC PROFILING (NER Bias & t-SNE Clustering)
# Objective A: Detect Entity Bias (Are we targeting specific groups?)
# Objective B: Visualise Category Overlap (Do strategies cluster?)
# ==============================================================================
print("\n[2/3] Running Semantic Analysis (NER & t-SNE)...")

# --- A. NER Distribution ---
def get_ner_distribution(texts):
    ner_counts = Counter()
    total_ents = 0
    # Enable NER only
    for doc in nlp.pipe(texts, disable=["parser", "tagger"]):
        for ent in doc.ents:
            ner_counts[ent.label_] += 1
            total_ents += 1
    if total_ents == 0: return {}
    return {k: v / total_ents for k, v in ner_counts.items()}

pcl_ner = get_ner_distribution(pcl_texts)
non_pcl_ner = get_ner_distribution(non_pcl_texts)

ner_df = pd.DataFrame([pcl_ner, non_pcl_ner], index=['PCL', 'Non-PCL']).T.fillna(0)
ner_df['diff'] = (ner_df['PCL'] - ner_df['Non-PCL']).abs()
ner_df = ner_df.sort_values(by='diff', ascending=False).head(8)

plt.figure(figsize=(10, 5))
ner_df[['PCL', 'Non-PCL']].plot(kind='bar', color=['#d62728', '#1f77b4'], width=0.8)
plt.title('Semantic Framing: NER Entity Distribution')
plt.ylabel('Proportion of Entities')
plt.xticks(rotation=45)
plt.savefig('eda_plots/semantic_ner_bias.png', bbox_inches='tight')
plt.close()

# --- B. t-SNE Clustering ---
print("      Generating t-SNE Embeddings (Loading SBERT)...")
model = SentenceTransformer('all-MiniLM-L6-v2')
df_viz = df_cat_official.copy()
if len(df_viz) > 3000: df_viz = df_viz.sample(3000, random_state=42) # Downsample for speed

embeddings = model.encode(df_viz['text'].tolist(), show_progress_bar=True)
tsne = TSNE(n_components=2, perplexity=30, random_state=42, init='pca', learning_rate=200)
tsne_results = tsne.fit_transform(embeddings)

df_viz['x'] = tsne_results[:, 0]
df_viz['y'] = tsne_results[:, 1]

plt.figure(figsize=(12, 8))
sns.scatterplot(data=df_viz, x='x', y='y', hue='category', palette='tab10', alpha=0.7, s=60)
plt.title('Semantic Landscape: t-SNE Projection of PCL Categories', fontsize=14)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('eda_plots/semantic_tsne_clusters.png')
plt.close()

# ==============================================================================
# ANALYSIS 3: SYNTACTIC PROFILING (POS Density)
# Objective: Detect "Decorative" vs "Factual" language structure
# ==============================================================================
print("\n[3/3] Running Syntactic Analysis (POS Density)...")

def get_pos_density(texts):
    pos_counts = Counter()
    total_tokens = 0
    # Enable tagger only
    for doc in nlp.pipe(texts, disable=["ner", "parser"]):
        for token in doc:
            if not token.is_punct and not token.is_space:
                pos_counts[token.pos_] += 1
                total_tokens += 1
    return {k: v / total_tokens for k, v in pos_counts.items()}

pcl_pos = get_pos_density(pcl_texts)
non_pcl_pos = get_pos_density(non_pcl_texts)

pos_df = pd.DataFrame([pcl_pos, non_pcl_pos], index=['PCL', 'Non-PCL']).T.fillna(0)
pos_df['diff'] = (pos_df['PCL'] - pos_df['Non-PCL']) 
pos_df_sorted = pos_df.sort_values(by='diff', ascending=False)

plt.figure(figsize=(12, 6))
top_tags = pd.concat([pos_df_sorted.head(5), pos_df_sorted.tail(5)])
top_tags[['PCL', 'Non-PCL']].plot(kind='bar', color=['#d62728', '#1f77b4'], width=0.8)
plt.title('Syntactic Signature: POS Tag Density')
plt.ylabel('Density (Count / Total Tokens)')
plt.xticks(rotation=45)
plt.savefig('eda_plots/syntactic_pos_density.png', bbox_inches='tight')
plt.close()

print("\nProcessing Complete. Check 'eda_plots/' for 4 new images.")