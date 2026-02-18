import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create a folder to save your plots so they don't clutter your working directory
os.makedirs('eda_plots', exist_ok=True)

# ==========================================
# 1. Load and Filter the Binary Data
# ==========================================
filepath_pcl = 'dontpatronizeme_pcl.tsv'
df_pcl = pd.read_csv(filepath_pcl, sep='\t', skiprows=4, header=None, 
                     names=['par_id', 'art_id', 'keyword', 'country', 'text', 'label'])

# Clean up any malformed rows
df_pcl = df_pcl.dropna(subset=['text', 'label'])
df_pcl['par_id'] = df_pcl['par_id'].astype(str)

# Load the official SemEval practice splits CSVs
df_train_split = pd.read_csv('practice-splits/train_semeval_parids-labels.csv')
df_dev_split = pd.read_csv('practice-splits/dev_semeval_parids-labels.csv')

# Extract the 'par_id' column from both splits and combine them into a set
train_ids = df_train_split['par_id'].astype(str).tolist()
dev_ids = df_dev_split['par_id'].astype(str).tolist()
official_ids = set(train_ids + dev_ids)

# Filter the main dataframe to only include the official training and dev paragraphs
df_official = df_pcl[df_pcl['par_id'].isin(official_ids)].copy()

print(f"Total rows in raw TSV: {len(df_pcl)}")
print(f"Total rows in official train+dev splits: {len(df_official)}")

# Create binary label (0 and 1 are negative, 2, 3, and 4 are positive)
df_official['label'] = pd.to_numeric(df_official['label'], errors='coerce')
df_official['binary_label'] = df_official['label'].apply(lambda x: 1 if x >= 2 else 0)

# ==========================================
# 2. Subtask 1: Binary Class Distribution 
# ==========================================
plt.figure(figsize=(8, 5))
ax1 = sns.countplot(data=df_official, x='binary_label', hue='binary_label', palette='viridis', legend=False)
plt.title('Subtask 1: Binary Class Distribution (0: No PCL, 1: PCL)')
plt.xlabel('Class (0 = Negative, 1 = Positive)')
plt.ylabel('Number of Paragraphs')

# Annotate EXACT NUMBERS and percentages 
total = len(df_official)
for p in ax1.patches:
    count = int(p.get_height())
    percentage = f'{100 * count / total:.1f}%'
    text = f'{count} ({percentage})'
    x = p.get_x() + p.get_width() / 2
    y = p.get_height()
    ax1.annotate(text, (x, y), ha='center', va='bottom', fontweight='bold')

plt.savefig('eda_plots/binary_class_distribution.png', bbox_inches='tight')
plt.close()
print("Saved Subtask 1 plot to eda_plots/binary_class_distribution.png")

# ==========================================
# 3. Token Count / Sequence Length Profiling
# ==========================================
df_official['word_count'] = df_official['text'].apply(lambda x: len(str(x).split()))

plt.figure(figsize=(10, 5))
ax2 = sns.histplot(df_official['word_count'], bins=50, kde=True, color='teal')
plt.title('Distribution of Paragraph Word Counts')
plt.xlabel('Number of Words')
plt.ylabel('Frequency')

# Calculate exact stats
min_words = int(df_official['word_count'].min())
max_words = int(df_official['word_count'].max())
avg_words = df_official['word_count'].mean()

# Calculate both Percentiles
percentile_95 = df_official['word_count'].quantile(0.95)
percentile_995 = df_official['word_count'].quantile(0.995)
exact_word_count_95 = int(percentile_95)
exact_word_count_995 = int(percentile_995)

# Print updated stats to the terminal
print(f"Word Count Stats -> Min: {min_words}, Max: {max_words}, Avg: {avg_words:.1f}")
print(f"Quantiles -> 95th: {exact_word_count_95}, 99.5th: {exact_word_count_995}")

# --- Add 95th Percentile markers ---
plt.axvline(percentile_95, color='red', linestyle='dashed', 
            label=f'95th Percentile: {exact_word_count_95} words')

# --- Add 99.5th Percentile markers (NEW) ---
plt.axvline(percentile_995, color='orange', linestyle='dashed', 
            label=f'99.5th Percentile: {exact_word_count_995} words')

# Add exact number text directly on the plot
ymin, ymax = ax2.get_ylim()
plt.text(percentile_95 + (df_official['word_count'].max() * 0.01), ymax * 0.9, 
         f'{exact_word_count_95}', color='red', fontweight='bold')

# Add text for 99.5th (offset slightly lower so they don't overlap)
plt.text(percentile_995 + (df_official['word_count'].max() * 0.01), ymax * 0.8, 
         f'{exact_word_count_995}', color='orange', fontweight='bold')

# Update the text box with the new metric
stats_text = (f"Minimum: {min_words} words\n"
              f"Maximum: {max_words} words\n"
              f"Average: {avg_words:.1f} words\n"
              f"99.5th %ile: {exact_word_count_995} words")

plt.text(0.95, 0.5, stats_text, transform=ax2.transAxes, fontsize=11,
         verticalalignment='center', horizontalalignment='right',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='gray', alpha=0.9))

plt.legend()
plt.savefig('eda_plots/sequence_length_distribution.png', bbox_inches='tight')
plt.close()
print("Saved sequence length plot with 99.5th percentile to eda_plots/sequence_length_distribution.png")

# ==========================================
# 4. Load and Filter Category Data
# ==========================================
filepath_cat = 'dontpatronizeme_categories.tsv'
df_cat = pd.read_csv(filepath_cat, sep='\t', skiprows=3, header=None,
                     names=['par_id', 'art_id', 'text', 'keyword', 'country', 
                            'span_start', 'span_end', 'span_text', 'category', 'num_annotators'])

df_cat['par_id'] = df_cat['par_id'].astype(str)
# Filter categorization data to match official splits
df_cat_official = df_cat[df_cat['par_id'].isin(official_ids)]

# ==========================================
# 5. Subtask 2: Category Distribution
# ==========================================
plt.figure(figsize=(10, 6))
# Order the categories by frequency to clearly show the "long tail"
order = df_cat_official['category'].value_counts().index
ax3 = sns.countplot(data=df_cat_official, y='category', hue='category', order=order, palette='magma', legend=False)
plt.title('Subtask 2: PCL Taxonomic Category Distribution')
plt.xlabel('Number of Categorized Spans')
plt.ylabel('PCL Category')

# Annotate EXACT NUMBERS on the tip of the horizontal bars
for p in ax3.patches:
    count = int(p.get_width())
    if count > 0:  # Check to ignore empty patches 
        x = p.get_width()
        y = p.get_y() + p.get_height() / 2
        ax3.annotate(f' {count}', (x, y), ha='left', va='center', fontweight='bold')

# Expand x-axis slightly so the text doesn't hit the edge of the image
xmax = ax3.get_xlim()[1]
ax3.set_xlim(0, xmax * 1.1)

plt.savefig('eda_plots/category_distribution.png', bbox_inches='tight')
plt.close()
print("Saved Subtask 2 plot to eda_plots/category_distribution.png")

# ==========================================
# 6. Subtask 2: Label Co-occurrence Heatmap
# ==========================================
df_cat_dummies = pd.get_dummies(df_cat_official['category'])
df_cat_grouped = df_cat_dummies.groupby(df_cat_official['par_id']).sum().clip(upper=1)
co_occurrence = df_cat_grouped.T.dot(df_cat_grouped)

plt.figure(figsize=(10, 8))
sns.heatmap(co_occurrence, annot=True, cmap='Blues', fmt='g')
plt.title('Subtask 2: PCL Category Co-occurrence Matrix')
plt.ylabel('PCL Category')
plt.xlabel('PCL Category')
plt.savefig('eda_plots/category_cooccurrence_heatmap.png', bbox_inches='tight')
plt.close()
print("Saved Co-occurrence Heatmap to eda_plots/category_cooccurrence_heatmap.png")