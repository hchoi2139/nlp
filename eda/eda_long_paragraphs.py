import pandas as pd
import numpy as np

# Load Binary Data
filepath_pcl = '../data/dontpatronizeme_pcl.tsv'
df_pcl = pd.read_csv(filepath_pcl, sep='\t', skiprows=4, header=None, 
                     names=['par_id', 'art_id', 'keyword', 'country', 'text', 'label'])
df_pcl = df_pcl.dropna(subset=['text', 'label'])
df_pcl['word_count'] = df_pcl['text'].str.split().str.len()

# ==============================================================================
# RIGOROUS LONG-SEQUENCE INSPECTION (Lengths > 250)
# ==============================================================================
print(f"{'='*60}")
print(f"{'WORD COUNT':<12} | {'LABEL':<5} | {'TEXT'}")
print(f"{'='*60}")

# Filter for >250 words and sort by length for readability
long_subset = df_pcl[df_pcl['word_count'] > 250].sort_values(by='word_count')

print(f"\n--- EXTREMELY LONG OUTLIERS (>250 words) : {len(long_subset)} samples found ---\n")

for _, row in long_subset.iterrows():
    # Print the metadata
    print(f"WORD COUNT: {row['word_count']} | LABEL: {row['label']}")
    # Print the text, followed by a separator so the massive paragraphs don't blend together
    print(f"{row['text']}")
    print(f"{'-'*80}\n")

print(f"{'='*60}")
print("End of Inspection.")