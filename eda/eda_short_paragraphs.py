import pandas as pd
import numpy as np
import os

# Load Binary Data
filepath_pcl = '../data/dontpatronizeme_pcl.tsv'
df_pcl = pd.read_csv(filepath_pcl, sep='\t', skiprows=4, header=None, 
                     names=['par_id', 'art_id', 'keyword', 'country', 'text', 'label'])
df_pcl = df_pcl.dropna(subset=['text', 'label'])
df_pcl['word_count'] = df_pcl['text'].str.split().str.len()

# ==============================================================================
# RIGOROUS SHORT-SEQUENCE INSPECTION (Lengths 1 to 9)
# ==============================================================================
print(f"{'='*60}")
print(f"{'WORD COUNT':<12} | {'LABEL':<5} | {'TEXT'}")
print(f"{'='*60}")

for length in range(1, 10):
    subset = df_pcl[df_pcl['word_count'] == length]
    print(f"\n--- LENGTH: {length} words ({len(subset)} samples found) ---")
    
    # If there are too many, we take a sample of 20 to keep the terminal readable, 
    # but you can change .head(20) to see all.
    for _, row in subset.head(10).iterrows():
        print(f"{length:<12} | {row['label']:<5} | {row['text']}")

print(f"\n{'='*60}")
print("End of Inspection.")