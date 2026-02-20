import sys
import os

# 1. Add the project root (nlp/) to Python's path so it can find the 'src' folder
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

# 2. Now the import will work perfectly
from src.data_loader import get_dataloader

# 3. Make sure the filepath points to the new data/ folder
filepath = 'data/dontpatronizeme_pcl.tsv' 

dataloader, tokenizer = get_dataloader(filepath, batch_size=8)

# Grab one batch
batch = next(iter(dataloader))

print("\n--- BATCH TEST ---")
print(f"Input IDs shape: {batch['input_ids'].shape}")
print(f"Attention Mask shape: {batch['attention_mask'].shape}")
print(f"Labels shape: {batch['labels'].shape}")
print(f"Labels in this batch: {batch['labels']}")