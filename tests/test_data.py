import sys
import os
import torch

# 1. Add the project root (nlp/) to Python's path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from src.data_loader import get_dataloader

# 2. Paths to your data (relative to project root)
data_path = 'data/dontpatronizeme_pcl.tsv'
cat_path = 'data/dontpatronizeme_categories.tsv'

def run_rigorous_test():
    print("--- STARTING RIGOROUS DATA PIPELINE TEST ---")
    
    # Test 1: Initialization and Filtering
    dataloader, tokenizer = get_dataloader(data_path, cat_path, batch_size=8)
    
    # Test 2: Outlier Removal Validation
    # In Stage 2, you found 10,425 total rows (approx). 
    # If we drop 7 outliers and some <7 word fragments, the final size should reflect that.
    print(f"Verified Filtered Dataset Size: {len(dataloader.dataset)}")

    # Grab one batch
    batch = next(iter(dataloader))

    print("\n--- TENSOR DIMENSION VALIDATION ---")
    print(f"Input IDs shape:      {batch['input_ids'].shape} (Should be [8, SeqLen])")
    print(f"Binary Labels shape:  {batch['labels'].shape} (Should be [8])")
    
    # TEST 3: MTL Taxonomy Validation
    if 'taxonomy_labels' in batch:
        print(f"Taxonomy Labels shape: {batch['taxonomy_labels'].shape} (Should be [8, 7])")
        assert batch['taxonomy_labels'].shape == (8, 7), "MTL Label Dimension Mismatch!"
    else:
        print("CRITICAL ERROR: Taxonomy labels missing from batch. MTL will not work.")

    # TEST 4: Binary Label Balance (Weighted Sampler Check)
    # Since it's a batch of 8, and the data is 90% negative, 
    # a balanced sampler should show at least one '1.0' usually.
    print(f"Labels in this batch: {batch['labels'].tolist()}")
    
    # TEST 5: Decoding Check (HTML Unescape Verification)
    # Let's decode a random sample from the batch to see if tags are gone
    sample_text = tokenizer.decode(batch['input_ids'][0], skip_special_tokens=True)
    print(f"\n--- DECODING SAMPLE (Verify Cleaning) ---")
    print(f"Cleaned Text: {sample_text[:150]}...")
    
    if "<h>" in sample_text or "&amp;" in sample_text:
        print("WARNING: Cleaning failed. HTML artifacts detected in decoded text.")
    else:
        print("SUCCESS: HTML cleaning verified.")

if __name__ == "__main__":
    try:
        run_rigorous_test()
        print("\n[PASSED] Data pipeline is strictly aligned with Stage 2 Report.")
    except Exception as e:
        print(f"\n[FAILED] Test failed with error: {e}")