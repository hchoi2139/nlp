import os
import sys
import csv
import pandas as pd
from collections import Counter
from tqdm import tqdm
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

# Ensure Python can find the src package
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.data_loader import clean_text

def run_error_analysis():
    print("--- INITIALIZING ERROR ANALYSIS & BASELINE COMPARISON ---")
    
    # 1. Load the exact 2094 true labels and text
    print("Loading Dev Set...")
    data_path = os.path.join(project_root, 'data', 'dontpatronizeme_pcl.tsv')
    raw_df = pd.read_csv(data_path, sep='\t', skiprows=4, header=None, names=['par_id', 'art_id', 'keyword', 'country', 'text', 'label'], quoting=csv.QUOTE_NONE)
    raw_df['par_id'] = raw_df['par_id'].astype(str)
    raw_df['binary_label'] = (raw_df['label'] >= 2).astype(int)
    
    # Load Categories to map to False Negatives (Preserving all duplicate spans)
    cat_path = os.path.join(project_root, 'data', 'dontpatronizeme_categories.tsv')
    if os.path.exists(cat_path):
        cat_df = pd.read_csv(cat_path, sep='\t', skiprows=4, header=None, 
                             names=['par_id', 'art_id', 'text', 'keyword', 'country', 'span_start', 'span_end', 'span_text', 'category', 'num_annotators'], quoting=csv.QUOTE_NONE)
        cat_df['par_id'] = cat_df['par_id'].astype(str)
        # Group categories by par_id as a complete list to preserve multiple annotations
        par_cats = cat_df.groupby('par_id')['category'].apply(list).reset_index()
        raw_df = pd.merge(raw_df, par_cats, on='par_id', how='left')
    else:
        raw_df['category'] = None

    dev_split_path = os.path.join(project_root, 'data', 'practice-splits', 'dev_semeval_parids-labels.csv')
    dev_split_df = pd.read_csv(dev_split_path)
    dev_split_df['par_id'] = dev_split_df['par_id'].astype(str)
    
    dev_df = pd.merge(dev_split_df[['par_id']], raw_df[['par_id', 'text', 'binary_label', 'category']], on='par_id', how='left')
    dev_df['text'] = dev_df['text'].apply(clean_text)
    
    # 2. Load your Ensemble Predictions
    my_preds_path = os.path.join(project_root, 'BestModel', 'dev.txt')
    with open(my_preds_path, 'r') as f:
        dev_df['my_pred'] = [int(line.strip()) for line in f.readlines()]
        
    # 3. Load/Cache the Baseline Model
    baseline_dir = os.path.join(project_root, 'checkpoints', 'baseline_roberta')
    
    if not os.path.exists(baseline_dir):
        print("Downloading and saving baseline model locally for future use...")
        os.makedirs(baseline_dir, exist_ok=True)
        tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/pcl_robertabase")
        model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/pcl_robertabase")
        tokenizer.save_pretrained(baseline_dir)
        model.save_pretrained(baseline_dir)
    else:
        print("Loading baseline model from local cache...")

    baseline_classifier = pipeline(
        "text-classification", 
        model=baseline_dir, 
        tokenizer=baseline_dir,
        device=0 # Uses GPU. Change to -1 if you are out of VRAM.
    )
    
    print("Generating Baseline Predictions...")
    base_preds = []
    for text in tqdm(dev_df['text'].tolist()):
        if not text.strip():
            base_preds.append(0)
            continue
        result = baseline_classifier(text[:1000], truncation=True, max_length=512)
        label_str = result[0]['label']
        base_preds.append(1 if label_str == 'LABEL_1' else 0)
        
    dev_df['base_pred'] = base_preds
    
    # 4. Bucketing the Results & Calculating Statistics
    def print_bucket_stats(bucket_df, bucket_name):
        total = len(bucket_df)
        true_pcl = len(bucket_df[bucket_df['binary_label'] == 1])
        true_not_pcl = len(bucket_df[bucket_df['binary_label'] == 0])
        
        print("\n=======================================================")
        print(f"{bucket_name} - Total: {total}")
        print(f"  -> True Label is PCL (1):     {true_pcl}")
        print(f"  -> True Label is NOT PCL (0): {true_not_pcl}")
        print("=======================================================")
        return true_pcl, true_not_pcl

    # Bucket 1: Both correct
    b1 = dev_df[(dev_df['binary_label'] == dev_df['my_pred']) & (dev_df['binary_label'] == dev_df['base_pred'])]
    print_bucket_stats(b1, "BUCKET 1: BOTH MODELS CORRECT")
    print("Examples:")
    for i in range(min(6, len(b1))):
        print(f"-> [Label: {b1.iloc[i]['binary_label']}] {b1.iloc[i]['text']}\n")

    # Bucket 2: Both incorrect
    b2 = dev_df[(dev_df['binary_label'] != dev_df['my_pred']) & (dev_df['binary_label'] != dev_df['base_pred'])]
    print_bucket_stats(b2, "BUCKET 2: BOTH MODELS WRONG")
    print("Examples:")
    for i in range(min(6, len(b2))):
        print(f"-> [Label: {b2.iloc[i]['binary_label']}] {b2.iloc[i]['text']}\n")

    # Bucket 3: Mine right, Base wrong (Hero Cases)
    b3 = dev_df[(dev_df['binary_label'] == dev_df['my_pred']) & (dev_df['binary_label'] != dev_df['base_pred'])]
    print_bucket_stats(b3, "BUCKET 3: ENSEMBLE CORRECT, BASELINE WRONG (Hero Cases)")
    print("Examples:")
    for i in range(min(6, len(b3))):
        row = b3.iloc[i]
        print(f"-> [True: {row['binary_label']} | MyPred: {row['my_pred']} | BasePred: {row['base_pred']}] {row['text']}\n")

    # Bucket 4: Mine wrong, Base right (Trade-offs)
    b4 = dev_df[(dev_df['binary_label'] != dev_df['my_pred']) & (dev_df['binary_label'] == dev_df['base_pred'])]
    print_bucket_stats(b4, "BUCKET 4: BASELINE CORRECT, ENSEMBLE WRONG (Trade-offs)")
    
    b4_fn = b4[b4['binary_label'] == 1].copy()
    b4_fp = b4[b4['binary_label'] == 0].copy()

    print("--- ENSEMBLE FALSE NEGATIVES (True PCL Missed) ---")
    if 'category' in b4_fn.columns and not b4_fn['category'].isnull().all():
        # Explode unique categories per paragraph to see how many paragraphs contain each trope
        unique_cats_per_par = b4_fn['category'].dropna().apply(lambda x: list(set(x)))
        cat_counts = unique_cats_per_par.explode().value_counts()
        
        print(f"\n[Taxonomy Breakdown for these {len(b4_fn)} Missed Samples (Count of Paragraphs containing the trope)]")
        for cat, count in cat_counts.items():
            print(f"   - {cat}: {count}")
            
    print("\nExamples:")
    for i in range(min(5, len(b4_fn))):
        row = b4_fn.iloc[i]
        if isinstance(row['category'], list):
            counts = Counter(row['category'])
            cats_str = ", ".join([f"{k} (x{v})" if v > 1 else k for k, v in counts.items()])
            cats_formatted = f" [{cats_str}]"
        else:
            cats_formatted = ""
        print(f"-> [True: {row['binary_label']} | MyPred: {row['my_pred']} | BasePred: {row['base_pred']}]{cats_formatted} {row['text']}\n")

    print("--- ENSEMBLE FALSE POSITIVES (Objective Text Flagged) ---")
    print("\nExamples:")
    for i in range(min(5, len(b4_fp))):
        row = b4_fp.iloc[i]
        print(f"-> [True: {row['binary_label']} | MyPred: {row['my_pred']} | BasePred: {row['base_pred']}] {row['text']}\n")

if __name__ == "__main__":
    run_error_analysis()