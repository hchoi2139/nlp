import os
import pandas as pd
from sklearn.metrics import f1_score

def verify():
    print("--- VERIFYING STRICT DEV.TXT PREDICTIONS ---")
    
    # 1. Load the raw TSV and binarize the labels (0 or 1)
    print("Loading raw TSV for true labels...")
    raw_df = pd.read_csv('data/dontpatronizeme_pcl.tsv', sep='\t', skiprows=4, header=None, names=['par_id', 'art_id', 'keyword', 'country', 'text', 'label'])
    raw_df['par_id'] = raw_df['par_id'].astype(str)
    raw_df['binary_label'] = (raw_df['label'] >= 2).astype(int)
    
    # 2. Load the official dev split to get the exact 2094 ordered IDs
    print("Aligning to official 2094-line dev split...")
    dev_split_df = pd.read_csv('data/practice-splits/dev_semeval_parids-labels.csv')
    dev_split_df['par_id'] = dev_split_df['par_id'].astype(str)
    
    # 3. Merge to get the perfectly ordered true labels
    dev_df = pd.merge(dev_split_df[['par_id']], raw_df[['par_id', 'binary_label']], on='par_id', how='left')
    true_labels = dev_df['binary_label'].fillna(0).astype(int).tolist()
    
    # 4. Load your predictions
    dev_txt_path = 'BestModel/dev.txt'
    if not os.path.exists(dev_txt_path):
        print(f"❌ ERROR: {dev_txt_path} not found!")
        return
        
    with open(dev_txt_path, 'r') as f:
        preds = [int(line.strip()) for line in f.readlines() if line.strip() != '']
        
    # 5. Verify lengths match
    if len(preds) != len(true_labels):
        print(f"❌ ERROR: Length mismatch! True labels: {len(true_labels)}, Predictions: {len(preds)}")
        return
        
    # 6. Calculate True Global F1 Score
    f1 = f1_score(true_labels, preds, zero_division=0)
    print(f"\n✅ Verification Complete!")
    print(f"Total Dev Examples: {len(preds)} (Matches Official Spec!)")
    print(f"True Global Dev F1 Score: {f1:.4f}")

if __name__ == "__main__":
    verify()