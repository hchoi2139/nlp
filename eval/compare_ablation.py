import os
import sys
import json
import csv
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from collections import Counter
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, DataCollatorWithPadding

# Ensure Python can find the src package
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.data_loader import clean_text

# ==========================================
# 1. STRIPPED-DOWN ABLATION MODEL DEF
# ==========================================
class PCLModelAblation(nn.Module):
    def __init__(self, model_name="microsoft/deberta-v3-base"):
        super().__init__()
        self.deberta = AutoModel.from_pretrained(model_name)
        hidden_size = self.deberta.config.hidden_size 
        self.binary_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.deberta(input_ids=input_ids, attention_mask=attention_mask)
        cls_token = outputs.last_hidden_state[:, 0, :]        
        binary_logits = self.binary_head(cls_token).squeeze(-1)
        return binary_logits

class UnlabeledPCLDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=256):
        self.encodings = tokenizer(df['text'].tolist(), truncation=True, padding=False, max_length=max_length)
    def __getitem__(self, idx):
        return {'input_ids': self.encodings['input_ids'][idx], 'attention_mask': self.encodings['attention_mask'][idx]}
    def __len__(self):
        return len(self.encodings['input_ids'])

def run_ablation_comparison():
    print("--- INITIALIZING ABLATION VS PROPOSED COMPARISON ---")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # 1. Load Dev Set Text, Labels, and Categories
    data_path = os.path.join(project_root, 'data', 'dontpatronizeme_pcl.tsv')
    raw_df = pd.read_csv(data_path, sep='\t', skiprows=4, header=None, names=['par_id', 'art_id', 'keyword', 'country', 'text', 'label'], quoting=csv.QUOTE_NONE)
    raw_df['par_id'] = raw_df['par_id'].astype(str)
    raw_df['binary_label'] = (raw_df['label'] >= 2).astype(int)
    
    cat_path = os.path.join(project_root, 'data', 'dontpatronizeme_categories.tsv')
    if os.path.exists(cat_path):
        cat_df = pd.read_csv(cat_path, sep='\t', skiprows=4, header=None, 
                             names=['par_id', 'art_id', 'text', 'keyword', 'country', 'span_start', 'span_end', 'span_text', 'category', 'num_annotators'], quoting=csv.QUOTE_NONE)
        cat_df['par_id'] = cat_df['par_id'].astype(str)
        par_cats = cat_df.groupby('par_id')['category'].apply(list).reset_index()
        raw_df = pd.merge(raw_df, par_cats, on='par_id', how='left')
    else:
        raw_df['category'] = None

    dev_split_path = os.path.join(project_root, 'data', 'practice-splits', 'dev_semeval_parids-labels.csv')
    dev_split_df = pd.read_csv(dev_split_path)
    dev_split_df['par_id'] = dev_split_df['par_id'].astype(str)
    
    dev_df = pd.merge(dev_split_df[['par_id']], raw_df[['par_id', 'text', 'binary_label', 'category']], on='par_id', how='left')
    dev_df['text'] = dev_df['text'].apply(clean_text)
    
    # 2. Load Proposed Model Predictions (LAN + MTL)
    my_preds_path = os.path.join(project_root, 'BestModel', 'dev.txt')
    with open(my_preds_path, 'r') as f:
        dev_df['proposed_pred'] = [int(line.strip()) for line in f.readlines()]
        
    # 3. Generate Ablated Model Predictions (No LAN, No MTL)
    print("Generating Ablation Ensemble Predictions...")
    ABLATION_DIR = '/vol/bitbucket/hc1721/nlp_scratch/ablation_study'
    with open(os.path.join(ABLATION_DIR, 'ablation_thresholds.json'), 'r') as f:
        thresholds = json.load(f)
        
    dev_dataset = UnlabeledPCLDataset(dev_df, tokenizer)
    dev_loader = DataLoader(dev_dataset, batch_size=32, shuffle=False, collate_fn=data_collator)
    
    all_fold_preds = []  
    model = PCLModelAblation().float().to(device)
    
    for fold in range(1, 6):
        model.load_state_dict(torch.load(os.path.join(ABLATION_DIR, f'model_fold_{fold}.pth'), map_location=device))
        model.eval()
        thresh = thresholds[f'Fold_{fold}']['Threshold']
        fold_preds = []
        with torch.no_grad():
            for batch in tqdm(dev_loader, desc=f"Ablation Fold {fold}"):
                logits = model(batch['input_ids'].to(device), batch['attention_mask'].to(device)).flatten().cpu().numpy()
                fold_preds.extend((logits > thresh).astype(int).tolist())
        all_fold_preds.append(fold_preds)

    dev_df['ablation_pred'] = (np.sum(np.array(all_fold_preds), axis=0) >= 3).astype(int)
    
    # 4. Bucketing the Results (Proposed vs Ablated)
    def print_bucket_stats(bucket_df, bucket_name):
        total = len(bucket_df)
        true_pcl = len(bucket_df[bucket_df['binary_label'] == 1])
        true_not_pcl = len(bucket_df[bucket_df['binary_label'] == 0])
        print("\n=======================================================")
        print(f"{bucket_name} - Total: {total}")
        print(f"  -> True Label is PCL (1):     {true_pcl}")
        print(f"  -> True Label is NOT PCL (0): {true_not_pcl}")
        print("=======================================================")
        return bucket_df

    print("\n--- MACRO COMPARISON: PROPOSED (LAN) vs ABLATED (Vanilla) ---")
    b1 = dev_df[(dev_df['proposed_pred'] == dev_df['binary_label']) & (dev_df['ablation_pred'] == dev_df['binary_label'])]
    print_bucket_stats(b1, "BUCKET 1: BOTH MODELS CORRECT")

    b2 = dev_df[(dev_df['proposed_pred'] != dev_df['binary_label']) & (dev_df['ablation_pred'] != dev_df['binary_label'])]
    print_bucket_stats(b2, "BUCKET 2: BOTH MODELS WRONG")

    b3 = dev_df[(dev_df['proposed_pred'] == dev_df['binary_label']) & (dev_df['ablation_pred'] != dev_df['binary_label'])]
    print_bucket_stats(b3, "BUCKET 3: PROPOSED CORRECT, ABLATION WRONG (Where LAN Added Value)")
    
    b3_fn = b3[b3['binary_label'] == 1].copy() # LAN found the PCL, Ablation missed it
    b3_fp = b3[b3['binary_label'] == 0].copy() # LAN ignored the objective text, Ablation falsely flagged it
    
    if not b3_fn.empty and 'category' in b3_fn.columns and not b3_fn['category'].isnull().all():
        print("\n[Taxonomy of True PCL that LAN successfully caught but Ablation missed]")
        cat_counts = b3_fn['category'].dropna().apply(lambda x: list(set(x))).explode().value_counts()
        for cat, count in cat_counts.items(): print(f"   - {cat}: {count}")

    print("\nExamples (LAN Correct [1], Ablation Missed [0]):")
    for i in range(min(3, len(b3_fn))):
        print(f"-> {b3_fn.iloc[i]['text']}\n")
        
    print("Examples (LAN Correctly Ignored [0], Ablation False Positive [1]):")
    for i in range(min(3, len(b3_fp))):
        print(f"-> {b3_fp.iloc[i]['text']}\n")

    b4 = dev_df[(dev_df['proposed_pred'] != dev_df['binary_label']) & (dev_df['ablation_pred'] == dev_df['binary_label'])]
    print_bucket_stats(b4, "BUCKET 4: ABLATION CORRECT, PROPOSED WRONG (Where LAN Hurt Performance)")
    
    b4_fn = b4[b4['binary_label'] == 1].copy() # Ablation found PCL, LAN missed it
    b4_fp = b4[b4['binary_label'] == 0].copy() # Ablation ignored it, LAN falsely flagged it

    if not b4_fn.empty and 'category' in b4_fn.columns and not b4_fn['category'].isnull().all():
        print("\n[Taxonomy of True PCL that Ablation caught but LAN missed]")
        cat_counts = b4_fn['category'].dropna().apply(lambda x: list(set(x))).explode().value_counts()
        for cat, count in cat_counts.items(): print(f"   - {cat}: {count}")

    print("\nExamples (LAN Missed [0], Ablation Correct [1]):")
    for i in range(min(3, len(b4_fn))):
        print(f"-> {b4_fn.iloc[i]['text']}\n")
        
    print("Examples (LAN False Positive [1], Ablation Correctly Ignored [0]):")
    for i in range(min(3, len(b4_fp))):
        print(f"-> {b4_fp.iloc[i]['text']}\n")

if __name__ == "__main__":
    run_ablation_comparison()