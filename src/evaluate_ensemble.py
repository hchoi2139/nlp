import os
import sys
import json
import csv
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, DataCollatorWithPadding
from sklearn.metrics import f1_score, classification_report

# Ensure Python can find the src package
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.model import PCLModelWithLAN
from src.data_loader import clean_text

class UnlabeledPCLDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=256):
        self.encodings = tokenizer(
            df['text'].tolist(),
            truncation=True,
            padding=False,
            max_length=max_length
        )
    def __getitem__(self, idx):
        return {
            'input_ids': self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx]
        }
    def __len__(self):
        return len(self.encodings['input_ids'])

def evaluate_ensemble():
    print("--- INITIALIZING ENSEMBLE EVALUATION ---")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # 1. Load the exact 2094 true labels and text
    print("Loading official dev set text and labels...")
    data_path = os.path.join(project_root, 'data', 'dontpatronizeme_pcl.tsv')
    
    # ADDED quoting=csv.QUOTE_NONE to prevent Pandas from merging lines with stray quotes
    raw_df = pd.read_csv(data_path, sep='\t', skiprows=4, header=None, names=['par_id', 'art_id', 'keyword', 'country', 'text', 'label'], quoting=csv.QUOTE_NONE)
    raw_df['par_id'] = raw_df['par_id'].astype(str)
    raw_df['binary_label'] = (raw_df['label'] >= 2).astype(int)
    
    dev_split_path = os.path.join(project_root, 'data', 'practice-splits', 'dev_semeval_parids-labels.csv')
    dev_split_df = pd.read_csv(dev_split_path)
    dev_split_df['par_id'] = dev_split_df['par_id'].astype(str)
    
    dev_df = pd.merge(dev_split_df[['par_id']], raw_df[['par_id', 'text', 'binary_label']], on='par_id', how='left')
    
    # Sanity check
    if dev_df['binary_label'].isna().sum() > 0:
        print(f"WARNING: {dev_df['binary_label'].isna().sum()} labels are missing! Check raw data loading.")
        
    dev_df['text'] = dev_df['text'].apply(clean_text)
    true_labels = dev_df['binary_label'].fillna(0).astype(int).tolist()
    
    dev_dataset = UnlabeledPCLDataset(dev_df, tokenizer)
    dev_loader = DataLoader(dev_dataset, batch_size=32, shuffle=False, collate_fn=data_collator)
    
    # 2. Load Thresholds (ABSOLUTE PATH TO SCRATCH SPACE)
    #thresh_path = '/vol/bitbucket/hc1721/nlp_scratch/checkpoints/kfold/fold_thresholds.json'
    thresh_path = os.path.join(project_root, 'BestModel', 'fold_thresholds.json')
    with open(thresh_path, 'r') as f:
        thresholds = json.load(f)
        
    all_fold_preds = []  
    all_fold_probs = []  
    
    model = PCLModelWithLAN().float().to(device)
    
    # 3. Iterate through all 5 models sequentially
    for fold in range(1, 6):
        # ABSOLUTE PATH TO SCRATCH SPACE
        #model_path = f'/vol/bitbucket/hc1721/nlp_scratch/checkpoints/kfold/model_fold_{fold}.pth'
        model_path = os.path.join(project_root, 'BestModel', f'model_fold_{fold}.pth')
        thresh = thresholds[f'Fold_{fold}']['Threshold']
        print(f"\nEvaluating Fold {fold} (Optimal Threshold: {thresh:.2f})...")
        
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        
        fold_preds = []
        fold_probs = []
        with torch.no_grad():
            for batch in tqdm(dev_loader, desc=f"Fold {fold} Inference"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                logits, _ = model(input_ids, attention_mask)
                
                # FIXED: flatten() instead of squeeze()
                logits = logits.flatten().cpu().numpy() 
                
                probs = 1 / (1 + np.exp(-logits))
                preds = (logits > thresh).astype(int)
                
                fold_probs.extend(probs.tolist())
                fold_preds.extend(preds.tolist())
                
        all_fold_probs.append(fold_probs)
        all_fold_preds.append(fold_preds)

    # 4. Ensembling Logic
    all_fold_preds = np.array(all_fold_preds) 
    all_fold_probs = np.array(all_fold_probs) 
    
    # Method A: Hard Voting
    sum_preds = np.sum(all_fold_preds, axis=0)
    hard_ensemble_preds = (sum_preds >= 3).astype(int)
    hard_f1 = f1_score(true_labels, hard_ensemble_preds, pos_label=1, average='binary')
    
    # Method B: Soft Voting
    avg_probs = np.mean(all_fold_probs, axis=0)
    soft_ensemble_preds = (avg_probs > 0.5).astype(int)
    soft_f1 = f1_score(true_labels, soft_ensemble_preds, pos_label=1, average='binary')
    
    print("\n========================================")
    print("        ENSEMBLE RESULTS (OFFICIAL DEV)   ")
    print("========================================")
    print(f"Hard Voting (Majority) F1:     {hard_f1:.4f}")
    print(f"Soft Voting (Avg Probs) F1:    {soft_f1:.4f}")
    print("========================================\n")
    
    print("Detailed Hard Voting Classification Report:")
    print(classification_report(true_labels, hard_ensemble_preds, digits=4))

if __name__ == "__main__":
    evaluate_ensemble()