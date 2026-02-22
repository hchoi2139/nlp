import os
import sys
import json
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, DataCollatorWithPadding

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.model import PCLModelWithLAN
from src.data_loader import clean_text

class UnlabeledPCLDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=256):
        self.encodings = tokenizer(
            df['text'].tolist(), truncation=True, padding=False, max_length=max_length
        )
    def __getitem__(self, idx):
        return {
            'input_ids': self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx]
        }
    def __len__(self):
        return len(self.encodings['input_ids'])

def generate_ensemble_predictions():
    print("--- INITIALIZING HARD VOTING ENSEMBLE ENGINE ---")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # 1. Load Thresholds
    with open('/vol/bitbucket/hc1721/nlp_scratch/checkpoints/kfold/fold_thresholds.json', 'r') as f:
        thresholds = json.load(f)

    # 2. Prepare Official Dev Set
    raw_df = pd.read_csv('data/dontpatronizeme_pcl.tsv', sep='\t', skiprows=4, header=None, names=['par_id', 'art_id', 'keyword', 'country', 'text', 'label'])
    raw_df['par_id'] = raw_df['par_id'].astype(str)
    
    dev_split_df = pd.read_csv('data/practice-splits/dev_semeval_parids-labels.csv')
    dev_split_df['par_id'] = dev_split_df['par_id'].astype(str)
    dev_df = pd.merge(dev_split_df[['par_id']], raw_df[['par_id', 'text']], on='par_id', how='left')
    dev_df['text'] = dev_df['text'].apply(clean_text)
    
    dev_dataset = UnlabeledPCLDataset(dev_df, tokenizer)
    dev_loader = DataLoader(dev_dataset, batch_size=32, shuffle=False, collate_fn=data_collator)

    # 3. Prepare Official Test Set
    test_path = 'data/task4_test.tsv'
    test_df = pd.read_csv(test_path, sep='\t', names=['par_id', 'art_id', 'keyword', 'country', 'text'], skiprows=1)
    test_df['text'] = test_df['text'].astype(str).apply(clean_text)
    test_dataset = UnlabeledPCLDataset(test_df, tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=data_collator)

    model = PCLModelWithLAN().float().to(device)
    all_dev_preds = []
    all_test_preds = []

    # 4. Run Inference across all 5 Folds
    for fold in range(1, 6):
        model_path = f'/vol/bitbucket/hc1721/nlp_scratch/checkpoints/kfold/model_fold_{fold}.pth'
        thresh = thresholds[f'Fold_{fold}']['Threshold']
        print(f"Running Fold {fold} (Threshold: {thresh:.2f})...")
        
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        
        # Dev Inference
        fold_dev_preds = []
        with torch.no_grad():
            for batch in tqdm(dev_loader, desc=f"Dev Fold {fold}"):
                logits, _ = model(batch['input_ids'].to(device), batch['attention_mask'].to(device))
                preds = (logits.squeeze().cpu().numpy() > thresh).astype(int)
                fold_dev_preds.extend(preds.tolist())
        all_dev_preds.append(fold_dev_preds)
        
        # Test Inference
        fold_test_preds = []
        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"Test Fold {fold}"):
                logits, _ = model(batch['input_ids'].to(device), batch['attention_mask'].to(device))
                preds = (logits.squeeze().cpu().numpy() > thresh).astype(int)
                fold_test_preds.extend(preds.tolist())
        all_test_preds.append(fold_test_preds)

    # 5. Apply Hard Voting (Majority Rule >= 3)
    final_dev_preds = (np.sum(all_dev_preds, axis=0) >= 3).astype(int)
    final_test_preds = (np.sum(all_test_preds, axis=0) >= 3).astype(int)

    # 6. Export
    output_dir = 'BestModel'
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, 'dev.txt'), 'w') as f:
        for p in final_dev_preds: f.write(f"{p}\n")
    with open(os.path.join(output_dir, 'test.txt'), 'w') as f:
        for p in final_test_preds: f.write(f"{p}\n")
        
    print(f"\n✅ SUCCESS! Saved {len(final_dev_preds)} Dev and {len(final_test_preds)} Test predictions using Hard Voting.")

if __name__ == "__main__":
    generate_ensemble_predictions()