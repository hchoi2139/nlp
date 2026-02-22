import os
import sys
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, DataCollatorWithPadding

# Ensure Python can find the src package
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.model import PCLModelWithLAN
from src.data_loader import get_dataloader

# --- HARDCODED WINNING PARAMETERS ---
BEST_FOLD_PATH = '/vol/bitbucket/hc1721/nlp_scratch/checkpoints/kfold/model_fold_2.pth'
BEST_THRESHOLD = 1.40

class UnlabeledPCLDataset(Dataset):
    """A lightweight dataset class for processing the unlabelled test TSV"""
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

def generate_predictions():
    print("--- INITIALIZING INFERENCE ENGINE ---")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # 1. Load the Best Model
    print(f"Loading weights from: {BEST_FOLD_PATH}")
    model = PCLModelWithLAN().float().to(device)
    model.load_state_dict(torch.load(BEST_FOLD_PATH, map_location=device))
    model.eval()

    # 2. Prepare the Dev Data (Using your exact isolation logic)
    print("\nProcessing Official Dev Set...")
    data_path = 'data/dontpatronizeme_pcl.tsv'
    cat_path = 'data/dontpatronizeme_categories.tsv'
    _, val_loader_base, _ = get_dataloader(data_path, cat_path, batch_size=32)
    dev_dataset = val_loader_base.dataset
    
    # Use standard collator for the dev set inference
    dev_loader = DataLoader(dev_dataset, batch_size=32, shuffle=False, collate_fn=data_collator)
    
    dev_preds = []
    with torch.no_grad():
        for batch in tqdm(dev_loader, desc="Predicting Dev"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            logits, _ = model(input_ids, attention_mask)
            
            # Apply our golden threshold
            preds = (logits.cpu().numpy() > BEST_THRESHOLD).astype(int)
            dev_preds.extend(preds)

    # 3. Prepare the Official Test Data (task4_test.tsv)
    print("\nProcessing Official Test Set...")
    test_path = 'data/task4_test.tsv'
    if not os.path.exists(test_path):
        print(f"⚠️ WARNING: Could not find {test_path}. Make sure it is in your data folder!")
    else:
        # Load the test set (assuming standard SemEval format without labels)
        test_df = pd.read_csv(test_path, sep='\t', names=['par_id', 'art_id', 'keyword', 'country', 'text'], skiprows=1)
        test_df['text'] = test_df['text'].astype(str)
        
        test_dataset = UnlabeledPCLDataset(test_df, tokenizer)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=data_collator)
        
        test_preds = []
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Predicting Test"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                logits, _ = model(input_ids, attention_mask)
                preds = (logits.cpu().numpy() > BEST_THRESHOLD).astype(int)
                test_preds.extend(preds)

    # 4. Write to Text Files
    output_dir = 'BestModel'
    os.makedirs(output_dir, exist_ok=True)
    
    dev_out = os.path.join(output_dir, 'dev.txt')
    with open(dev_out, 'w') as f:
        for p in dev_preds:
            f.write(f"{p}\n")
    print(f"✅ Saved {len(dev_preds)} predictions to {dev_out}")
            
    if os.path.exists(test_path):
        test_out = os.path.join(output_dir, 'test.txt')
        with open(test_out, 'w') as f:
            for p in test_preds:
                f.write(f"{p}\n")
        print(f"✅ Saved {len(test_preds)} predictions to {test_out}")

if __name__ == "__main__":
    generate_predictions()