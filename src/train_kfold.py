import os
import sys
import json
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import AutoTokenizer, DataCollatorWithPadding, get_cosine_schedule_with_warmup

# Ensure Python can find the src package
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import your custom architecture and dataset class
from src.model import PCLModelWithLAN
from src.data_loader import PCLDataset 

class EarlyStopping:
    def __init__(self, patience=3, delta=0.01):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_f1):
        if self.best_score is None:
            self.best_score = val_f1
        elif val_f1 < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_f1
            self.counter = 0

def compute_kl_loss(logits1, logits2):
    p = torch.sigmoid(logits1)
    q = torch.sigmoid(logits2)
    eps = 1e-8
    kl_p_q = p * torch.log(p / (q + eps) + eps) + (1 - p) * torch.log((1 - p) / (1 - q + eps) + eps)
    kl_q_p = q * torch.log(q / (p + eps) + eps) + (1 - q) * torch.log((1 - q) / (1 - p + eps) + eps)
    return torch.mean(kl_p_q + kl_q_p) / 2

def get_optimizer_grouped_parameters(model, base_lr=1e-5, weight_decay=0.01):
    no_decay = ["bias", "LayerNorm.weight"]
    zone1_params, zone2_params, zone3_params, zone4_params = [], [], [], []
    
    for n, p in model.named_parameters():
        if any(nd in n for nd in no_decay):
            zone4_params.append(p)
        elif "head" in n or "label_attention" in n or "label_embeddings" in n:
            zone3_params.append(p)
        elif "embeddings" in n or any(f"layer.{i}." in n for i in range(4)):
            zone1_params.append(p)
        else:
            zone2_params.append(p)

    return [
        {"params": zone1_params, "weight_decay": weight_decay, "lr": base_lr * 0.1},
        {"params": zone2_params, "weight_decay": weight_decay, "lr": base_lr},
        {"params": zone3_params, "weight_decay": weight_decay, "lr": base_lr * 5},
        {"params": zone4_params, "weight_decay": 0.0, "lr": base_lr},
    ]

def train_kfold():
    print("--- INITIALIZING 5-FOLD PIPELINE WITH OPTUNA RECIPE ---")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")
    
    # --- DYNAMIC OPTUNA LOADING ---
    optuna_path = '/vol/bitbucket/hc1721/nlp_scratch/optuna_study/best_hyperparameters.json'
    with open(optuna_path, 'r') as f:
        best_params = json.load(f)
        
    BEST_LR = best_params['lr']
    BEST_WD = best_params['weight_decay']
    BEST_WARMUP = best_params['warmup_ratio']
    BEST_KL = best_params['kl_alpha']
    BEST_MTL = best_params['mtl_weight']
    print(f"Loaded Hyperparameters: LR={BEST_LR:.2e}, WD={BEST_WD:.4f}, KL={BEST_KL:.2f}")

    # --- 1. PREPARE FULL DATASET ---
    df = pd.read_csv('data/dontpatronizeme_pcl.tsv', sep='\t', names=['par_id', 'art_id', 'keyword', 'country', 'text', 'label'], skiprows=4)
    df['label'] = df['label'].apply(lambda x: 1 if int(x) > 1 else 0)

    df = df.dropna(subset=['text'])
    df['text'] = df['text'].astype(str)
    
    # Using your exact placeholder logic that bypasses the KeyError
    if 'taxonomy_labels' not in df.columns:
         df['taxonomy_labels'] = [np.zeros(7) for _ in range(len(df))]

    # --- 2. TOKENIZER & COLLATOR SETUP ---
    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    checkpoint_dir = '/vol/bitbucket/hc1721/nlp_scratch/checkpoints/kfold'
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    fold_metrics = {}

    # --- 3. K-FOLD LOOP ---
    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df['label'])):
        print(f"\n========================================")
        print(f"               FOLD {fold + 1}/5")
        print(f"========================================")
        
        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)
        
        # Strategic Over-Sampling
        class_counts = train_df['label'].value_counts().sort_index().values
        class_weights = 1.0 / class_counts
        sample_weights = [class_weights[label] for label in train_df['label']]
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
        
        train_dataset = PCLDataset(train_df, tokenizer, max_length=256)
        val_dataset = PCLDataset(val_df, tokenizer, max_length=256)
        
        # Inject the DataCollator here
        train_loader = DataLoader(train_dataset, batch_size=16, sampler=sampler, collate_fn=data_collator)
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=data_collator)
        
        # Initialize Model & Optimizer
        model = PCLModelWithLAN().float().to(device)
        grouped_params = get_optimizer_grouped_parameters(model, base_lr=BEST_LR, weight_decay=BEST_WD)
        optimizer = torch.optim.AdamW(grouped_params)
        
        criterion_binary = nn.BCEWithLogitsLoss()
        criterion_taxonomy = nn.BCEWithLogitsLoss()
        
        epochs = 10
        total_steps = len(train_loader) * epochs
        warmup_steps = int(total_steps * BEST_WARMUP)
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
        
        early_stopping = EarlyStopping(patience=3, delta=0.01)
        best_f1, best_thresh = 0.0, 0.0
        
        for epoch in range(epochs):
            model.train()
            loop = tqdm(train_loader, desc=f"Fold {fold+1} - Epoch {epoch+1}")
            for batch in loop:
                input_ids, attention_mask = batch['input_ids'].to(device), batch['attention_mask'].to(device)
                labels = batch['labels'].to(device).float()
                tax_labels = batch['taxonomy_labels'].to(device).float()
                
                optimizer.zero_grad()
                
                # R-Drop Dual Pass
                b_logits1, t_logits1 = model(input_ids, attention_mask)
                b_logits2, t_logits2 = model(input_ids, attention_mask)
                
                loss_binary = 0.5 * (criterion_binary(b_logits1, labels) + criterion_binary(b_logits2, labels))
                loss_kl = compute_kl_loss(b_logits1, b_logits2)
                loss_tax = 0.5 * (criterion_taxonomy(t_logits1, tax_labels) + criterion_taxonomy(t_logits2, tax_labels))
                
                loss = loss_binary + (BEST_KL * loss_kl) + (BEST_MTL * loss_tax)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                optimizer.step()
                scheduler.step()
                loop.set_postfix(loss=loss.item())
        
            # Validation Threshold Sweep
            model.eval()
            all_logits, all_labels = [], []
            with torch.no_grad():
                for batch in val_loader:
                    input_ids, attention_mask = batch['input_ids'].to(device), batch['attention_mask'].to(device)
                    labels = batch['labels'].numpy()
                    logits, _ = model(input_ids, attention_mask)
                    all_logits.extend(logits.cpu().numpy())
                    all_labels.extend(labels)
                    
            all_logits, all_labels = np.array(all_logits), np.array(all_labels)
            epoch_best_f1, epoch_best_thresh = 0.0, 0.0
            
            for thresh in np.arange(-2.0, 4.0, 0.1):
                preds = (all_logits > thresh).astype(int)
                f1 = f1_score(all_labels, preds, zero_division=0)
                if f1 > epoch_best_f1:
                    epoch_best_f1, epoch_best_thresh = f1, thresh
                    
            print(f"Validation F1: {epoch_best_f1:.4f} (Thresh: {epoch_best_thresh:.2f})")
            
            if epoch_best_f1 > best_f1:
                best_f1, best_thresh = epoch_best_f1, epoch_best_thresh
                torch.save(model.state_dict(), os.path.join(checkpoint_dir, f'model_fold_{fold+1}.pth'))
                
            early_stopping(epoch_best_f1)
            if early_stopping.early_stop:
                print(f"Early stopping triggered at Epoch {epoch+1}!")
                break
                
        print(f"⭐ Fold {fold+1} Complete | Optimal Threshold: {best_thresh:.2f} | Validation F1: {best_f1:.4f}")
        
        # Save exact threshold to dictionary
        fold_metrics[f"Fold_{fold+1}"] = {"Threshold": float(best_thresh), "F1": float(best_f1)}

    # --- 4. EXPORT JSON & FINAL REPORT ---
    with open(os.path.join(checkpoint_dir, 'fold_thresholds.json'), 'w') as f:
        json.dump(fold_metrics, f, indent=4)

    print("\n" + "="*50)
    print("           K-FOLD ENSEMBLE COMPLETE")
    print("="*50)
    avg_f1 = np.mean([metrics["F1"] for metrics in fold_metrics.values()])
    for fold, metrics in fold_metrics.items():
        print(f"{fold}: Threshold = {metrics['Threshold']:.2f}, F1 = {metrics['F1']:.4f}")
    print(f"\n🏆 Average Validation F1: {avg_f1:.4f}")
    print(f"📁 Thresholds saved to: {os.path.join(checkpoint_dir, 'fold_thresholds.json')}")

if __name__ == "__main__":
    train_kfold()