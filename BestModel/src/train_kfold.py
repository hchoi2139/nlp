import os
import sys
import json
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset
from transformers import AutoTokenizer, DataCollatorWithPadding, get_cosine_schedule_with_warmup

# Ensure Python can find the src package
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.model import PCLModelWithLAN
from src.data_loader import get_dataloader  # We use your bulletproof data loader

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
    print("--- INITIALIZING STRICTLY ISOLATED 5-FOLD PIPELINE ---")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # --- DYNAMIC OPTUNA LOADING ---
    optuna_path = '/vol/bitbucket/hc1721/nlp_scratch/optuna_study/best_hyperparameters.json'
    with open(optuna_path, 'r') as f:
        best_params = json.load(f)
        
    BEST_LR = best_params['lr']
    BEST_WD = best_params['weight_decay']
    BEST_WARMUP = best_params['warmup_ratio']
    BEST_KL = best_params['kl_alpha']
    BEST_MTL = best_params['mtl_weight']
    
    # --- 1. STRICT ISOLATION DATA LOADING ---
    data_path = 'data/dontpatronizeme_pcl.tsv'
    cat_path = 'data/dontpatronizeme_categories.tsv'
    
    # Your dataloader perfectly isolates the Train and Dev splits
    train_loader_base, _, _ = get_dataloader(data_path, cat_path, batch_size=16)
    
    # We extract STRICTLY the official 8,329 training dataset
    train_dataset_strictly_isolated = train_loader_base.dataset
    
    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    print("Extracting labels from the isolated training set for K-Fold stratification...")
    labels = []
    for i in tqdm(range(len(train_dataset_strictly_isolated)), desc="Parsing Labels"):
        val = train_dataset_strictly_isolated[i]['labels']
        labels.append(val.item() if isinstance(val, torch.Tensor) else int(val))
    labels = np.array(labels, dtype=int)
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    checkpoint_dir = '/vol/bitbucket/hc1721/nlp_scratch/checkpoints/kfold'
    os.makedirs(checkpoint_dir, exist_ok=True)
    fold_metrics = {}

    # --- 3. K-FOLD LOOP ---
    # We split strictly the labels array of the 8,329 Train set
    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
        print(f"\n========================================")
        print(f"               FOLD {fold + 1}/5")
        print(f"========================================")
        
        # Subsets created strictly from the isolated training data
        train_sub = Subset(train_dataset_strictly_isolated, train_idx)
        val_sub = Subset(train_dataset_strictly_isolated, val_idx)
        
        # Strategic Over-Sampling
        train_labels = labels[train_idx]
        class_counts = np.bincount(train_labels)
        class_weights = 1.0 / class_counts
        sample_weights = [class_weights[label] for label in train_labels]
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
        
        # Inject the standard collator
        train_loader = DataLoader(train_sub, batch_size=16, sampler=sampler, collate_fn=data_collator)
        val_loader = DataLoader(val_sub, batch_size=16, shuffle=False, collate_fn=data_collator)
        
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
                b_labels = batch['labels'].to(device).float()
                tax_labels = batch['taxonomy_labels'].to(device).float()
                
                optimizer.zero_grad()
                
                b_logits1, t_logits1 = model(input_ids, attention_mask)
                b_logits2, t_logits2 = model(input_ids, attention_mask)
                
                loss_binary = 0.5 * (criterion_binary(b_logits1, b_labels) + criterion_binary(b_logits2, b_labels))
                loss_kl = compute_kl_loss(b_logits1, b_logits2)
                loss_tax = 0.5 * (criterion_taxonomy(t_logits1, tax_labels) + criterion_taxonomy(t_logits2, tax_labels))
                
                loss = loss_binary + (BEST_KL * loss_kl) + (BEST_MTL * loss_tax)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                optimizer.step()
                scheduler.step()
                loop.set_postfix(loss=loss.item())
        
            # Validation
            model.eval()
            all_logits, all_labels = [], []
            with torch.no_grad():
                for batch in val_loader:
                    input_ids, attention_mask = batch['input_ids'].to(device), batch['attention_mask'].to(device)
                    b_labels = batch['labels'].numpy()
                    logits, _ = model(input_ids, attention_mask)
                    all_logits.extend(logits.cpu().numpy())
                    all_labels.extend(b_labels)
                    
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
        fold_metrics[f"Fold_{fold+1}"] = {"Threshold": float(best_thresh), "F1": float(best_f1)}

    with open(os.path.join(checkpoint_dir, 'fold_thresholds.json'), 'w') as f:
        json.dump(fold_metrics, f, indent=4)

if __name__ == "__main__":
    train_kfold()