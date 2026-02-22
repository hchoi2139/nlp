import os
import sys
import json
import torch
import torch.nn as nn
import numpy as np
import optuna
from tqdm import tqdm
from sklearn.metrics import f1_score
from transformers import get_cosine_schedule_with_warmup

# Ensure Python can find the src package
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.data_loader import get_dataloader
from src.model import PCLModelWithLAN

# --- SEPARATE DIRECTORY FOR OPTUNA ---
OPTUNA_DIR = '/vol/bitbucket/hc1721/nlp_scratch/optuna_study'
os.makedirs(OPTUNA_DIR, exist_ok=True)

def compute_kl_loss(logits1, logits2):
    p = torch.sigmoid(logits1)
    q = torch.sigmoid(logits2)
    eps = 1e-8
    kl_p_q = p * torch.log(p / (q + eps) + eps) + (1 - p) * torch.log((1 - p) / (1 - q + eps) + eps)
    kl_q_p = q * torch.log(q / (p + eps) + eps) + (1 - q) * torch.log((1 - q) / (1 - p + eps) + eps)
    return torch.mean(kl_p_q + kl_q_p) / 2

def get_optimizer_grouped_parameters(model, base_lr, weight_decay):
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

    # LLRD is maintained, scaled by the Optuna-selected base_lr
    return [
        {"params": zone1_params, "weight_decay": weight_decay, "lr": base_lr * 0.1},
        {"params": zone2_params, "weight_decay": weight_decay, "lr": base_lr},
        {"params": zone3_params, "weight_decay": weight_decay, "lr": base_lr * 5.0},
        {"params": zone4_params, "weight_decay": 0.0, "lr": base_lr},
    ]

def objective(trial):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Suggest Hyperparameters
    lr = trial.suggest_float("lr", 1e-6, 1e-4, log=True)
    weight_decay = trial.suggest_float("weight_decay", 0.0, 0.1)
    warmup_ratio = trial.suggest_float("warmup_ratio", 0.0, 0.2)
    kl_alpha = trial.suggest_float("kl_alpha", 0.5, 3.0)
    mtl_weight = trial.suggest_float("mtl_weight", 0.1, 1.0)
    
    # 2. Load Data (Official Split)
    data_path = 'data/dontpatronizeme_pcl.tsv'
    cat_path = 'data/dontpatronizeme_categories.tsv'
    train_loader, val_loader, _ = get_dataloader(data_path, cat_path, batch_size=16)
    
    # 3. Initialize Model & Optimizer
    model = PCLModelWithLAN().float().to(device)
    grouped_params = get_optimizer_grouped_parameters(model, base_lr=lr, weight_decay=weight_decay)
    optimizer = torch.optim.AdamW(grouped_params)
    
    # 4. Learning Rate Scheduler (Cosine Annealing with Warmup)
    epochs = 4 # We allow slightly longer training since Pruning will kill bad runs early
    total_steps = len(train_loader) * epochs
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    
    criterion_binary = nn.BCEWithLogitsLoss()
    criterion_taxonomy = nn.BCEWithLogitsLoss()
    
    best_val_f1 = 0.0

    # 5. Training Loop
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
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
            
            # Apply Optuna-suggested weights
            loss = loss_binary + (kl_alpha * loss_kl) + (mtl_weight * loss_tax)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            scheduler.step()

        # 6. Validation Phase
        model.eval()
        all_logits, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                logits, _ = model(batch['input_ids'].to(device), batch['attention_mask'].to(device))
                all_logits.extend(logits.cpu().numpy())
                all_labels.extend(batch['labels'].numpy())
                
        all_logits, all_labels = np.array(all_logits), np.array(all_labels)
        
        # Fast local threshold sweep to evaluate this trial
        epoch_best_f1 = 0.0
        for thresh in np.arange(-1.0, 3.0, 0.2):
            preds = (all_logits > thresh).astype(int)
            f1 = f1_score(all_labels, preds, zero_division=0)
            if f1 > epoch_best_f1:
                epoch_best_f1 = f1

        if epoch_best_f1 > best_val_f1:
            best_val_f1 = epoch_best_f1

        # 7. OPTUNA PRUNING (The Time Saver)
        # Report the intermediate F1 score to Optuna
        trial.report(best_val_f1, epoch)
        # If this F1 score is worse than the median of previous runs at this epoch, kill the trial
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return best_val_f1

if __name__ == "__main__":
    print("--- STARTING OPTUNA HYPERPARAMETER SWEEP ---")
    
    # Initialize a study that maximizes F1, using the Median Pruner
    study = optuna.create_study(
        direction="maximize", 
        study_name="pcl_deberta_tuning",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0)
    )
    
    # Run 30 trials
    study.optimize(objective, n_trials=30)
    
    print("\n" + "="*50)
    print("🏆 OPTIMIZATION COMPLETE")
    print("="*50)
    print("Best Trial:")
    trial = study.best_trial
    print(f"  F1 Score: {trial.value:.4f}")
    print("  Best Hyperparameters:")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
        
    # Save the golden recipe to the isolated directory
    with open(os.path.join(OPTUNA_DIR, 'best_hyperparameters.json'), 'w') as f:
        json.dump(trial.params, f, indent=4)
        
    print(f"\n📁 Golden recipe saved to: {os.path.join(OPTUNA_DIR, 'best_hyperparameters.json')}")