import os
import sys
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score, classification_report

# Ensure Python can find the src package
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.data_loader import get_dataloader
from src.model import PCLModelWithLAN

def evaluate_ensemble():
    print("--- INITIALIZING ENSEMBLE EVALUATION ON OFFICIAL DEV SPLIT ---")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint_dir = '/vol/bitbucket/hc1721/nlp_scratch/checkpoints/kfold'
    
    # 1. Load ONLY the Official Dev Split
    data_path = 'data/dontpatronizeme_pcl.tsv'
    cat_path = 'data/dontpatronizeme_categories.tsv'
    _, val_loader, _ = get_dataloader(data_path, cat_path, batch_size=16)

    # We need to extract the raw labels from the val_loader for sklearn metrics
    all_labels = []
    for batch in val_loader:
        all_labels.extend(batch['labels'].numpy())
    labels = np.array(all_labels)

    # 2. Collect Predictions from all 5 Models
    all_fold_logits = [] 
    
    for fold in range(1, 6):
        print(f"\n--- Loading Model: Fold {fold} ---")
        model = PCLModelWithLAN().float().to(device)
        model.load_state_dict(torch.load(os.path.join(checkpoint_dir, f'model_fold_{fold}.pth')))
        model.eval()
        
        fold_logits = []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Inference Fold {fold}"):
                logits, _ = model(batch['input_ids'].to(device), batch['attention_mask'].to(device))
                fold_logits.extend(logits.cpu().numpy())
        all_fold_logits.append(np.array(fold_logits))

    # 3. Ensemble Average (The "Wisdom of the Crowd")
    ensemble_logits = np.mean(all_fold_logits, axis=0)

    # 4. Global Threshold Sweep
    best_f1, best_thresh = 0.0, 0.0
    # Sweep from -2.0 to 4.0 in 0.1 increments
    for thresh in np.arange(-2.0, 4.0, 0.1):
        preds = (ensemble_logits > thresh).astype(int)
        f1 = f1_score(labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_thresh = f1, thresh

    print(f"\n" + "="*50)
    print(f"🏆 FINAL ENSEMBLE RESULT (OFFICIAL DEV SPLIT)")
    print(f"="*50)
    print(f"Optimal Ensemble Threshold : {best_thresh:.2f}")
    print(f"Final Combined F1-Score    : {best_f1:.4f}")
    print("\nDetailed Classification Report:")
    print(classification_report(labels, (ensemble_logits > best_thresh).astype(int), target_names=['Not Patronizing', 'Patronizing']))

if __name__ == "__main__":
    evaluate_ensemble()