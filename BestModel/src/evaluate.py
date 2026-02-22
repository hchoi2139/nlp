import os
import sys
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report

# Ensure Python can find the src package
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.data_loader import get_dataloader
from src.model import PCLModelWithLAN

def evaluate_model():
    print("--- INITIALIZING EVALUATION ON OFFICIAL DEV SPLIT ---")
    
    # Use GPU if available, fallback to CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Evaluating on device: {device}")

    # Load Data 
    data_path = 'data/dontpatronizeme_pcl.tsv'
    cat_path = 'data/dontpatronizeme_categories.tsv'
    
    # Correctly unpack all three variables, isolating the val_loader
    _, val_loader, _ = get_dataloader(data_path, cat_path, batch_size=16)

    # Initialize the architecture and force FP32 (to match training)
    model = PCLModelWithLAN().float().to(device)
    
    # Locate the best model weights
    checkpoint_path = '/vol/bitbucket/hc1721/nlp_scratch/checkpoints/best_model.pth'
    if not os.path.exists(checkpoint_path):
        print(f"🚨 ERROR: No checkpoint found at {checkpoint_path}")
        return
        
    print(f"Loading weights from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    
    # Put model in evaluation mode (turns off dropout)
    model.eval()
    
    all_logits = []
    all_labels = []

    print("\n--- STARTING INFERENCE ---")
    with torch.no_grad(): 
        for batch in tqdm(val_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Keep labels on CPU for sklearn
            labels = batch['labels'].numpy() 
            
            # Forward pass
            binary_logits, _ = model(input_ids, attention_mask)
            
            # Store raw logits instead of hard 0/1 predictions
            all_logits.extend(binary_logits.cpu().numpy())
            all_labels.extend(labels)

    # --- THRESHOLD SWEEP ---
    print("\n" + "="*50)
    print("                 OPTIMIZED DEV RESULTS")
    print("="*50)
    
    all_logits = np.array(all_logits)
    all_labels = np.array(all_labels)
    
    best_f1 = 0.0
    best_thresh = 0.0
    
    # Test thresholds from -2.0 to 4.0 in steps of 0.1
    thresholds = np.arange(-2.0, 4.0, 0.1)
    for thresh in thresholds:
        preds = (all_logits > thresh).astype(int)
        f1 = f1_score(all_labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh

    print(f"Optimal Logit Threshold : {best_thresh:.2f}")
    
    # Apply the best threshold to get final metrics
    final_preds = (all_logits > best_thresh).astype(int)
    
    f1 = f1_score(all_labels, final_preds)
    precision = precision_score(all_labels, final_preds)
    recall = recall_score(all_labels, final_preds)
    
    print(f"Binary F1-Score : {f1:.4f}")
    print(f"Precision       : {precision:.4f}")
    print(f"Recall          : {recall:.4f}")
    print("\nDetailed Classification Report:")
    print(classification_report(all_labels, final_preds, target_names=['Not Patronizing', 'Patronizing']))

if __name__ == "__main__":
    evaluate_model()