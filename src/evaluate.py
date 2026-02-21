import os
import sys
import torch
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
    
    # Keep it on CPU to bypass Slurm GPU locks on the login node
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
        print("Did the training script finish successfully?")
        return
        
    print(f"Loading weights from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    
    # Put model in evaluation mode (turns off dropout)
    model.eval()
    
    all_preds = []
    all_labels = []

    print("\n--- STARTING INFERENCE ---")
    with torch.no_grad(): # No need to calculate gradients for evaluation
        for batch in tqdm(val_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Keep labels on CPU for sklearn
            labels = batch['labels'].numpy() 
            
            # Forward pass
            binary_logits, _ = model(input_ids, attention_mask)
            
            # Convert logits to binary predictions
            # Because we used BCEWithLogitsLoss, a logit > 0 means probability > 0.5
            preds = (binary_logits > 0).cpu().numpy().astype(int)
            
            all_preds.extend(preds)
            all_labels.extend(labels)

    # Calculate Final Metrics
    print("\n" + "="*50)
    print("                 OFFICIAL DEV RESULTS")
    print("="*50)
    
    f1 = f1_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    
    print(f"Binary F1-Score : {f1:.4f}")
    print(f"Precision       : {precision:.4f}")
    print(f"Recall          : {recall:.4f}")
    print("\nDetailed Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=['Not Patronizing', 'Patronizing']))

if __name__ == "__main__":
    evaluate_model()