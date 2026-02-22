import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# Ensure Python can find the src package
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from src.data_loader import get_dataloader
from src.model import PCLModelWithLAN

def compute_kl_loss(logits1, logits2):
    """Calculates symmetric KL Divergence for binary logits."""
    p = torch.sigmoid(logits1)
    q = torch.sigmoid(logits2)
    
    # Prevent log(0) with a tiny epsilon
    eps = 1e-8
    
    # KL(p || q)
    kl_p_q = p * torch.log(p / (q + eps) + eps) + (1 - p) * torch.log((1 - p) / (1 - q + eps) + eps)
    # KL(q || p)
    kl_q_p = q * torch.log(q / (p + eps) + eps) + (1 - q) * torch.log((1 - q) / (1 - p + eps) + eps)
    
    # Symmetric average
    return torch.mean(kl_p_q + kl_q_p) / 2

def get_optimizer_grouped_parameters(model, base_lr=1e-5, weight_decay=0.01):
    no_decay = ["bias", "LayerNorm.weight"]
    
    zone1_params = [] # Slow learning
    zone2_params = [] # Standard learning
    zone3_params = [] # Fast learning
    zone4_params = [] # No weight decay
    
    for n, p in model.named_parameters():
        # 1. Filter out biases and LayerNorms first (Zone 4)
        if any(nd in n for nd in no_decay):
            zone4_params.append(p)
            continue
            
        # 2. Custom Head & LAN (Zone 3)
        # We check this FIRST so "label_embeddings" gets caught here 
        # and doesn't fall down into the Zone 1 "embeddings" check.
        if "head" in n or "label_attention" in n or "label_embeddings" in n:
            zone3_params.append(p)
            
        # 3. Lower Layers & Base Embeddings (Zone 1)
        elif "embeddings" in n or any(f"layer.{i}." in n for i in range(4)):
            zone1_params.append(p)
            
        # 4. Middle/High Layers & Anything Else (Zone 2)
        else:
            zone2_params.append(p)

    optimizer_grouped_parameters = [
        {"params": zone1_params, "weight_decay": weight_decay, "lr": base_lr * 0.1},
        {"params": zone2_params, "weight_decay": weight_decay, "lr": base_lr},
        {"params": zone3_params, "weight_decay": weight_decay, "lr": base_lr * 5},
        {"params": zone4_params, "weight_decay": 0.0, "lr": base_lr},
    ]
    return optimizer_grouped_parameters

def train_model():
    print("--- INITIALIZING PCL TRAINING PIPELINE ---")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")

    data_path = 'data/dontpatronizeme_pcl.tsv'
    cat_path = 'data/dontpatronizeme_categories.tsv'
    
    train_loader, val_loader, _ = get_dataloader(data_path, cat_path, batch_size=16)

    # --- 2. THE MIXED PRECISION FIX ---
    # Force the model into FP32 to prevent epsilon division-by-zero
    model = PCLModelWithLAN().float().to(device)
    
    grouped_params = get_optimizer_grouped_parameters(model, base_lr=1e-5, weight_decay=0.01)
    optimizer = torch.optim.AdamW(grouped_params)
    
    #criterion_binary = FocalLoss(alpha=0.75, gamma=2.0)
    criterion_binary = nn.BCEWithLogitsLoss()
    criterion_taxonomy = nn.BCEWithLogitsLoss()

    epochs = 3
    mtl_weight = 0.5 
    
    # --- CHECKPOINT DIRECTORY SETUP ---
    checkpoint_dir = '/vol/bitbucket/hc1721/nlp_scratch/checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_loss = float('inf')

    print("\n--- STARTING TRAINING ---")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch in loop:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # --- 3. FORCE FLOAT LABELS ---
            labels = batch['labels'].to(device).float()
            taxonomy_labels = batch['taxonomy_labels'].to(device).float()
            
            optimizer.zero_grad()
            
            # R-Drop
            binary_logits1, taxonomy_logits1 = model(input_ids, attention_mask)
            binary_logits2, taxonomy_logits2 = model(input_ids, attention_mask)

            loss_bce1 = criterion_binary(binary_logits1, labels)
            loss_bce2 = criterion_binary(binary_logits2, labels)
            loss_binary = 0.5 * (loss_bce1 + loss_bce2)

            kl_alpha = 1.0
            loss_kl = compute_kl_loss(binary_logits1, binary_logits2)

            loss_tax1 = criterion_taxonomy(taxonomy_logits1, taxonomy_labels)
            loss_tax2 = criterion_taxonomy(taxonomy_logits2, taxonomy_labels)
            loss_taxonomy = 0.5 * (loss_tax1 + loss_tax2)
            
            loss = loss_binary + (kl_alpha * loss_kl) + (mtl_weight * loss_taxonomy)
            
            loss.backward()
            
            # --- 4. STRICT GRADIENT CLIPPING ---
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            
            optimizer.step()
            
            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} Completed | Average Loss: {avg_loss:.4f}")

        # --- NEW: SAVE CHECKPOINT LOGIC ---
        checkpoint_state = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'loss': avg_loss,
        }
        
        # Save the latest epoch
        last_path = os.path.join(checkpoint_dir, 'last_checkpoint.pth')
        torch.save(checkpoint_state, last_path)
        
        # Save a separate copy if it is the best so far
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_path = os.path.join(checkpoint_dir, 'best_model.pth')
            torch.save(checkpoint_state, best_path)
            print(f"⭐ New best model saved to {best_path}")

    print("\n🏆 TRAINING COMPLETE AND WEIGHTS SAVED!")

if __name__ == "__main__":
    train_model()