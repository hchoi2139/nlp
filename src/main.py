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

def get_optimizer_grouped_parameters(model, base_lr=1e-5, weight_decay=0.01):
    no_decay = ["bias", "LayerNorm.weight"]
    
    # We create distinct learning zones
    optimizer_grouped_parameters = [
        # Zone 1: Lower Layers (Slow Learning - preserve pre-trained syntax)
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) 
                       and ("embeddings" in n or any(f"layer.{i}." in n for i in range(4)))],
            "weight_decay": weight_decay,
            "lr": base_lr * 0.1, # 1e-6
        },
        # Zone 2: Middle and High Layers (Standard Learning - semantics)
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) 
                       and any(f"layer.{i}." in n for i in range(4, 12))],
            "weight_decay": weight_decay,
            "lr": base_lr, # 1e-5
        },
        # Zone 3: Custom Head & Label Attention (Fast Learning - task specific)
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) 
                       and ("head" in n or "label_attention" in n or "label_embeddings" in n)],
            "weight_decay": weight_decay,
            "lr": base_lr * 5, # 5e-5
        },
        # Zone 4: Biases and LayerNorm (No Weight Decay for stability)
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
            "lr": base_lr,
        },
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
    model = PCLModelWithLAN().to(device).float()
    
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
            
            binary_logits, taxonomy_logits = model(input_ids, attention_mask)
            
            loss_binary = criterion_binary(binary_logits, labels)
            loss_taxonomy = criterion_taxonomy(taxonomy_logits, taxonomy_labels)
            
            loss = loss_binary + (mtl_weight * loss_taxonomy)
            
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