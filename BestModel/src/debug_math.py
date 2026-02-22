import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

# Fix paths
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.data_loader import get_dataloader
from src.model import PCLModelWithLAN

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        logits = logits.view(-1)
        targets = targets.view(-1).float()
        log_pt = -F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = torch.exp(log_pt)
        focal_loss = -self.alpha * ((1 - pt) ** self.gamma) * log_pt
        return focal_loss.mean()

# --- FORENSIC SCANNER ---
def check_tensor(tensor, name):
    if torch.isnan(tensor).any():
        print(f"🚨 NaN detected in: {name}")
        return True
    if torch.isinf(tensor).any():
        print(f"🚨 Inf detected in: {name}")
        return True
    return False

def run_sanity_check():
    print("🧪 STARTING FORENSIC CPU SANITY CHECK...")
    device = torch.device('cpu')
    
    data_path = 'data/dontpatronizeme_pcl.tsv'
    cat_path = 'data/dontpatronizeme_categories.tsv'
    train_loader, _ = get_dataloader(data_path, cat_path, batch_size=4)

    model = PCLModelWithLAN().to(device).float()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6)
    
    criterion_binary = FocalLoss()
    criterion_taxonomy = nn.BCEWithLogitsLoss()

    model.train()
    
    for i, batch in enumerate(train_loader):
        if i >= 3: break 
        print(f"\n--- BATCH {i+1} ---")
        
        # 1. Check Weights BEFORE forward pass
        for name, param in model.named_parameters():
            if param.requires_grad and check_tensor(param.data, f"Weight: {name}"):
                sys.exit(1)

        input_ids = batch['input_ids']
        mask = batch['attention_mask']
        labels = batch['labels'].float()
        tax_labels = batch['taxonomy_labels'].float()
        
        # 2. Check Input Data
        if check_tensor(input_ids.float(), "input_ids"): sys.exit(1)
        if check_tensor(mask.float(), "attention_mask"): sys.exit(1)

        optimizer.zero_grad()
        
        # 3. Step-by-step Interception (Mimicking model.py forward pass)
        outputs = model.deberta(input_ids=input_ids, attention_mask=mask)
        sequence_output = outputs.last_hidden_state
        
        # --- THE MISSING FIX ---
        # Force float32 to match the custom attention weights
        sequence_output = sequence_output.to(model.label_embeddings.dtype)
        
        if check_tensor(sequence_output, "DeBERTa sequence_output"): sys.exit(1)
        
        cls_token = sequence_output[:, 0, :]
        batch_size = sequence_output.size(0)
        
        queries = model.label_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
        key_padding_mask = (mask == 0)
        
        normed_sequence = model.layer_norm(sequence_output)
        if check_tensor(normed_sequence, "LayerNorm Output"): sys.exit(1)

        lan_output, attn_weights = model.attention(
            query=queries, 
            key=normed_sequence, 
            value=normed_sequence, 
            key_padding_mask=key_padding_mask
        )
        if check_tensor(attn_weights, "Attention Weights (Softmax output)"): sys.exit(1)
        if check_tensor(lan_output, "LAN Output"): sys.exit(1)

        taxonomy_logits = model.taxonomy_head(lan_output).squeeze(-1) 
        lan_aggregated = lan_output.mean(dim=1) 
        combined_features = torch.cat([cls_token, lan_aggregated], dim=1) 
        binary_logits = model.binary_head(combined_features).squeeze(-1)
        
        if check_tensor(binary_logits, "Binary Logits"): sys.exit(1)
        if check_tensor(taxonomy_logits, "Taxonomy Logits"): sys.exit(1)

        # 4. Check Loss Calculation
        l_bin = criterion_binary(binary_logits, labels)
        l_tax = criterion_taxonomy(taxonomy_logits, tax_labels)
        loss = l_bin + (0.5 * l_tax)
        
        if check_tensor(loss, "Total Loss"): sys.exit(1)

        loss.backward()
        
        # 5. Check Gradients BEFORE step
        for name, param in model.named_parameters():
            if param.grad is not None:
                if check_tensor(param.grad, f"Gradient of {name}"): sys.exit(1)

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        
        print(f"Batch {i+1} Success | Loss: {loss.item():.4f} | Grad: {grad_norm:.2f}")

if __name__ == "__main__":
    run_sanity_check()