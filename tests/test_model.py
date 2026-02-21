import sys
import os
import torch
import torch.nn as nn

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from src.data_loader import get_dataloader
from src.model import PCLModelWithLAN

def run_gradient_check():
    print("\n--- 1. LOADING DATA ---")
    data_path = 'data/dontpatronizeme_pcl.tsv'
    cat_path = 'data/dontpatronizeme_categories.tsv'
    
    dataloader, tokenizer = get_dataloader(data_path, cat_path, batch_size=4)
    batch = next(iter(dataloader))
    
    print("\n--- 2. INSTANTIATING MODEL & OPTIMIZER ---")
    model = PCLModelWithLAN()
    
    # Push to GPU to test precision issues
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    batch = {k: v.to(device) for k, v in batch.items()}

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    criterion = nn.BCEWithLogitsLoss()

    print("\n--- 3. RUNNING FORWARD PASS ---")
    binary_logits, taxonomy_logits = model(
        input_ids=batch['input_ids'], 
        attention_mask=batch['attention_mask']
    )
    
    print(f"Binary Logits shape: {binary_logits.shape}")
    print(f"Taxonomy Logits shape: {taxonomy_logits.shape}")

    print("\n--- 4. RUNNING BACKWARD PASS (GRADIENT CHECK) ---")
    loss_binary = criterion(binary_logits, batch['labels'])
    loss_taxonomy = criterion(taxonomy_logits, batch['taxonomy_labels'])
        
    total_loss = loss_binary + loss_taxonomy
    print(f"Total Dummy Loss: {total_loss.item():.4f}")
    
    total_loss.backward()

    # --- 5. VERIFYING GRADIENTS ---
    tests_passed = True
    
    if model.label_embeddings.grad is not None and model.label_embeddings.grad.abs().sum() > 0:
        print("[PASS] LAN Label Embeddings received gradients.")
    else:
        print("[FAIL] LAN Label Embeddings are disconnected!")
        tests_passed = False

    if model.binary_head[0].weight.grad is not None and model.binary_head[0].weight.grad.abs().sum() > 0:
        print("[PASS] Binary Head received gradients.")
    else:
        print("[FAIL] Binary Head is disconnected!")
        tests_passed = False
        
    first_encoder_layer = model.deberta.embeddings.word_embeddings.weight
    if first_encoder_layer.grad is not None and first_encoder_layer.grad.abs().sum() > 0:
        print("[PASS] DeBERTa Base Encoder received gradients.")
    else:
        print("[FAIL] DeBERTa Base Encoder is frozen or disconnected!")
        tests_passed = False

    if tests_passed:
        print("\n🏆 SUCCESS: The computational graph is fully connected. The model will learn!")
    else:
        print("\n❌ ERROR: Gradient flow failed. The model will not learn properly.")

if __name__ == "__main__":
    run_gradient_check()