import torch
import torch.nn as nn
from transformers import AutoModel

class PCLModelWithLAN(nn.Module):
    def __init__(self, model_name="microsoft/deberta-v3-base", num_taxonomy_labels=7):
        super().__init__()
        
        self.deberta = AutoModel.from_pretrained(model_name)
        hidden_size = self.deberta.config.hidden_size 
        
        # FIX 1: Scale down the random embeddings so they don't explode on Batch 1
        self.label_embeddings = nn.Parameter(torch.randn(num_taxonomy_labels, hidden_size) * 0.02)
        
        self.layer_norm = nn.LayerNorm(hidden_size)
        
        # FIX 2: Define the missing MultiheadAttention layer
        self.attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=8, batch_first=True)
        
        self.taxonomy_head = nn.Linear(hidden_size, 1) 
        self.binary_head = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, 1)
        )
        
        # FIX 3: Zero-initialize the final prediction layer in the Sequential block
        nn.init.constant_(self.binary_head[-1].weight, 0)
        nn.init.constant_(self.binary_head[-1].bias, 0)

    def forward(self, input_ids, attention_mask):
        outputs = self.deberta(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state 
        
        # --- MIXED PRECISION FIX ---
        sequence_output = sequence_output.to(self.label_embeddings.dtype)
        
        cls_token = sequence_output[:, 0, :]        
        batch_size = sequence_output.size(0)
        
        queries = self.label_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
        key_padding_mask = (attention_mask == 0)

        # FIX 4: Apply LayerNorm to the sequence BEFORE passing it to attention
        normed_sequence = self.layer_norm(sequence_output)

        lan_output, _ = self.attention(
            query=queries, 
            key=normed_sequence, 
            value=normed_sequence, 
            key_padding_mask=key_padding_mask
        )

        taxonomy_logits = self.taxonomy_head(lan_output).squeeze(-1) 
        lan_aggregated = lan_output.mean(dim=1) 
        
        combined_features = torch.cat([cls_token, lan_aggregated], dim=1) 
        binary_logits = self.binary_head(combined_features).squeeze(-1)

        return binary_logits, taxonomy_logits