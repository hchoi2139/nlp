import pandas as pd
import torch
import re
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import AutoTokenizer, DataCollatorWithPadding
import os

# ==========================================
# 1. TEXT CLEANING (From EDA Stage 2)
# ==========================================
def clean_text(text):
    if not isinstance(text, str): return ""
    text = re.sub(r'&[a-zA-Z0-9#]+;', ' ', text)       # HTML entities
    text = re.sub(r'<[^>]+>', ' ', text)               # HTML tags like <h>
    text = re.sub(r'[@#]\w+', ' ', text)               # Mentions/Hashtags
    text = re.sub(r'(?:http[s]?://|www\.)\S+', ' ', text) # URLs
    text = re.sub(r'\s{2,}', ' ', text).strip()        # Extra whitespaces
    return text

# ==========================================
# 2. PYTORCH DATASET
# ==========================================
class PCLDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=256):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length # Capping at 256 to drop narrative bloat

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = row['text']
        
        # Tokenize without padding (DataCollator handles dynamic padding)
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None
        )
        
        item = {key: torch.tensor(val) for key, val in encoding.items()}
        item['labels'] = torch.tensor(row['label'], dtype=torch.float)
        
        return item

# ==========================================
# 3. DATALOADER SETUP
# ==========================================
def get_dataloader(filepath, tokenizer_name="microsoft/deberta-v3-base", batch_size=16):
    print("Loading data and tokenizer...")
    df = pd.read_csv(filepath, sep='\t', skiprows=4, header=None, 
                     names=['par_id', 'art_id', 'keyword', 'country', 'text', 'label'])
    df = df.dropna(subset=['text', 'label'])
    
    # Apply EDA Filters
    df['word_count'] = df['text'].str.split().str.len()
    df = df[df['word_count'] >= 7].copy() # Drop context-starved fragments
    df['text'] = df['text'].apply(clean_text)
    
    # Binarize Labels (0,1 -> 0 | 2,3,4 -> 1)
    df['label'] = (df['label'] >= 2).astype(int)
    
    # Setup Imbalance Sampler
    class_counts = df['label'].value_counts().sort_index().values
    class_weights = 1.0 / class_counts
    sample_weights = [class_weights[l] for l in df['label'].values]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    dataset = PCLDataset(df, tokenizer, max_length=256)
    
    # Dynamic Padding Collate Function
    collate_fn = DataCollatorWithPadding(tokenizer=tokenizer)
    
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, collate_fn=collate_fn)
    
    print(f"Pipeline Ready! Filtered dataset size: {len(df)}")
    return dataloader, tokenizer