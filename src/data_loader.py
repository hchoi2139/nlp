import pandas as pd
import torch
import re
import html
import ast
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import AutoTokenizer, DataCollatorWithPadding
import os

# ==========================================
# 1. TEXT CLEANING (Strictly following EDA)
# ==========================================
def clean_text(text):
    if not isinstance(text, str): return ""
    
    # 1. Convert HTML entities to UTF-8 properly (e.g., &amp; -> &, &quot; -> ")
    text = html.unescape(text)
    
    # 2. Remove raw HTML tags to neutralize structural noise (e.g., <h>, <br>)
    text = re.sub(r'<[^>]+>', ' ', text)
    
    # 3. Remove @mentions, #hashtags, and URLs to prevent spurious correlation
    text = re.sub(r'[@#]\w+', ' ', text)
    text = re.sub(r'(?:http[s]?://|www\.)\S+', ' ', text)
    
    # 4. Collapse multiple spaces
    text = re.sub(r'\s{2,}', ' ', text).strip()
    
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
        
        # Include the 7 taxonomic categories for MTL
        if 'taxonomy_labels' in row:
            item['taxonomy_labels'] = torch.tensor(row['taxonomy_labels'], dtype=torch.float)
            
        return item

# ==========================================
# 3. DATALOADER SETUP
# ==========================================
def get_dataloader(data_filepath, categories_filepath=None, tokenizer_name="microsoft/deberta-v3-base", batch_size=16):
    print("Loading data and tokenizer...")
    
    # 1. Load Main Binary Dataset
    df = pd.read_csv(data_filepath, sep='\t', skiprows=4, header=None, 
                     names=['par_id', 'art_id', 'keyword', 'country', 'text', 'label'])
    df = df.dropna(subset=['text', 'label'])
    
    # 2. Filter context-starved fragments & extreme outliers
    df['word_count'] = df['text'].str.split().str.len()
    
    # Drop paragraphs with fewer than 7 words
    df = df[df['word_count'] >= 7].copy() 
    
    # Drop the 7 extreme sequence outliers identified in EDA profiling
    df = df.sort_values(by='word_count', ascending=False).iloc[7:].sort_index()
    
    # 3. Apply strict text cleaning
    df['text'] = df['text'].apply(clean_text)
    
    # 4. Binarize Labels (0,1 -> 0 | 2,3,4 -> 1)
    df['label'] = (df['label'] >= 2).astype(int)
    
    # 5. Load and Merge Taxonomy Categories (If provided)
    if categories_filepath:
        try:
            df_cat = pd.read_csv(categories_filepath, sep='\t', header=None, names=['par_id', 'taxonomy_labels'])
            # Convert string representation of lists to actual Python lists
            if isinstance(df_cat['taxonomy_labels'].iloc[0], str):
                df_cat['taxonomy_labels'] = df_cat['taxonomy_labels'].apply(ast.literal_eval)
            
            # Merge on par_id
            df = pd.merge(df, df_cat, on='par_id', how='left')
            
            # Fill missing taxonomy rows with zeros
            df['taxonomy_labels'] = df['taxonomy_labels'].apply(lambda x: x if isinstance(x, list) else [0.0]*7)
            print("Successfully merged MTL taxonomy categories.")
        except Exception as e:
            print(f"Warning: Could not merge taxonomy categories. MTL will fail. Error: {e}")

    # 6. Setup Imbalance Sampler
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