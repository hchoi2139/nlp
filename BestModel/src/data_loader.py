import pandas as pd
import torch
import re
import html
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import AutoTokenizer, DataCollatorWithPadding

def clean_text(text):
    if not isinstance(text, str): return ""
    text = html.unescape(text)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'[@#]\w+', ' ', text)
    text = re.sub(r'(?:http[s]?://|www\.)\S+', ' ', text)
    text = re.sub(r'\s{2,}', ' ', text).strip()
    return text

class PCLDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=256):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = row['text']
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None
        )
        
        item = {key: torch.tensor(val) for key, val in encoding.items()}
        item['labels'] = torch.tensor(row['label'], dtype=torch.float)
        
        if 'taxonomy_labels' in row:
            item['taxonomy_labels'] = torch.tensor(row['taxonomy_labels'], dtype=torch.float)
            
        return item

def get_dataloader(data_filepath, categories_filepath=None, tokenizer_name="microsoft/deberta-v3-base", batch_size=16):
    print("Loading binary dataset...")
    df = pd.read_csv(data_filepath, sep='\t', skiprows=4, header=None, 
                     names=['par_id', 'art_id', 'keyword', 'country', 'text', 'label'])
    df = df.dropna(subset=['text', 'label'])
    
    # Cast par_id to string to prevent merge failures
    df['par_id'] = df['par_id'].astype(str)

    df['word_count'] = df['text'].str.split().str.len()
    df = df[df['word_count'] >= 7].copy() 
    df = df.sort_values(by='word_count', ascending=False).iloc[7:].sort_index()
    df['text'] = df['text'].apply(clean_text)
    df['label'] = (df['label'] >= 2).astype(int)
    
    if categories_filepath:
        try:
            print("Loading and parsing taxonomy categories...")
            df_cat = pd.read_csv(categories_filepath, sep='\t', skiprows=4, header=None, 
                                 names=['par_id', 'art_id', 'text', 'keyword', 'country', 
                                        'span_start', 'span_end', 'span_text', 'category_label', 'num_annotators'])
            
            df_cat['par_id'] = df_cat['par_id'].astype(str)
            
            tag2id = {
                'Unbalanced_power_relations':0, 'Shallow_solution':1, 'Presupposition':2, 
                'Authority_voice':3, 'Metaphors':4, 'Compassion':5, 'The_poorer_the_merrier':6
            }
            df_cat['cat_id'] = df_cat['category_label'].map(tag2id)
            
            def make_multi_hot(cat_ids):
                vec = [0.0] * 7
                for cid in cat_ids.dropna():
                    vec[int(cid)] = 1.0
                return vec
                
            multi_hot_df = df_cat.groupby('par_id')['cat_id'].apply(make_multi_hot).reset_index()
            multi_hot_df.rename(columns={'cat_id': 'taxonomy_labels'}, inplace=True)
            
            df = pd.merge(df, multi_hot_df, on='par_id', how='left')
            df['taxonomy_labels'] = df['taxonomy_labels'].apply(lambda x: x if isinstance(x, list) else [0.0]*7)
            print("Successfully merged MTL taxonomy categories.")
        except Exception as e:
            print(f"Warning: Could not merge taxonomy categories. Error: {e}")

    print("Loading official practice splits...")
    train_split_df = pd.read_csv('data/practice-splits/train_semeval_parids-labels.csv')
    dev_split_df = pd.read_csv('data/practice-splits/dev_semeval_parids-labels.csv')

    train_ids = set(train_split_df['par_id'].astype(str).tolist())
    dev_ids = set(dev_split_df['par_id'].astype(str).tolist())

    train_df = df[df['par_id'].isin(train_ids)].copy().reset_index(drop=True)
    val_df = df[df['par_id'].isin(dev_ids)].copy().reset_index(drop=True)

    print(f"Data Split Complete: {len(train_df)} Train | {len(val_df)} Val")
    
    class_counts = train_df['label'].value_counts().sort_index().values
    class_weights = 1.0 / class_counts
    sample_weights = [class_weights[l] for l in train_df['label'].values]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    train_dataset = PCLDataset(train_df, tokenizer, max_length=256)
    val_dataset = PCLDataset(val_df, tokenizer, max_length=256)
    collate_fn = DataCollatorWithPadding(tokenizer=tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    print("Pipeline Ready!")
    return train_loader, val_loader, tokenizer