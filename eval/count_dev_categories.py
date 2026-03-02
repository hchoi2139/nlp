import os
import csv
import pandas as pd

def count_dev_categories():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    
    # 1. Load the Dev Set IDs
    dev_split_path = os.path.join(project_root, 'data', 'practice-splits', 'dev_semeval_parids-labels.csv')
    dev_split_df = pd.read_csv(dev_split_path)
    dev_ids = set(dev_split_df['par_id'].astype(str).tolist())
    
    # 2. Load the Categories data
    cat_path = os.path.join(project_root, 'data', 'dontpatronizeme_categories.tsv')
    cat_df = pd.read_csv(cat_path, sep='\t', skiprows=4, header=None, 
                         names=['par_id', 'art_id', 'text', 'keyword', 'country', 
                                'span_start', 'span_end', 'span_text', 'category', 'num_annotators'], 
                         quoting=csv.QUOTE_NONE)
    cat_df['par_id'] = cat_df['par_id'].astype(str)
    
    # 3. Filter categories to ONLY include Dev Set paragraphs
    dev_cat_df = cat_df[cat_df['par_id'].isin(dev_ids)]
    
    # 4. Calculate distributions
    # Method A: How many unique paragraphs contain the trope at least once? 
    # (This matches the logic we used for your 80 missed samples)
    unique_cats_per_par = dev_cat_df.groupby('par_id')['category'].apply(lambda x: list(set(x))).explode()
    par_counts = unique_cats_per_par.value_counts()
    
    # Method B: How many total span annotations exist? (Just for reference)
    span_counts = dev_cat_df['category'].value_counts()
    
    print("=== DEV SET CATEGORY DISTRIBUTION ===")
    print("\n1. Number of PARAGRAPHS containing each category:")
    print("   (Use these numbers as your denominator for the error analysis)")
    for cat, count in par_counts.items():
        print(f"   - {cat}: {count}")
        
    print("\n2. Number of TOTAL SPAN ANNOTATIONS for each category:")
    for cat, count in span_counts.items():
        print(f"   - {cat}: {count}")

if __name__ == "__main__":
    count_dev_categories()