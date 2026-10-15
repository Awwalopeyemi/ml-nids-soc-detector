import pandas as pd
import numpy as np
import os

def run_eda(file_path):
    print(f"[*] Loading dataset from {file_path}...")
    
    if not os.path.exists(file_path):
        print(f"[-] Error: File not found at {file_path}.")
        print("[*] Tip: Place your downloaded CICIDS2017 or benchmark CSV file inside the 'data/raw/' directory.")
        return
        
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()
    
    print(f"[*] Initial Dataset Shape: {df.shape}")
    
    initial_rows = len(df)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    nan_counts = df.isna().sum().sum()
    print(f"[*] Found {nan_counts} NaN/Infinite values across the dataset.")
    
    df.dropna(inplace=True)
    print(f"[*] Rows after cleaning NaNs/Infs: {len(df)} (Dropped {initial_rows - len(df)} invalid rows)")
    
    label_col = None
    for col in df.columns:
        if 'label' in col.lower():
            label_col = col
            break
            
    if label_col:
        print(f"\n[+] Class Distribution using column '{label_col}':")
        print(df[label_col].value_counts(normalize=False))
        print("\n[+] Percentage Breakdown:")
        print(df[label_col].value_counts(normalize=True) * 100)
    else:
        print("[-] Warning: Could not find a 'Label' or target column in the dataset.")

if __name__ == "__main__":
   
    sample_path = "data/raw/sample_traffic.csv"
    run_eda(sample_path)