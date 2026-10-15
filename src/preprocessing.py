import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import joblib
import os

def preprocess_pipeline(file_path):
    print(f"[*] Loading dataset from {file_path}...")
    
    if not os.path.exists(file_path):
        print(f"[-] Error: File not found at {file_path}. Please place your raw Parquet file in data/raw/")
        return
        
    df = pd.read_parquet(file_path)
    
    df.columns = df.columns.str.strip()
    
    print(f"[*] Initial Dataset Shape: {df.shape}")
    
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    
    label_col = None
    for col in df.columns:
        if 'label' in col.lower():
            label_col = col
            break
            
    if not label_col:
        print("[-] Error: Could not locate target label column in dataset.")
        return

    X = df.drop(columns=[label_col])
    y = df[label_col]
    
    cols_to_drop = ['Flow ID', 'Source IP', 'Destination IP', 'Timestamp', 'SimillarHTTP']
    X = X.drop(columns=[c for c in cols_to_drop if c in X.columns])
    
    y_binary = y.apply(lambda val: 0 if str(val).strip().lower() == 'benign' else 1)
    
    print(f"[+] Target Distribution:\n{y_binary.value_counts(normalize=False)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_binary, test_size=0.2, random_state=42, stratify=y_binary
    )
    
    print("[*] Scaling features using StandardScaler...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("[*] Applying SMOTE over-sampling to training data...")
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
    
    print(f"[*] Balanced Training Set Shape: {X_train_balanced.shape}")
    
    os.makedirs("models", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    
    joblib.dump(scaler, "models/scaler.joblib")
    
    np.save("data/processed/X_train.npy", X_train_balanced)
    np.save("data/processed/y_train.npy", y_train_balanced)
    np.save("data/processed/X_test.npy", X_test_scaled)
    np.save("data/processed/y_test.npy", y_test)
    
    print("[+] Preprocessing pipeline completed successfully! Artifacts saved in models/ and data/processed/")

if __name__ == "__main__":
    preprocess_pipeline("ml-nids-soc-detector/data/raw/sample_traffic.parquet")