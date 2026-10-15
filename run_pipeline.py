import os
import time
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import roc_auc_score

def run_security_pipeline():
    print("Initializing Network Intrusion Detection Pipeline...")
    start_time = time.time()
    
    model_path = "models/xgboost_nids.joblib"
    X_test_path = "data/processed/X_test.npy"
    y_test_path = "data/processed/y_test.npy"
    output_dir = "data/outputs"
    output_csv = os.path.join(output_dir, "security_alerts.csv")
    archive_dir = os.path.join(output_dir, "archive")
    
    if not os.path.exists(model_path) or not os.path.exists(X_test_path):
        print("Error: Required model or test data artifacts missing.")
        return
        
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(archive_dir, exist_ok=True)
    
    print("Loading model and test data...")
    model = joblib.load(model_path)
    X_test = np.load(X_test_path)
    y_test = np.load(y_test_path)
    
    print("Executing model inference across network flows...")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    roc_auc = roc_auc_score(y_test, y_proba)
    print(f"Inference complete. ROC-AUC Score: {roc_auc:.4f}")
    
    malicious_indices = np.where(y_pred == 1)[0]
    print(f"Flagged malicious flows: {len(malicious_indices)} out of {len(y_test)}")
    
    alert_records = []
    for idx in malicious_indices:
        confidence = float(y_proba[idx])
        severity = "HIGH" if confidence > 0.85 else "MEDIUM"
        
        alert_records.append({
            "Timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Sample_Index": int(idx),
            "Severity": severity,
            "Malicious_Probability": round(confidence, 4)
        })
        
    df_alerts = pd.DataFrame(alert_records)
    
    # Save a timestamped archive copy of this specific run
    run_timestamp_str = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    archive_path = os.path.join(archive_dir, f"archive_alerts_{run_timestamp_str}.csv")
    df_alerts.to_csv(archive_path, index=False)
    
    # Append to master persistent log file
    if os.path.exists(output_csv):
        df_existing = pd.read_csv(output_csv)
        df_combined = pd.concat([df_existing, df_alerts]).drop_duplicates(subset=["Sample_Index", "Timestamp"])
        df_combined.to_csv(output_csv, index=False)
    else:
        df_alerts.to_csv(output_csv, index=False)
        
    print(f"Master incident log updated at {output_csv}")
    print(f"Run archived successfully at {archive_path}")
    
    elapsed = time.time() - start_time
    print(f"Pipeline execution finished in {elapsed:.2f} seconds.")

if __name__ == "__main__":
    run_security_pipeline()
    