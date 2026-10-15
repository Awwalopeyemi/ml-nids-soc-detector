import numpy as np
import joblib
import os
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

def run_inference():
    print("[*] Loading model and test artifacts...")
    
    model_path = "models/xgboost_nids.joblib"
    X_test_path = "data/processed/X_test.npy"
    y_test_path = "data/processed/y_test.npy"
    
    if not os.path.exists(model_path) or not os.path.exists(X_test_path):
        print("[-] Error: Model or test data artifacts not found. Please run training first.")
        return
        
    
    model = joblib.load(model_path)
    X_test = np.load(X_test_path)
    y_test = np.load(y_test_path)
    
    print(f"[*] Loaded test set of shape: {X_test.shape}")
    print("[*] Running batch inference...")
    
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    roc_auc = roc_auc_score(y_test, y_proba)
    print(f"\n[+] Inference Complete!")
    print(f"    ROC-AUC Score: {roc_auc:.4f}")
    
    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred, target_names=["Benign (0)", "Malicious (1)"]))
    
    print("\n--- Confusion Matrix ---")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    # Simulation lol..
    malicious_indices = np.where(y_pred == 1)[0]
    print(f"\n[!] SOC Alert Monitor: Flagged {len(malicious_indices)} malicious flow(s) out of {len(y_test)} total flows.")
    
    if len(malicious_indices) > 0:
        os.makedirs("data/outputs", exist_ok=True)
        alert_summary_path = "data/outputs/security_alerts.csv"
        
        # log saving 
        alert_data = np.column_stack((malicious_indices, y_proba[malicious_indices]))
        np.savetxt(alert_summary_path, alert_data, delimiter=",", header="Sample_Index,Malicious_Probability", comments="")
        print(f"[+] Security alert log exported successfully to {alert_summary_path}")

if __name__ == "__main__":
    run_inference()