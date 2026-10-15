#  ML-Powered Network Intrusion Detection System (NIDS)

A production-grade Machine Learning pipeline designed to detect network anomalies and cyber attacks (such as DDoS, botnets, and port scans) using benchmark network traffic datasets (CICIDS2017 / NSL-KDD), integrated directly with SIEM log forwarding.

## Architecture Overview
* **Data Ingestion & EDA:** Automated cleaning of network flow telemetry (handling division-by-zero infinite values and severe class imbalance).
* **Feature Engineering & Preprocessing:** Robust scaling, encoding, and synthetic oversampling (`SMOTE`) to prevent data leakage.
* **Model Training & Evaluation:** Gradient-boosting classifiers (XGBoost / LightGBM) evaluated using Precision, Recall, and ROC-AUC rather than misleading accuracy metrics.
* **Operational SIEM Logging:** Real-time inference engine formatting threat predictions into structured JSON logs for SIEM ingestion.