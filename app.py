import streamlit as st
import pandas as pd
import os

st.set_page_config(
    page_title="Network Intrusion Detection Security Dashboard",
    layout="wide"
)

st.title("Network Intrusion Detection Security Dashboard")
st.markdown("Operational monitoring panel for machine learning intrusion detection logs.")

alert_path = "data/outputs/security_alerts.csv"

if os.path.exists(alert_path):
    df_alerts = pd.read_csv(alert_path)
    
    # Sidebar Filter Controls
    st.sidebar.header("Filter Controls")
    
    available_severities = df_alerts["Severity"].unique().tolist() if "Severity" in df_alerts.columns else []
    selected_severities = st.sidebar.multiselect(
        "Filter by Severity", 
        options=available_severities, 
        default=available_severities
    )
    
    min_conf = float(df_alerts["Malicious_Probability"].min()) if "Malicious_Probability" in df_alerts.columns else 0.0
    conf_threshold = st.sidebar.slider(
        "Minimum Confidence Threshold", 
        min_value=0.0, 
        max_value=1.0, 
        value=min_conf, 
        step=0.01
    )
    
    # Apply Filters to Dataset
    filtered_df = df_alerts.copy()
    if "Severity" in filtered_df.columns and selected_severities:
        filtered_df = filtered_df[filtered_df["Severity"].isin(selected_severities)]
    if "Malicious_Probability" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["Malicious_Probability"] >= conf_threshold]
    
    # Metrics Row
    total_threats = len(filtered_df)
    total_evaluated = 77943
    threat_ratio = (total_threats / total_evaluated) * 100 if total_evaluated > 0 else 0
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Filtered Active Threats", total_threats)
    col2.metric("Total Flows Evaluated", total_evaluated)
    col3.metric("Threat Ratio", f"{threat_ratio:.2f}%")
    
    st.markdown("---")
    
    st.subheader("Anomaly Confidence Distribution")
    st.markdown("Probability scores associated with filtered security events:")
    if "Malicious_Probability" in filtered_df.columns and not filtered_df.empty:
        st.bar_chart(filtered_df.set_index("Sample_Index")["Malicious_Probability"], height=300, width="stretch")
    else:
        st.info("No data available matching the selected filter criteria.")
        
    st.markdown("---")
    
    st.subheader("Incident Event Log")
    st.markdown("Detailed breakdown of filtered malicious network streams:")
    st.dataframe(filtered_df, width="stretch", height=500)
    
else:
    st.warning("Security alert log not found. Execute prediction script to populate metrics.")

st.markdown("---")
st.caption("Security Operations Center | Local Monitoring Node")