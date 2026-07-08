import pandas as pd
import statsmodels.api as sm
from sklearn.cluster import KMeans
import os

dataset_path = "upi_transactions_2024.csv"

def _load_data():
    if os.path.exists(dataset_path):
        return pd.read_csv(dataset_path)
    return pd.DataFrame()

def calculate_what_if(target_network: str) -> dict:
    """
    Performs a counterfactual simulation using Logistic Regression.
    Predicts probability of 'SUCCESS' based on network_type, amount (INR), and device_type.
    """
    df = _load_data()
    if df.empty:
        return {"error": "Dataset not found or is empty."}
        
    required_cols = ["network_type", "amount (INR)", "device_type", "transaction_status"]
    if not all(col in df.columns for col in required_cols):
        return {"error": "Missing required columns in dataset."}
        
    clean_df = df.dropna(subset=required_cols).copy()
    if clean_df.empty:
        return {"error": "No valid data available after dropping NaNs."}
        
    clean_df["target"] = (clean_df["transaction_status"].str.upper() == "SUCCESS").astype(int)
    
    is_3g_wifi = clean_df["network_type"].isin(["3G", "WiFi"])
    subset_3g_wifi = clean_df[is_3g_wifi].copy()
    
    if subset_3g_wifi.empty:
        return {"error": "No data found for users on 3G or WiFi to establish baseline."}
        
    baseline_success_rate = subset_3g_wifi["target"].mean() * 100
    
    X_cat = pd.get_dummies(clean_df[["network_type", "device_type"]], drop_first=True, dtype=float)
    X_num = clean_df[["amount (INR)"]]
    
    X = pd.concat([X_num, X_cat], axis=1)
    X = sm.add_constant(X) 
    y = clean_df["target"]
    
    try:
        model = sm.Logit(y, X).fit(disp=0)
    except Exception as e:
        return {"error": f"Model fitting failed: {str(e)}"}
        
    sim_df = subset_3g_wifi[["amount (INR)", "network_type", "device_type"]].copy()
    sim_df["network_type"] = target_network 
    
    sim_cat = pd.get_dummies(sim_df[["network_type", "device_type"]], drop_first=True, dtype=float)
    sim_num = sim_df[["amount (INR)"]]
    
    sim_X = pd.concat([sim_num, sim_cat], axis=1)
    
    sim_X = sim_X.reindex(columns=X.columns.drop('const', errors='ignore'), fill_value=0.0)
    sim_X = sm.add_constant(sim_X, has_constant='add')
    
    try:
        predicted_probs = model.predict(sim_X)
        simulated_success_rate = predicted_probs.mean() * 100
    except Exception as e:
         return {"error": f"Prediction failed: {str(e)}"}
         
    return {
        "baseline_network": "3G/WiFi",
        "target_network": target_network,
        "baseline_success_rate_percent": round(baseline_success_rate, 2),
        "simulated_success_rate_percent": round(simulated_success_rate, 2),
        "absolute_improvement_percent": round(simulated_success_rate - baseline_success_rate, 2)
    }

def get_user_clusters() -> dict:
    """
    Groups transactions based on amount (INR) and hour_of_day into 3 clusters using K-Means.
    """
    df = _load_data()
    if df.empty:
        return {"error": "Dataset not found or is empty."}
        
    required_cols = ["amount (INR)", "hour_of_day"]
    if not all(col in df.columns for col in required_cols):
         return {"error": "Missing required columns in dataset."}
         
    clean_df = df.dropna(subset=required_cols).copy()
    if clean_df.empty:
         return {"error": "No valid data available after dropping NaNs."}
         
    X = clean_df[required_cols]
    
    try:
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        clean_df["cluster"] = kmeans.fit_predict(X)
    except Exception as e:
        return {"error": f"Clustering failed: {str(e)}"}
        
    summary = clean_df.groupby("cluster").agg(
        transaction_count=("amount (INR)", "count"),
        avg_amount_inr=("amount (INR)", "mean"),
        avg_hour_of_day=("hour_of_day", "mean")
    ).round(2).reset_index()
    
    clusters_list = []
    for _, row in summary.iterrows():
        clusters_list.append({
            "cluster_id": int(row["cluster"]),
            "transaction_count": int(row["transaction_count"]),
            "avg_amount_inr": float(row["avg_amount_inr"]),
            "avg_hour_of_day": float(row["avg_hour_of_day"])
        })
        
    return {"clusters": clusters_list}
