import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score

def prepare_clustering_features(master_df: pd.DataFrame) -> tuple:
    # Drop target and ID column if it exists as a standard column
    cols_to_drop = ['churn_label']
    if 'customer_id' in master_df.columns:
        cols_to_drop.append('customer_id')
        
    df = master_df.drop(columns=[col for col in cols_to_drop if col in master_df.columns])
    
    # Select only numeric features for K-Means
    numeric_df = df.select_dtypes(include=[np.number])
    feature_names = numeric_df.columns.tolist()
    
    # Scale features (K-Means is distance-based, so this is mandatory)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(numeric_df)
    
    return X_scaled, feature_names, scaler

def find_optimal_k(X_scaled: np.ndarray, k_range: range = range(2, 11)) -> dict:
    inertia_values = []
    silhouette_scores = []
    calinski_harabasz_scores = []
    
    for k in k_range:
        # n_init='auto' suppresses scikit-learn warnings
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
        labels = kmeans.fit_predict(X_scaled)
        
        inertia_values.append(float(kmeans.inertia_))
        silhouette_scores.append(float(silhouette_score(X_scaled, labels)))
        calinski_harabasz_scores.append(float(calinski_harabasz_score(X_scaled, labels)))
        
    k_list = list(k_range)
    opt_k_sil = k_list[np.argmax(silhouette_scores)]
    opt_k_cal = k_list[np.argmax(calinski_harabasz_scores)]
    
    return {
        "inertia_values": inertia_values,
        "silhouette_scores": silhouette_scores,
        "calinski_harabasz_scores": calinski_harabasz_scores,
        "optimal_k_silhouette": int(opt_k_sil),
        "optimal_k_calinski": int(opt_k_cal)
    }

def train_segmentation_model(X_scaled: np.ndarray, n_clusters: int, random_state: int = 42) -> tuple:
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init='auto')
    labels = kmeans.fit_predict(X_scaled)
    return kmeans, labels

def profile_clusters(master_df: pd.DataFrame, cluster_labels: np.ndarray, feature_names: list) -> pd.DataFrame:
    df = master_df.copy()
    df['cluster'] = cluster_labels
    
    profiles = []
    
    for cluster_id, group in df.groupby('cluster'):
        profile = {}
        
        # Aggregate size and churn rate
        profile['cluster_size'] = len(group)
        if 'churn_label' in group.columns:
            profile['churn_rate'] = group['churn_label'].mean()
            
        # Calculate means for numeric and modes for categorical features
        for col in df.columns:
            if col in ['cluster', 'churn_label', 'customer_id']:
                continue
            if pd.api.types.is_numeric_dtype(df[col]):
                profile[col] = group[col].mean()
            else:
                profile[col] = group[col].mode()[0] if not group[col].mode().empty else 'unknown'
                
        profiles.append(profile)
        
    # Dataframe index automatically becomes cluster ID
    profile_df = pd.DataFrame(profiles)
    
    return profile_df