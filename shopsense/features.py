import pandas as pd
import numpy as np

def compute_rfm_features(transactions_df: pd.DataFrame, snapshot_date: str) -> pd.DataFrame:
    df = transactions_df.copy()
    snapshot = pd.to_datetime(snapshot_date)
    
    # Calculate net spend (accounting for discounts and returns)
    df['net_spend'] = df['unit_price'] * df['quantity'] * (1 - df['discount_pct'])
    df['net_spend'] = np.where(df['return_flag'] == 1, 0, df['net_spend'])
    
    # Aggregate to customer level
    rfm = df.groupby('customer_id').agg(
        recency_days=('transaction_date', lambda x: (snapshot - x.max()).days),
        frequency=('transaction_id', 'count'),
        monetary_total=('net_spend', 'sum')
    )
    
    rfm['monetary_avg'] = rfm['monetary_total'] / rfm['frequency']
    
    # Safe quintile scoring using rank to prevent duplicate edge errors
    # Recency: Lower days is better (5 = most recent)
    rfm['rfm_recency_score'] = pd.qcut(rfm['recency_days'].rank(method='first'), 5, labels=[5, 4, 3, 2, 1]).astype(int)
    # Frequency & Monetary: Higher is better (5 = highest)
    rfm['rfm_frequency_score'] = pd.qcut(rfm['frequency'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    rfm['rfm_monetary_score'] = pd.qcut(rfm['monetary_total'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    
    rfm['rfm_total_score'] = rfm['rfm_recency_score'] + rfm['rfm_frequency_score'] + rfm['rfm_monetary_score']
    
    # Segment Mapping
    def map_segment(score):
        if 13 <= score <= 15: return 'Champions'
        elif 10 <= score <= 12: return 'Loyal'
        elif 7 <= score <= 9: return 'At Risk'
        elif 4 <= score <= 6: return 'Hibernating'
        else: return 'Lost'
        
    rfm['rfm_segment'] = rfm['rfm_total_score'].apply(map_segment)
    
    # Order columns exactly as requested
    cols = ['recency_days', 'frequency', 'monetary_total', 'monetary_avg', 
            'rfm_recency_score', 'rfm_frequency_score', 'rfm_monetary_score', 
            'rfm_total_score', 'rfm_segment']
    return rfm[cols]


def compute_behavioral_features(events_df: pd.DataFrame, transactions_df: pd.DataFrame, snapshot_date: str) -> pd.DataFrame:
    events = events_df.copy()
    snapshot = pd.to_datetime(snapshot_date)
    
    # Basic counts and metrics
    events['event_date_only'] = events['event_date'].dt.date
    
    behaviors = events.groupby('customer_id').agg(
        total_sessions=('event_date_only', 'nunique'),
        avg_session_duration=('session_duration_sec', 'mean'),
        total_page_views=('event_type', lambda x: (x == 'page_view').sum()),
        cart_add_count=('event_type', lambda x: (x == 'add_to_cart').sum()),
        wishlist_count=('event_type', lambda x: (x == 'wishlist_add').sum()),
        preferred_device=('device_type', lambda x: x.mode()[0] if not x.empty else 'unknown'),
        preferred_category=('page_category', lambda x: x.mode()[0] if not x.empty else 'unknown'),
        days_since_last_event=('event_date', lambda x: (snapshot - x.max()).days)
    )
    
    # Cart to Purchase Ratio (Need purchase events from events table)
    purchase_counts = events[events['event_type'] == 'purchase'].groupby('customer_id').size()
    behaviors['purchase_count_evt'] = purchase_counts
    behaviors['purchase_count_evt'] = behaviors['purchase_count_evt'].fillna(0)
    behaviors['cart_to_purchase_ratio'] = np.where(
        behaviors['cart_add_count'] > 0, 
        behaviors['purchase_count_evt'] / behaviors['cart_add_count'], 
        0.0
    )
    behaviors.drop(columns=['purchase_count_evt'], inplace=True)
    
    # Event Recency Trend (12 weeks linear regression slope)
    twelve_weeks_ago = snapshot - pd.Timedelta(days=84)
    recent_events = events[(events['event_date'] >= twelve_weeks_ago) & (events['event_date'] <= snapshot)].copy()
    
    # Assign week index 0-11
    recent_events['week_idx'] = ((recent_events['event_date'] - twelve_weeks_ago).dt.days // 7)
    recent_events['week_idx'] = np.clip(recent_events['week_idx'], 0, 11)
    
    weekly_counts = recent_events.groupby(['customer_id', 'week_idx']).size().unstack(fill_value=0)
    
    # Reindex to ensure all 12 weeks are present (0 to 11)
    weekly_counts = weekly_counts.reindex(columns=range(12), fill_value=0)
    
    x = np.arange(12)
    def calculate_slope(y):
        # np.polyfit returns [slope, intercept]
        return np.polyfit(x, y, 1)[0]
    
    slopes = weekly_counts.apply(calculate_slope, axis=1)
    behaviors['event_recency_trend'] = slopes
    behaviors['event_recency_trend'] = behaviors['event_recency_trend'].fillna(0.0)
    
    cols = ['total_sessions', 'avg_session_duration', 'total_page_views', 'cart_add_count', 
            'cart_to_purchase_ratio', 'wishlist_count', 'preferred_device', 'preferred_category', 
            'days_since_last_event', 'event_recency_trend']
    return behaviors[cols]


def compute_transaction_features(transactions_df: pd.DataFrame, snapshot_date: str) -> pd.DataFrame:
    df = transactions_df.copy()
    snapshot = pd.to_datetime(snapshot_date)
    
    df['net_spend'] = df['unit_price'] * df['quantity'] * (1 - df['discount_pct'])
    df['net_spend'] = np.where(df['return_flag'] == 1, 0, df['net_spend'])
    df['days_ago'] = (snapshot - df['transaction_date']).dt.days
    
    txns = df.groupby('customer_id').agg(
        category_diversity=('category', 'nunique'),
        preferred_payment=('payment_method', lambda x: x.mode()[0] if not x.empty else 'unknown'),
        avg_discount_received=('discount_pct', 'mean'),
        return_rate=('return_flag', 'mean'),
        revenue_last_30d=('net_spend', lambda x: x[df.loc[x.index, 'days_ago'] <= 30].sum()),
        revenue_last_90d=('net_spend', lambda x: x[df.loc[x.index, 'days_ago'] <= 90].sum()),
        revenue_last_180d=('net_spend', lambda x: x[df.loc[x.index, 'days_ago'] <= 180].sum()),
        peak_season_purchase_ratio=('transaction_date', lambda x: (x.dt.month.isin([10, 11, 12])).mean())
    )
    
    # Purchase Gap (mean and std of days between consecutive purchases)
    sorted_df = df.sort_values(['customer_id', 'transaction_date'])
    sorted_df['prev_date'] = sorted_df.groupby('customer_id')['transaction_date'].shift(1)
    sorted_df['gap_days'] = (sorted_df['transaction_date'] - sorted_df['prev_date']).dt.days
    
    gaps = sorted_df.groupby('customer_id').agg(
        purchase_gap_mean=('gap_days', 'mean'),
        purchase_gap_std=('gap_days', 'std')
    )
    
    txns = txns.join(gaps)
    
    cols = ['category_diversity', 'preferred_payment', 'avg_discount_received', 'return_rate',
            'revenue_last_30d', 'revenue_last_90d', 'revenue_last_180d', 
            'purchase_gap_mean', 'purchase_gap_std', 'peak_season_purchase_ratio']
    return txns[cols]


def build_master_feature_table(customers_df: pd.DataFrame, rfm_df: pd.DataFrame, 
                               behavioral_df: pd.DataFrame, transaction_df: pd.DataFrame) -> pd.DataFrame:
    
    # Start with customers and set index
    master = customers_df.set_index('customer_id').copy()
    
    # Drop signup_date as it's not a ML feature
    if 'signup_date' in master.columns:
        master.drop(columns=['signup_date'], inplace=True)
    
    # Save churn_label to append at the end
    churn_label = master.pop('churn_label')
    
    # Join the engineered features
    master = master.join(rfm_df, how='left')
    master = master.join(behavioral_df, how='left')
    master = master.join(transaction_df, how='left')
    
    # Fill NAs for customers with zero transactions/events
    numeric_cols = master.select_dtypes(include=[np.number]).columns
    master[numeric_cols] = master[numeric_cols].fillna(0)
    
    # Categorical fills
    cat_cols = master.select_dtypes(include=['object', 'category']).columns
    master[cat_cols] = master[cat_cols].fillna('Unknown')
    
    # Add churn label to the very end
    master['churn_label'] = churn_label
    
    return master