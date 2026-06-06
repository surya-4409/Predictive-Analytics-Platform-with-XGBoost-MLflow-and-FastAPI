import pandas as pd
from shopsense.features import compute_rfm_features

def test_compute_rfm_features_columns():
    # Dummy data with 5 unique customers to satisfy the qcut (quintile) math
    df = pd.DataFrame({
        'transaction_id': ['T1', 'T2', 'T3', 'T4', 'T5'],
        'customer_id': ['C1', 'C2', 'C3', 'C4', 'C5'],
        'transaction_date': pd.to_datetime(['2023-01-01', '2023-01-10', '2023-02-01', '2023-03-01', '2023-04-01']),
        'quantity': [1, 2, 1, 3, 1],
        'unit_price': [100, 50, 200, 30, 400],
        'discount_pct': [0.0, 0.1, 0.0, 0.0, 0.2],
        'return_flag': [0, 0, 0, 0, 0]
    })
    rfm = compute_rfm_features(df, '2023-12-31')
    
    expected_cols = ['recency_days', 'frequency', 'monetary_total', 'monetary_avg', 
                     'rfm_recency_score', 'rfm_frequency_score', 'rfm_monetary_score', 
                     'rfm_total_score', 'rfm_segment']
    
    assert list(rfm.columns) == expected_cols
    assert len(rfm) == 5