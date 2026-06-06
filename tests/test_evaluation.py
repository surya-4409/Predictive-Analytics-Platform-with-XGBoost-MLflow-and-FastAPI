import pandas as pd
from shopsense.evaluation import detect_feature_drift

def test_detect_feature_drift():
    ref_df = pd.DataFrame({'age': [25, 30, 35], 'city': ['A', 'B', 'A']})
    cur_df = pd.DataFrame({'age': [26, 31, 36], 'city': ['A', 'A', 'A']})
    
    drift_df = detect_feature_drift(ref_df, cur_df, ['age'], ['city'])
    
    assert isinstance(drift_df, pd.DataFrame)
    assert 'drift_detected' in drift_df.columns