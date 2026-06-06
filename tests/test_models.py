from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from shopsense.models.churn_model import build_churn_preprocessing_pipeline

def test_build_churn_preprocessing_pipeline():
    num_cols = ['age', 'recency_days']
    cat_cols = ['city', 'gender']
    
    pipeline = build_churn_preprocessing_pipeline(num_cols, cat_cols)
    
    assert isinstance(pipeline, ColumnTransformer)
    assert len(pipeline.transformers) == 2 # num and cat