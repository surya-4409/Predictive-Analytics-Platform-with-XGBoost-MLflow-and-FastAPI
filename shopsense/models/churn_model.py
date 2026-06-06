import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score, 
    precision_score, recall_score, f1_score, balanced_accuracy_score, 
    confusion_matrix, classification_report
)
from sklearn.feature_selection import RFECV
from sklearn.model_selection import RandomizedSearchCV
import mlflow
import xgboost as xgb
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

def build_churn_preprocessing_pipeline(numeric_features: list, categorical_features: list) -> Pipeline:
    # Numeric transformer: Median imputation -> Standard Scaling
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Categorical transformer: Most frequent imputation -> OneHotEncoding
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Combine into a ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop' # Drop any columns not explicitly specified (like customer_id)
    )
    
    return preprocessor

def train_churn_model(X_train: pd.DataFrame, y_train: pd.Series, preprocessing_pipeline: ColumnTransformer, 
                      model_type: str = "xgboost", random_state: int = 42) -> Pipeline:
    
    # Calculate class weight ratio for imbalanced datasets
    pos_count = sum(y_train == 1)
    neg_count = sum(y_train == 0)
    scale_weight = neg_count / pos_count if pos_count > 0 else 1.0

    # Select the model based on input
    if model_type == "xgboost":
        classifier = xgb.XGBClassifier(
            scale_pos_weight=scale_weight, 
            random_state=random_state,
            use_label_encoder=False,
            eval_metric='logloss'
        )
    elif model_type == "lightgbm":
        classifier = LGBMClassifier(class_weight='balanced', random_state=random_state)
    elif model_type == "random_forest":
        classifier = RandomForestClassifier(class_weight='balanced', random_state=random_state)
    elif model_type == "logistic_regression":
        classifier = LogisticRegression(class_weight='balanced', random_state=random_state, max_iter=1000)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    # Build the final pipeline
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessing_pipeline),
        ('classifier', classifier)
    ])
    
    # Enable MLflow autologging
    mlflow.sklearn.autolog(log_models=False) # We will log the model manually in Phase 5
    
    # Fit the pipeline
    model_pipeline.fit(X_train, y_train)
    
    return model_pipeline

def find_optimal_threshold(model, X_val: pd.DataFrame, y_val: pd.Series, metric: str = "f1") -> dict:
    # Get raw probabilities for class 1
    y_probs = model.predict_proba(X_val)[:, 1]
    
    thresholds = np.arange(0.05, 0.96, 0.01)
    results = []
    
    best_threshold = 0.5
    best_score = -1.0
    
    for thresh in thresholds:
        y_pred = (y_probs >= thresh).astype(int)
        
        # Calculate metrics
        prec = precision_score(y_val, y_pred, zero_division=0)
        rec = recall_score(y_val, y_pred, zero_division=0)
        f1 = f1_score(y_val, y_pred, zero_division=0)
        bal_acc = balanced_accuracy_score(y_val, y_pred)
        
        results.append({
            'threshold': thresh,
            'precision': prec,
            'recall': rec,
            'f1': f1,
            'balanced_accuracy': bal_acc
        })
        
        # Track the best threshold based on the requested target metric
        current_score = locals()[metric] # dynamically grab f1, prec, rec, or bal_acc
        if current_score > best_score:
            best_score = current_score
            best_threshold = thresh
            
    return {
        "optimal_threshold": float(best_threshold),
        "optimal_metric_value": float(best_score),
        "threshold_curve": pd.DataFrame(results)
    }

def evaluate_churn_model(model, X_test: pd.DataFrame, y_test: pd.Series, threshold: float = 0.5) -> dict:
    y_probs = model.predict_proba(X_test)[:, 1]
    y_pred = (y_probs >= threshold).astype(int)
    
    roc_auc = roc_auc_score(y_test, y_probs)
    pr_auc = average_precision_score(y_test, y_probs)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    
    cm = confusion_matrix(y_test, y_pred).tolist()
    report = classification_report(y_test, y_pred, zero_division=0)
    
    return {
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "balanced_accuracy": float(bal_acc),
        "confusion_matrix": cm,
        "classification_report": report
    }

def run_rfe_feature_selection(X_train: pd.DataFrame, y_train: pd.Series, preprocessing_pipeline, n_features_to_select: int = 20) -> dict:
    # Transform the data first
    X_train_transformed = preprocessing_pipeline.fit_transform(X_train)
    
    # Get feature names after one-hot encoding
    feature_names = preprocessing_pipeline.get_feature_names_out()
    
    # Use a base XGBoost classifier for RFE
    estimator = xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
    
    # Run Recursive Feature Elimination with Cross-Validation
    selector = RFECV(estimator, step=1, cv=5, scoring='roc_auc', min_features_to_select=n_features_to_select)
    selector.fit(X_train_transformed, y_train)
    
    selected_features = [feature_names[i] for i, selected in enumerate(selector.support_) if selected]
    
    return {
        "selected_features": selected_features,
        "optimal_n_features": int(selector.n_features_),
        "cv_scores_mean": selector.cv_results_['mean_test_score'].tolist(),
        "cv_scores_std": selector.cv_results_['std_test_score'].tolist()
    }

def tune_churn_model(X_train: pd.DataFrame, y_train: pd.Series, preprocessing_pipeline, model_type: str = "xgboost", n_trials: int = 30, random_state: int = 42) -> dict:
    pos_count = sum(y_train == 1)
    scale_weight = sum(y_train == 0) / pos_count if pos_count > 0 else 1.0
    
    # Base model and param grid setup
    if model_type == "xgboost":
        base_model = xgb.XGBClassifier(scale_pos_weight=scale_weight, random_state=random_state, use_label_encoder=False, eval_metric='logloss')
        param_grid = {
            'classifier__n_estimators': [100, 200, 300],
            'classifier__max_depth': [3, 4, 5, 6],
            'classifier__learning_rate': [0.01, 0.05, 0.1, 0.2],
            'classifier__subsample': [0.6, 0.8, 1.0],
            'classifier__colsample_bytree': [0.6, 0.8, 1.0]
        }
    else:
        raise ValueError("Only XGBoost tuning is implemented in this snippet.")
        
    # We must construct a pipeline to prevent data leakage during CV
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessing_pipeline),
        ('classifier', base_model)
    ])
    
    # Randomized Search
    search = RandomizedSearchCV(
        pipeline, param_distributions=param_grid, n_iter=n_trials,
        scoring='roc_auc', cv=5, random_state=random_state, n_jobs=-1
    )
    
    search.fit(X_train, y_train)
    
    return {
        "best_params": search.best_params_,
        "best_cv_score": float(search.best_score_),
        "best_model": search.best_estimator_
    }