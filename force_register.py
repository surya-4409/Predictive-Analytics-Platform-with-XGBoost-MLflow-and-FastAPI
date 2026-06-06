import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient

# 1. Point exactly to the Docker MLflow server
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("shopsense_churn_experiment")

# 2. Train a fast, dummy model with the exact schema the API expects
print("Training and logging model to MLflow Docker container...")
with mlflow.start_run() as run:
    model = RandomForestClassifier(n_estimators=10, max_depth=3)
    X = pd.DataFrame([{"total_sessions": 15, "recency_days": 5, "is_premium": True, "age": 30}])
    y = [0]
    model.fit(X, y)
    
    signature = infer_signature(X, model.predict(X))
    mlflow.sklearn.log_model(model, "model", signature=signature)
    run_id = run.info.run_id

# 3. Register it to Staging
model_name = "shopsense_churn_model"
model_uri = f"runs:/{run_id}/model"
reg_model = mlflow.register_model(model_uri, model_name)

client = MlflowClient()
client.transition_model_version_stage(
    name=model_name,
    version=reg_model.version,
    stage="Staging",
    archive_existing_versions=True
)
print("✅ Model successfully registered and moved to Staging!")