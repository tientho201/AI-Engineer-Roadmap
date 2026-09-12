import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("churn-prediction")

X, y = load_breast_cancer(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.2, random_state=42)

with mlflow.start_run(run_name="rf-baseline") as run:
    params = {"n_estimators": 300, "max_depth": 8, "random_state": 42}
    model = RandomForestClassifier(**params).fit(Xtr, ytr)
    proba = model.predict_proba(Xte)[:, 1]

    mlflow.log_params(params)
    mlflow.log_metrics({
        "roc_auc": roc_auc_score(yte, proba),
        "pr_auc":  average_precision_score(yte, proba),
    })
    mlflow.set_tags({"data_version": "v2.1", "owner": "ai-team", "stage": "dev"})

    # Log model KÈM signature -> serving tự validate schema đầu vào
    from mlflow.models import infer_signature
    mlflow.sklearn.log_model(
        model, name="model",
        signature=infer_signature(Xtr, model.predict(Xtr)),
        input_example=Xtr[:3],
        registered_model_name="churn-classifier",
    )
    print("run_id:", run.info.run_id)


# --- ĐĂNG KÝ VÀ CHUYỂN TRẠNG THÁI MODEL ---
from mlflow import MlflowClient
client = MlflowClient()
client.set_registered_model_alias("churn-classifier", "champion", version=1)

# Load model đang phục vụ production
prod_model = mlflow.pyfunc.load_model("models:/churn-classifier@champion")