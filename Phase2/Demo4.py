from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import numpy as np

X, y = load_breast_cancer(return_X_y=True)

models = {
    "DecisionTree": DecisionTreeClassifier(max_depth=5, random_state=0),
    "RandomForest": RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=-1),
    "XGBoost":      XGBClassifier(n_estimators=300, learning_rate=0.05,
                                  max_depth=4, subsample=0.8,
                                  colsample_bytree=0.8, eval_metric="logloss"),
    "LightGBM":     LGBMClassifier(n_estimators=300, learning_rate=0.05,
                                   num_leaves=31, verbose=-1),
}

for name, m in models.items():
    s = cross_val_score(m, X, y, cv=5, scoring="roc_auc")
    print(f"{name:14s} AUC = {s.mean():.4f} ± {s.std():.4f}")
