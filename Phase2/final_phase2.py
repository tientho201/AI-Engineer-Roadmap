"""
churn_pipeline.py — Dự án tổng hợp Phase 2
Dataset gợi ý: Telco Customer Churn (Kaggle)
"""
import numpy as np, pandas as pd, optuna
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import (classification_report, roc_auc_score,
                             average_precision_score, precision_recall_curve)
from lightgbm import LGBMClassifier


optuna.logging.set_verbosity(optuna.logging.WARNING)


def build_pipeline(num_cols, cat_cols, **params):
    prep = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), num_cols),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore",
                                               sparse_output=False))]), cat_cols),
    ])
    return Pipeline([("prep", prep),
                     ("clf", LGBMClassifier(verbose=-1, **params))])


def tune(X, y, num_cols, cat_cols, n_trials=40):
    cv = StratifiedKFold(5, shuffle=True, random_state=42)

    def objective(trial):
        p = {
            "n_estimators":  trial.suggest_int("n_estimators", 200, 800, step=100),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves":    trial.suggest_int("num_leaves", 8, 64, log=True),
            "subsample":     trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 5.0),
        }
        return cross_val_score(build_pipeline(num_cols, cat_cols, **p),
                               X, y, cv=cv, scoring="average_precision").mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    return study.best_params


def pick_threshold(y_true, proba, min_recall=0.75):
    """Chọn ngưỡng đảm bảo bắt được min_recall khách rời bỏ."""
    prec, rec, thr = precision_recall_curve(y_true, proba)
    ok = np.where(rec[:-1] >= min_recall)[0]
    i = ok[-1]
    return float(thr[i]), float(prec[i]), float(rec[i])


if __name__ == "__main__":
    df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
    y = (df.pop("Churn") == "Yes").astype(int)
    num_cols = df.select_dtypes("number").columns.tolist()
    cat_cols = df.select_dtypes("object")
    print(cat_cols.dtypes)
    # best = tune(df, y, num_cols, cat_cols)
    # print(best)