import optuna
from optuna.samplers import TPESampler
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score, StratifiedKFold
from lightgbm import LGBMClassifier

optuna.logging.set_verbosity(optuna.logging.WARNING)
X, y = load_breast_cancer(return_X_y=True)
cv = StratifiedKFold(5, shuffle=True, random_state=0)


def objective(trial):
    params = {
        "n_estimators":     trial.suggest_int("n_estimators", 100, 1000, step=50),
        "learning_rate":    trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "num_leaves":       trial.suggest_int("num_leaves", 8, 128, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_lambda":       trial.suggest_float("reg_lambda", 1e-8, 10, log=True),
        "verbose": -1,
    }
    return cross_val_score(LGBMClassifier(**params), X, y,
                           cv=cv, scoring="roc_auc").mean()


study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=42))
study.optimize(objective, n_trials=50, show_progress_bar=False)

print("AUC tốt nhất:", round(study.best_value, 5))
for k, v in study.best_params.items():
    print(f"  {k}: {v}")

# Feature nào quan trọng nhất trong việc tune?
print(optuna.importance.get_param_importances(study))
