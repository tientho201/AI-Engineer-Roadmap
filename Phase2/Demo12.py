import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE

X, y = make_classification(n_samples=10000, weights=[.98, .02],
                           n_features=20, random_state=0)
Xtr, Xte, ytr, yte = train_test_split(X, y, stratify=y, test_size=.3, random_state=0)
ratio = (ytr == 0).sum() / (ytr == 1).sum()

def ap(model, Xt=Xtr, yt=ytr):
    model.fit(Xt, yt)
    return average_precision_score(yte, model.predict_proba(Xte)[:, 1])

print("1. Không làm gì        :", round(ap(LGBMClassifier(verbose=-1)), 4))
print("2. scale_pos_weight    :", round(ap(LGBMClassifier(scale_pos_weight=ratio, verbose=-1)), 4))
print("3. class_weight balanced:", round(ap(LGBMClassifier(class_weight="balanced", verbose=-1)), 4))

Xs, ys = SMOTE(random_state=0).fit_resample(Xtr, ytr)   # CHỈ oversample trên TRAIN
print("4. SMOTE               :", round(ap(LGBMClassifier(verbose=-1), Xs, ys), 4))