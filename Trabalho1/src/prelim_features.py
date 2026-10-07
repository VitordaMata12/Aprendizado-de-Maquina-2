"""Experimento preliminar: escolha da representação molecular com BR.

Compara conjuntos de atributos usando BR + Random Forest e BR + Regressão
Logística em validação cruzada estratificada iterativa de 5 dobras.
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from evaluation import all_metrics
from mlmethods import BinaryRelevance

ROOT = Path(__file__).resolve().parents[1]
D = np.load(ROOT / "data" / "features.npz")
Y = D["Y"]
reps = {
    "Descritores": D["X_desc"],
    "MACCS": D["X_maccs"].astype(float),
    "Morgan": D["X_morgan"].astype(float),
    "Descritores+MACCS": np.hstack([D["X_desc"], D["X_maccs"]]),
    "Descritores+Morgan": np.hstack([D["X_desc"], D["X_morgan"]]),
}
bases = {
    "RF": RandomForestClassifier(n_estimators=150, n_jobs=2, random_state=0),
    "LR": make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=3000)),
}
cv = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=42)
rows = []
for rname, X in reps.items():
    for bname, base in bases.items():
        t0 = time.time()
        for f, (tr, te) in enumerate(cv.split(X, Y)):
            P = BinaryRelevance(base).fit(X[tr], Y[tr]).predict_proba(X[te])
            rows.append({"rep": rname, "base": bname, "fold": f, **all_metrics(Y[te], P)})
        print(f"{rname:20s} {bname}  {time.time() - t0:6.1f}s", flush=True)
res = pd.DataFrame(rows)
res.to_csv(ROOT / "results" / "prelim_features.csv", index=False)
summary = res.groupby(["rep", "base"])[["auc_macro", "f1_micro", "f1_macro", "hamming_loss", "ranking_loss"]].mean()
print(summary.round(4).to_string())
