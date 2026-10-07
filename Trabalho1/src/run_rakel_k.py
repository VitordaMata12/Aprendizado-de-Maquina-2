"""Sensibilidade do RAkEL ao tamanho k dos conjuntos de rótulos.

k = 1 reproduz o BR e k = L (27) reproduz o LP; valores intermediários
interpolam entre independência total e modelagem conjunta de todos os rótulos.
Usa as mesmas 10 dobras do experimento principal e Random Forest como base.
"""
import pickle
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from evaluation import all_metrics
from mlmethods import AutoLR, LabelPowerset, random_labelsets
from run_main import N_FOLDS, SEED, load_data

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "rakel_k"
OUT.mkdir(parents=True, exist_ok=True)
KS = [1, 2, 3, 4, 6, 9, 14, 27]


def rakel_stream(base, k, n_partitions, seed, Xtr, Ytr, Xte):
    """Mesmo procedimento da classe RAkEL (mesmos conjuntos, mesma regra da média),
    mas cada LP é treinado, usado no teste e descartado em seguida, para não manter
    na memória dezenas de florestas com centenas de classes (k grande)."""
    rng = np.random.default_rng(seed)
    L = Ytr.shape[1]
    sets = random_labelsets(L, k, n_partitions, rng)
    keys = {}
    for s in sets:
        key = tuple(sorted(s))
        keys[key] = keys.get(key, 0) + 1
    S = np.zeros((Xte.shape[0], L))
    C = np.zeros(L)
    for key, mult in keys.items():
        idx = list(key)
        P = LabelPowerset(base).fit(Xtr, Ytr[:, idx]).predict_proba(Xte)
        S[:, idx] += mult * P
        C[idx] += mult
    return S / C, len(keys)


def main():
    bname = sys.argv[1] if len(sys.argv) > 1 else "RF"
    ks = KS if bname == "RF" else [1, 2, 3, 4, 6, 9]  # LP multinomial com LR fica caro para k grande
    X, Y, _ = load_data()
    cv = MultilabelStratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    splits = list(enumerate(cv.split(X, Y)))
    if len(sys.argv) > 2 and sys.argv[2] == "rev":  # permite rodar dois processos em paralelo
        splits = splits[::-1]
    for f, (tr, te) in splits:
        path = OUT / f"fold_{f:02d}_{bname}.pkl"
        if path.exists():
            continue
        base = (RandomForestClassifier(n_estimators=100, n_jobs=2, random_state=0) if bname == "RF"
                else make_pipeline(StandardScaler(), AutoLR(C=0.05, max_iter=3000)))
        res = {}
        for k in ks:
            t0 = time.time()
            P, n_models = rakel_stream(base, k, 6, SEED + f, X[tr], Y[tr], X[te])
            res[k] = {**all_metrics(Y[te], P), "n_models": n_models, "time": time.time() - t0}
            print(f"[{bname}] dobra {f} k={k:2d} {res[k]['time']:6.1f}s AUC={res[k]['auc_macro']:.4f} "
                  f"HL={res[k]['hamming_loss']:.4f}", flush=True)
        with open(path, "wb") as fh:
            pickle.dump(res, fh)


if __name__ == "__main__":
    main()
