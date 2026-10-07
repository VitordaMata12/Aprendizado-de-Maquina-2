"""Experimento principal: comparação de métodos multirrótulo na base SIDER.

Validação cruzada estratificada iterativa (10 dobras). Para cada dobra e cada
classificador base (RF e LR), treina todos os métodos e salva métricas,
probabilidades e informações auxiliares em results/main/fold_XX_<base>.pkl.
O script é retomável: dobras já concluídas são puladas.

Uso: python run_main.py [RF|LR] [n_dobras]
"""
import pickle
import sys
import warnings
import time
from pathlib import Path

import numpy as np
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from evaluation import all_metrics, pairwise_diversity
from mlmethods import (AutoLR, Bagged, BinaryRelevance, ClassifierChain, LabelPowerset,
                       MLkNN, Prior, RAkEL, StackedBR)

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "main"
OUT.mkdir(parents=True, exist_ok=True)

N_FOLDS = 10
N_ORDERS = 10      # ordens aleatórias de CC avaliadas individualmente
N_MEMBERS = 10     # membros de EBR/ECC
SEED = 42


def load_data():
    D = np.load(ROOT / "data" / "features.npz")
    X = np.hstack([D["X_desc"], D["X_maccs"].astype(float)])
    return X, D["Y"].astype(np.int8), list(D["label_names"])


def make_base(name):
    if name == "RF":
        return RandomForestClassifier(n_estimators=100, n_jobs=2, random_state=0)
    return make_pipeline(StandardScaler(), AutoLR(C=0.05, max_iter=3000))


def run_fold(f, tr, te, X, Y, bname):
    path = OUT / f"fold_{f:02d}_{bname}.pkl"
    if path.exists():
        print(f"[{bname}] dobra {f} já existe, pulando", flush=True)
        return
    base = make_base(bname)
    Xtr, Ytr, Xte, Yte = X[tr], Y[tr], X[te], Y[te]
    out = {"fold": f, "base": bname, "test_idx": te, "metrics": {}, "proba": {}, "time": {}}

    def evaluate(name, model):
        t0 = time.time()
        model.fit(Xtr, Ytr)
        P = model.predict_proba(Xte)
        out["time"][name] = time.time() - t0
        out["metrics"][name] = all_metrics(Yte, P)
        out["proba"][name] = P.astype(np.float32)
        print(f"[{bname}] dobra {f} {name:10s} {out['time'][name]:7.1f}s "
              f"AUC={out['metrics'][name]['auc_macro']:.4f} "
              f"F1mi={out['metrics'][name]['f1_micro']:.4f}", flush=True)
        return model

    rng = np.random.default_rng(SEED + f)
    evaluate("Prior", Prior())
    evaluate("BR", BinaryRelevance(base))
    evaluate("LP", LabelPowerset(base))

    # CC: várias ordens aleatórias (mesmas ordens para RF e LR na mesma dobra)
    orders = [rng.permutation(Y.shape[1]) for _ in range(N_ORDERS)]
    cc_probas = []
    for i, order in enumerate(orders):
        m = ClassifierChain(base, order=order)
        t0 = time.time()
        m.fit(Xtr, Ytr)
        P = m.predict_proba(Xte)
        cc_probas.append(P)
        out["metrics"][f"CC_o{i}"] = all_metrics(Yte, P)
        out["time"][f"CC_o{i}"] = time.time() - t0
    cc_probas = np.stack(cc_probas)
    out["cc_orders"] = orders
    out["proba"]["CC"] = cc_probas[0].astype(np.float32)  # uma ordem "representativa"
    out["metrics"]["CC"] = out["metrics"]["CC_o0"]
    # ensemble das cadeias sem bagging (diversidade só pela ordem)
    Pens = cc_probas.mean(axis=0)
    out["metrics"]["ECC-nb"] = all_metrics(Yte, Pens)
    out["proba"]["ECC-nb"] = Pens.astype(np.float32)
    out["div"] = {"ECC-nb": pairwise_diversity((cc_probas >= 0.5).astype(np.int8), Yte)}
    print(f"[{bname}] dobra {f} CC x{N_ORDERS} ok; ECC-nb AUC={out['metrics']['ECC-nb']['auc_macro']:.4f}", flush=True)

    ebr = evaluate("EBR", Bagged(lambda s: BinaryRelevance(base), N_MEMBERS, random_state=SEED + f))
    out["div"]["EBR"] = pairwise_diversity((ebr.last_member_probas_ >= 0.5).astype(np.int8), Yte)
    out["member_metrics"] = {"EBR": [all_metrics(Yte, P) for P in ebr.last_member_probas_]}

    ecc = evaluate("ECC", Bagged(lambda s: ClassifierChain(base, random_state=s), N_MEMBERS,
                                 random_state=SEED + f))
    out["div"]["ECC"] = pairwise_diversity((ecc.last_member_probas_ >= 0.5).astype(np.int8), Yte)
    out["member_metrics"]["ECC"] = [all_metrics(Yte, P) for P in ecc.last_member_probas_]

    evaluate("RAkEL", RAkEL(base, k=3, n_partitions=6, random_state=SEED + f))
    stk = evaluate("Stacked-BR", StackedBR(base, inner_folds=5, random_state=SEED + f))
    out["meta_coef"] = stk.meta_coef()

    if bname == "RF":  # ML-kNN não depende do classificador base: roda uma vez
        scaler = StandardScaler().fit(Xtr)
        t0 = time.time()
        mk = MLkNN(k=10, s=1.0).fit(scaler.transform(Xtr), Ytr)
        P = mk.predict_proba(scaler.transform(Xte))
        out["time"]["ML-kNN"] = time.time() - t0
        out["metrics"]["ML-kNN"] = all_metrics(Yte, P)
        out["proba"]["ML-kNN"] = P.astype(np.float32)

    with open(path, "wb") as fh:
        pickle.dump(out, fh)


def main():
    bname = sys.argv[1] if len(sys.argv) > 1 else "RF"
    n_run = int(sys.argv[2]) if len(sys.argv) > 2 else N_FOLDS
    X, Y, _ = load_data()
    cv = MultilabelStratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for f, (tr, te) in enumerate(cv.split(X, Y)):
        if f >= n_run:
            break
        run_fold(f, tr, te, X, Y, bname)


if __name__ == "__main__":
    main()
