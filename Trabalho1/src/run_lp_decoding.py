"""LP com decodificação clássica (moda: a combinação mais provável) versus
decodificação marginal (soma das probabilidades das combinações que contêm cada
rótulo, limiar 0,5). Usa as mesmas dobras do experimento principal.
"""
import pickle
import warnings
from pathlib import Path

import numpy as np
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

from evaluation import all_metrics
from mlmethods import LabelPowerset
from run_main import N_FOLDS, SEED, load_data, make_base

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]


def main():
    X, Y, _ = load_data()
    cv = MultilabelStratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    out = []
    for f, (tr, te) in enumerate(cv.split(X, Y)):
        seen = {tuple(r) for r in Y[tr]}
        for bname in ["RF", "LR"]:
            lp = LabelPowerset(make_base(bname)).fit(X[tr], Y[tr])
            proba = lp.model_.predict_proba(X[te])
            sets = lp.labelsets_[lp.model_.classes_]
            P_marg = proba @ sets
            Y_mode = sets[np.argmax(proba, axis=1)]
            m_marg = all_metrics(Y[te], P_marg)
            # para a moda, o "escore" de cada rótulo é a própria decisão (AUC/RL degeneram)
            m_mode = all_metrics(Y[te], Y_mode.astype(float))
            nov_marg = np.mean([tuple(r) not in seen for r in (P_marg >= 0.5).astype(np.int8)])
            out.append({"fold": f, "base": bname, "marg": m_marg, "mode": m_mode,
                        "novel_marg": nov_marg, "novel_mode": 0.0,
                        "n_classes": len(lp.model_.classes_)})
            print(f"dobra {f} {bname}: moda subset={m_mode['subset_acc']:.3f} HL={m_mode['hamming_loss']:.3f} "
                  f"| marginal subset={m_marg['subset_acc']:.3f} HL={m_marg['hamming_loss']:.3f} "
                  f"novas={nov_marg:.2f}", flush=True)
    with open(ROOT / "results" / "lp_decoding.pkl", "wb") as fh:
        pickle.dump(out, fh)


if __name__ == "__main__":
    main()
