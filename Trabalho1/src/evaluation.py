"""Métricas multirrótulo e utilidades de avaliação."""
import numpy as np
from scipy import stats
from sklearn.metrics import (f1_score, hamming_loss, jaccard_score,
                             label_ranking_loss, roc_auc_score)

METRICS = {
    # nome: (função, maior_é_melhor)
    "hamming_loss": (lambda Y, Yp, P: hamming_loss(Y, Yp), False),
    "subset_acc": (lambda Y, Yp, P: float(np.mean(np.all(Y == Yp, axis=1))), True),
    "jaccard": (lambda Y, Yp, P: jaccard_score(Y, Yp, average="samples", zero_division=1.0), True),
    "f1_micro": (lambda Y, Yp, P: f1_score(Y, Yp, average="micro", zero_division=0), True),
    "f1_macro": (lambda Y, Yp, P: f1_score(Y, Yp, average="macro", zero_division=0), True),
    "auc_macro": (lambda Y, Yp, P: macro_auc(Y, P), True),
    "ranking_loss": (lambda Y, Yp, P: label_ranking_loss(Y, P), False),
}


def macro_auc(Y, P):
    """AUC-ROC média por rótulo, ignorando rótulos sem as duas classes no teste."""
    vals = [roc_auc_score(Y[:, j], P[:, j]) for j in range(Y.shape[1])
            if 0 < Y[:, j].sum() < Y.shape[0]]
    return float(np.mean(vals))


def all_metrics(Y, P, threshold=0.5):
    Yp = (P >= threshold).astype(np.int8)
    return {name: fn(Y, Yp, P) for name, (fn, _) in METRICS.items()}


def corrected_ttest(a, b, n_train, n_test):
    """t-teste pareado com correção de Nadeau & Bengio (2003) para validação cruzada.

    a, b: valores da métrica por dobra para os dois métodos.
    Retorna (estatística t, p-valor bicaudal).
    """
    d = np.asarray(a) - np.asarray(b)
    k = len(d)
    var = np.var(d, ddof=1)
    if var == 0:
        return 0.0, 1.0
    t = d.mean() / np.sqrt((1 / k + n_test / n_train) * var)
    p = 2 * stats.t.sf(np.abs(t), df=k - 1)
    return float(t), float(p)


def holm(pvals):
    """Correção de Holm-Bonferroni para comparações múltiplas."""
    pvals = np.asarray(pvals)
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj


def pairwise_diversity(member_preds, Y):
    """Diversidade entre membros de um ensemble, calculada por rótulo.

    member_preds: (M, n, L) predições binárias dos membros; Y: (n, L) verdade.
    Para cada par de membros e cada rótulo, monta a tabela de concordância
    (a = ambos acertam, b = só o 1º acerta, c = só o 2º acerta, d = ambos erram)
    e calcula a medida de desacordo (b + c) / N e a estatística Q de Yule
    Q = (ad - bc) / (ad + bc). Retorna as médias sobre pares e rótulos.
    """
    M = member_preds.shape[0]
    correct = member_preds == Y[None]
    dis, qs = [], []
    for i in range(M):
        for j in range(i + 1, M):
            ci, cj = correct[i], correct[j]
            a = (ci & cj).sum(axis=0)
            b = (ci & ~cj).sum(axis=0)
            c = (~ci & cj).sum(axis=0)
            d = (~ci & ~cj).sum(axis=0)
            dis.append(((b + c) / Y.shape[0]).mean())
            den = a * d + b * c
            q = np.where(den > 0, (a * d - b * c) / np.maximum(den, 1), 1.0)
            qs.append(q.mean())
    return float(np.mean(dis)), float(np.mean(qs))
