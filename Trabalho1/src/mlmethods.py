"""Métodos de classificação multirrótulo usados no trabalho.

Todos os métodos seguem a mesma interface:
    fit(X, Y)            -> self
    predict_proba(X)     -> matriz (n, L) com o escore/probabilidade de cada rótulo
    predict(X)           -> matriz binária (n, L) com limiar 0,5

Transformação do problema : BR, LP, CC
Adaptação de algoritmo    : MLkNN
Ensembles / combinação    : EBR, ECC, RAkEL, BR empilhado (stacking)
Linha de base             : Prior (prediz a prevalência de cada rótulo)
"""
import numpy as np
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.neighbors import NearestNeighbors


class AutoLR(LogisticRegression):
    """Regressão logística L2 cujo otimizador depende do nº de classes.

    O problema de otimização é o mesmo (convexo); muda só o solver: Newton-Cholesky
    nos problemas binários (BR, CC, ECC...) e L-BFGS nos multiclasse (LP, RAkEL).
    """

    def fit(self, X, y, sample_weight=None):
        self.solver = "newton-cholesky" if len(np.unique(y)) <= 2 else "lbfgs"
        return super().fit(X, y, sample_weight)


def _pos_proba(model, X):
    """Probabilidade da classe positiva, mesmo se o treino só viu uma classe."""
    proba = model.predict_proba(X)
    classes = list(model.classes_)
    if 1 in classes:
        return proba[:, classes.index(1)]
    return np.zeros(X.shape[0])


class MLBase:
    threshold = 0.5

    def predict(self, X):
        return (self.predict_proba(X) >= self.threshold).astype(np.int8)


class Prior(MLBase):
    """Linha de base: cada rótulo recebe sua prevalência no treino."""

    def fit(self, X, Y):
        self.p_ = Y.mean(axis=0)
        return self

    def predict_proba(self, X):
        return np.tile(self.p_, (X.shape[0], 1))


class BinaryRelevance(MLBase):
    """BR: um classificador binário independente por rótulo."""

    def __init__(self, base):
        self.base = base

    def fit(self, X, Y):
        self.models_ = [clone(self.base).fit(X, Y[:, j]) for j in range(Y.shape[1])]
        return self

    def predict_proba(self, X):
        return np.column_stack([_pos_proba(m, X) for m in self.models_])


class LabelPowerset(MLBase):
    """LP: cada combinação de rótulos vira uma classe de um problema multiclasse.

    O escore de cada rótulo é a probabilidade marginal: soma das probabilidades
    das combinações (classes) que contêm o rótulo.
    """

    def __init__(self, base):
        self.base = base

    def fit(self, X, Y):
        keys, y_cls = np.unique(Y, axis=0, return_inverse=True)
        self.labelsets_ = keys  # (n_classes, L)
        self.model_ = clone(self.base).fit(X, y_cls.ravel())
        return self

    def predict_proba(self, X):
        proba = self.model_.predict_proba(X)  # (n, classes vistas)
        sets = self.labelsets_[self.model_.classes_]
        return proba @ sets


class ClassifierChain(MLBase):
    """CC: classificadores binários encadeados segundo uma ordem dos rótulos.

    No treino, o j-ésimo classificador recebe X mais os rótulos verdadeiros dos
    j-1 anteriores; na predição, recebe as predições (0/1) dos anteriores.
    """

    def __init__(self, base, order=None, random_state=None):
        self.base = base
        self.order = order
        self.random_state = random_state

    def fit(self, X, Y):
        L = Y.shape[1]
        rng = np.random.default_rng(self.random_state)
        self.order_ = np.array(self.order) if self.order is not None else rng.permutation(L)
        self.models_ = []
        Xa = X
        for j in self.order_:
            self.models_.append(clone(self.base).fit(Xa, Y[:, j]))
            Xa = np.hstack([Xa, Y[:, [j]]])
        return self

    def predict_proba(self, X):
        n, L = X.shape[0], len(self.order_)
        P = np.zeros((n, L))
        Xa = X
        for m, j in zip(self.models_, self.order_):
            p = _pos_proba(m, Xa)
            P[:, j] = p
            Xa = np.hstack([Xa, (p >= 0.5).astype(X.dtype)[:, None]])
        return P


class Bagged(MLBase):
    """Ensemble homogêneo por bagging de um método multirrótulo (EBR, ECC...).

    Cada membro é treinado numa amostra bootstrap; a combinação usa a regra da
    média das probabilidades (Kittler et al., 1998). Guarda as predições de cada
    membro para a análise de diversidade.
    """

    def __init__(self, make_member, n_members=10, random_state=0, bootstrap=True):
        self.make_member = make_member
        self.n_members = n_members
        self.random_state = random_state
        self.bootstrap = bootstrap

    def fit(self, X, Y):
        rng = np.random.default_rng(self.random_state)
        n = X.shape[0]
        self.members_ = []
        for i in range(self.n_members):
            idx = rng.integers(0, n, n) if self.bootstrap else np.arange(n)
            member = self.make_member(int(rng.integers(1 << 30)))
            self.members_.append(member.fit(X[idx], Y[idx]))
        return self

    def member_probas(self, X):
        return np.stack([m.predict_proba(X) for m in self.members_])  # (M, n, L)

    def predict_proba(self, X):
        self.last_member_probas_ = self.member_probas(X)
        return self.last_member_probas_.mean(axis=0)


def random_labelsets(L, k, n_partitions, rng):
    """Gera conjuntos de k rótulos a partir de partições aleatórias dos L rótulos.

    Cada partição cobre todos os rótulos uma vez; com n_partitions partições,
    cada rótulo aparece exatamente n_partitions vezes no ensemble.
    """
    sets = []
    for _ in range(n_partitions):
        perm = rng.permutation(L)
        sets.extend(perm[i:i + k] for i in range(0, L, k))
    return sets


class RAkEL(MLBase):
    """RAkEL (RAndom k-labELsets): ensemble de LPs sobre subconjuntos de k rótulos.

    k = 1 equivale a um ensemble de BRs; k = L equivale a um ensemble de LPs.
    O escore de cada rótulo é a média dos escores marginais dados pelos LPs que
    o contêm (regra da média).
    """

    def __init__(self, base, k=3, n_partitions=6, random_state=0):
        self.base = base
        self.k = k
        self.n_partitions = n_partitions
        self.random_state = random_state

    def fit(self, X, Y):
        rng = np.random.default_rng(self.random_state)
        self.L_ = Y.shape[1]
        self.sets_ = random_labelsets(self.L_, self.k, self.n_partitions, rng)
        # conjuntos repetidos (ex.: k = L) reaproveitam o mesmo modelo: com classificador
        # base determinístico o resultado é idêntico e o custo cai muito
        cache = {}
        self.models_ = []
        for s in self.sets_:
            key = tuple(sorted(s))
            if key not in cache:
                cache[key] = (np.array(key), LabelPowerset(self.base).fit(X, Y[:, list(key)]))
            self.models_.append(cache[key])
        self.sets_ = [m[0] for m in self.models_]
        self.models_ = [m[1] for m in self.models_]
        return self

    def predict_proba(self, X):
        S = np.zeros((X.shape[0], self.L_))
        C = np.zeros(self.L_)
        for s, m in zip(self.sets_, self.models_):
            S[:, s] += m.predict_proba(X)
            C[s] += 1
        return S / C


class StackedBR(MLBase):
    """BR empilhado (2BR / generalização por pilha aplicada ao multirrótulo).

    Nível 0: BR com o classificador base; suas probabilidades fora-da-dobra
    (validação cruzada interna) formam os meta-atributos.
    Nível 1: para cada rótulo, uma regressão logística que recebe os escores de
    TODOS os rótulos, aprendendo assim as dependências entre eles.
    """

    def __init__(self, base, inner_folds=5, C=1.0, random_state=0):
        self.base = base
        self.inner_folds = inner_folds
        self.C = C
        self.random_state = random_state

    @staticmethod
    def _logit(P):
        P = np.clip(P, 1e-3, 1 - 1e-3)
        return np.log(P / (1 - P))

    def fit(self, X, Y):
        n, L = Y.shape
        Z = np.zeros((n, L))
        kf = KFold(self.inner_folds, shuffle=True, random_state=self.random_state)
        for tr, va in kf.split(X):
            Z[va] = BinaryRelevance(self.base).fit(X[tr], Y[tr]).predict_proba(X[va])
        self.level0_ = BinaryRelevance(self.base).fit(X, Y)
        Zl = self._logit(Z)
        self.meta_ = [LogisticRegression(C=self.C, max_iter=2000).fit(Zl, Y[:, j]) for j in range(L)]
        return self

    def meta_coef(self):
        """Matriz (L, L): peso do escore do rótulo i (coluna) na predição do rótulo j (linha)."""
        return np.vstack([m.coef_[0] for m in self.meta_])

    def predict_proba(self, X):
        Zl = self._logit(self.level0_.predict_proba(X))
        return np.column_stack([_pos_proba(m, Zl) for m in self.meta_])


class MLkNN(MLBase):
    """ML-kNN (Zhang & Zhou, 2007): kNN + regra de Bayes com estimativa MAP.

    Para cada rótulo, estima a priori P(H_j) e as verossimilhanças
    P(C_j = c | H_j) de c vizinhos (dentre k) possuírem o rótulo, com
    suavização de Laplace s. A saída é a posteriori P(H_j | C_j).
    """

    def __init__(self, k=10, s=1.0, metric="euclidean"):
        self.k = k
        self.s = s
        self.metric = metric

    def fit(self, X, Y):
        n, L = Y.shape
        k, s = self.k, self.s
        self.Y_ = Y
        self.nn_ = NearestNeighbors(n_neighbors=k + 1, metric=self.metric).fit(X)
        idx = self.nn_.kneighbors(X, return_distance=False)[:, 1:]  # exclui o próprio ponto
        counts = Y[idx].sum(axis=1)  # (n, L) nº de vizinhos com o rótulo
        self.prior_ = (s + Y.sum(axis=0)) / (2 * s + n)
        self.cond_pos_ = np.zeros((L, k + 1))
        self.cond_neg_ = np.zeros((L, k + 1))
        for j in range(L):
            pos = Y[:, j] == 1
            c1 = np.bincount(counts[pos, j], minlength=k + 1)
            c0 = np.bincount(counts[~pos, j], minlength=k + 1)
            self.cond_pos_[j] = (s + c1) / (s * (k + 1) + c1.sum())
            self.cond_neg_[j] = (s + c0) / (s * (k + 1) + c0.sum())
        return self

    def predict_proba(self, X):
        idx = self.nn_.kneighbors(X, n_neighbors=self.k, return_distance=False)
        counts = self.Y_[idx].sum(axis=1)
        L = self.Y_.shape[1]
        P = np.zeros(counts.shape)
        for j in range(L):
            a = self.prior_[j] * self.cond_pos_[j, counts[:, j]]
            b = (1 - self.prior_[j]) * self.cond_neg_[j, counts[:, j]]
            P[:, j] = a / (a + b)
        return P
