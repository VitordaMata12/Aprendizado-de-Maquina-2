"""Análise exploratória da base SIDER: estatísticas multirrótulo e figuras."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from matplotlib.colors import LinearSegmentedColormap
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform

import style

ROOT = Path(__file__).resolve().parents[1]
style.setup()
D = np.load(ROOT / "data" / "features.npz")
Y = D["Y"].astype(int)
names = [style.LABELS_PT[n] for n in D["label_names"]]
n, L = Y.shape

# ---------------- estatísticas ----------------
card = Y.sum(1)
counts = Y.sum(0)
keys, inv, freq = np.unique(Y, axis=0, return_inverse=True, return_counts=True)
irlbl = counts.max() / counts
phi = np.corrcoef(Y.T)
off = phi[np.triu_indices(L, 1)]
ev = np.linalg.svd(Y - Y.mean(0), compute_uv=False) ** 2
# fração de exemplos de teste cujo conjunto de rótulos não aparece no treino (10 dobras)
cv = MultilabelStratifiedKFold(n_splits=10, shuffle=True, random_state=42)
X = np.hstack([D["X_desc"], D["X_maccs"]])
unseen = []
for tr, te in cv.split(X, Y):
    seen = {tuple(r) for r in Y[tr]}
    unseen.append(np.mean([tuple(r) not in seen for r in Y[te]]))
# correlação entre cardinalidade e cada rótulo
corr_card = [np.corrcoef(card, Y[:, j])[0, 1] for j in range(L)]

stats = {
    "n_exemplos": int(n), "n_rotulos": int(L),
    "n_atributos": int(X.shape[1]), "n_desc": int(D["X_desc"].shape[1]),
    "n_maccs": int(D["X_maccs"].shape[1]),
    "cardinalidade": float(card.mean()), "cardinalidade_dp": float(card.std()),
    "densidade": float(Y.mean()),
    "card_min": int(card.min()), "card_max": int(card.max()),
    "conjuntos_distintos": int(len(keys)),
    "conjuntos_unicos_freq1": int((freq == 1).sum()),
    "exemplos_conj_unico": float((freq[inv.ravel()] == 1).mean()),
    "conjunto_mais_freq": int(freq.max()),
    "MeanIR": float(irlbl.mean()), "MaxIR": float(irlbl.max()),
    "prev_min": float(Y.mean(0).min()), "prev_max": float(Y.mean(0).max()),
    "rotulos_prev_maior_50": int((Y.mean(0) > 0.5).sum()),
    "phi_medio": float(off.mean()), "phi_frac_pos": float((off > 0).mean()),
    "phi_max": float(off.max()),
    "pc1_var": float(ev[0] / ev.sum()),
    "teste_conj_nao_visto_media": float(np.mean(unseen)),
    "corr_card_media": float(np.mean(corr_card)),
    "n_sais": int((D["n_frag"] > 1).sum()),
}
(ROOT / "results").mkdir(exist_ok=True)
with open(ROOT / "results" / "eda_stats.json", "w") as fh:
    json.dump(stats, fh, indent=2, ensure_ascii=False)
for k, v in stats.items():
    print(f"{k:28s} {v}")

# ---------------- Figura 1: prevalência + cardinalidade ----------------
order = np.argsort(Y.mean(0))
fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.3, 3.3), gridspec_kw={"width_ratios": [1.35, 1]})
prev = Y.mean(0)[order]
a1.barh(np.arange(L), prev, height=0.72, color=style.BLUE)
a1.set_yticks(np.arange(L))
a1.set_yticklabels([names[i] for i in order], fontsize=6.3)
a1.set_xlim(0, 1)
a1.axvline(0.5, color=style.GRAY, lw=0.6, ls=(0, (3, 2)))
a1.set_xlabel("Prevalência (fração de fármacos)")
a1.set_title("(a) Prevalência de cada rótulo", loc="left")
style.comma_axes(a1, which="x")
a1.tick_params(axis="y", length=0)
a1.set_ylim(-0.7, L - 0.3)
bins = np.arange(0.5, L + 1.5, 1)
a2.hist(card, bins=bins, color=style.BLUE, edgecolor="white", linewidth=0.6)
a2.axvline(card.mean(), color=style.ORANGE, lw=1.2)
a2.text(card.mean() - 0.6, a2.get_ylim()[1] * 0.93, f"média = {card.mean():.1f}".replace(".", ","),
        color=style.INK, fontsize=7, ha="right")
a2.set_xlabel("Nº de rótulos por fármaco")
a2.set_ylabel("Nº de fármacos")
a2.set_title("(b) Cardinalidade dos exemplos", loc="left")
fig.tight_layout(w_pad=1.2)
fig.savefig(ROOT / "figures" / "fig_eda.pdf")
fig.savefig(ROOT / "figures" / "fig_eda.png", dpi=220)

# ---------------- Figura 2: correlação phi entre rótulos ----------------
dist = 1 - phi
np.fill_diagonal(dist, 0)
Z = linkage(squareform(dist, checks=False), "average")
leaf = leaves_list(Z)
# todas as correlações são positivas: escala sequencial (um matiz, claro -> escuro)
cmap = LinearSegmentedColormap.from_list(
    "seq", ["#f7f6f3", "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"])
print("phi mínimo fora da diagonal:", off.min())
fig, ax = plt.subplots(figsize=(4.4, 3.8))
P = phi[np.ix_(leaf, leaf)].copy()
np.fill_diagonal(P, np.nan)
im = ax.imshow(P, cmap=cmap, vmin=0, vmax=0.55)
ax.set_xticks(range(L))
ax.set_yticks(range(L))
ax.set_xticklabels([names[i] for i in leaf], rotation=90, fontsize=5.6)
ax.set_yticklabels([names[i] for i in leaf], fontsize=5.6)
ax.tick_params(length=0)
for s in ax.spines.values():
    s.set_visible(False)
cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
cb.set_label("Coeficiente $\\phi$", fontsize=7)
cb.ax.tick_params(labelsize=6)
cb.outline.set_visible(False)
style.comma_axes(cb.ax, which="y")
fig.tight_layout()
fig.savefig(ROOT / "figures" / "fig_phi.pdf")
fig.savefig(ROOT / "figures" / "fig_phi.png", dpi=220)
print("figuras salvas")
