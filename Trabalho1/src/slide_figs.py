"""Versões das figuras para os slides (fontes grandes, fundo da apresentação)."""
import json
import pickle
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform

import style

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures" / "slides"
OUT.mkdir(parents=True, exist_ok=True)
BG = "#F7F5F0"
INK = "#14233A"
INK2 = "#3E4A5A"
BLUE = "#2A78D6"
ORANGE = "#D2571F"

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Liberation Sans", "DejaVu Sans"],
    "font.size": 22, "axes.titlesize": 24, "axes.labelsize": 22,
    "xtick.labelsize": 20, "ytick.labelsize": 20, "legend.fontsize": 20,
    "axes.edgecolor": "#B9B6AD", "axes.linewidth": 1.0,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "legend.frameon": False, "grid.color": "#E2DED4", "grid.linewidth": 1.0,
})

D = np.load(ROOT / "data" / "features.npz")
Y = D["Y"].astype(int)
names = [style.LABELS_PT[n] for n in D["label_names"]]
L = Y.shape[1]


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=100)
    plt.close(fig)


def fig_prevalence():
    order = np.argsort(Y.mean(0))
    prev = Y.mean(0)[order]
    fig, ax = plt.subplots(figsize=(10.4, 7.6))
    colors = [ORANGE if p < 0.5 else BLUE for p in prev]
    ax.barh(np.arange(L), prev, height=0.72, color=colors)
    ax.set_yticks(np.arange(L))
    ax.set_yticklabels([names[i] for i in order], fontsize=17)
    ax.set_xlim(0, 1)
    ax.axvline(0.5, color="#8A8985", lw=1.4, ls=(0, (4, 3)))
    ax.set_xlabel("prevalência (fração dos 1.427 fármacos)")
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(-0.7, L - 0.3)
    ax.text(0.515, 1.2, "50%", color=INK2, fontsize=19)
    style.comma_axes(ax, which="x")
    fig.tight_layout()
    save(fig, "prevalencia")


def fig_cardinality():
    card = Y.sum(1)
    fig, ax = plt.subplots(figsize=(6.0, 7.6))
    ax.hist(card, bins=np.arange(0.5, L + 1.5, 1), color=BLUE, edgecolor=BG, linewidth=1.2)
    ax.axvline(card.mean(), color=ORANGE, lw=2.4)
    ax.text(card.mean() - 0.8, ax.get_ylim()[1] * 0.95, "média 15,3", color=INK, fontsize=19, ha="right")
    ax.set_xlabel("rótulos por fármaco")
    ax.set_ylabel("nº de fármacos")
    fig.tight_layout()
    save(fig, "cardinalidade")


def fig_phi():
    phi = np.corrcoef(Y.T)
    dist = 1 - phi
    np.fill_diagonal(dist, 0)
    leaf = leaves_list(linkage(squareform(dist, checks=False), "average"))
    seq = LinearSegmentedColormap.from_list(
        "seq", ["#F3F1EB", "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"])
    P = phi[np.ix_(leaf, leaf)].copy()
    np.fill_diagonal(P, np.nan)
    fig, ax = plt.subplots(figsize=(9.6, 8.2))
    im = ax.imshow(P, cmap=seq, vmin=0, vmax=0.55)
    ax.set_xticks(range(L))
    ax.set_yticks(range(L))
    ax.set_xticklabels([names[i] for i in leaf], rotation=90, fontsize=11.5)
    ax.set_yticklabels([names[i] for i in leaf], fontsize=11.5)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    cb.set_label("correlação φ", fontsize=18)
    cb.ax.tick_params(labelsize=15)
    cb.outline.set_visible(False)
    style.comma_axes(cb.ax, which="y")
    fig.tight_layout()
    save(fig, "phi")


METHOD_LABEL = {"LP": "LP", "CC": "CC (1 ordem)", "ECC-nb": "CC×10", "EBR": "EBR", "ECC": "ECC",
                "RAkEL": "RAkEL (k=3)", "Stacked-BR": "BR empilhado", "ML-kNN": "ML-kNN"}


def fig_delta_methods():
    """Diferença para o BR (mesmo classificador base): direita = melhor nas duas métricas."""
    d = pd.read_csv(ROOT / "results" / "metrics_by_fold.csv")
    sig = pd.read_csv(ROOT / "results" / "significance_vs_BR.csv")
    means = d.groupby(["base", "method"]).mean(numeric_only=True)
    methods = list(METHOD_LABEL)
    fig, axes = plt.subplots(1, 2, figsize=(16.0, 6.6), sharey=True)
    for ax, (met, lab, sign) in zip(axes, [("auc_macro", "ganho de AUC-ROC macro sobre o BR", 1),
                                            ("ranking_loss", "redução da ranking loss sobre o BR", -1)]):
        ax.axvline(0, color="#8A8985", lw=1.4, ls=(0, (4, 3)))
        for base, color, dy, name in [("RF", BLUE, -0.17, "Random Forest"), ("LR", ORANGE, 0.17, "Regressão Logística")]:
            for i, m in enumerate(methods):
                if (base, m) not in means.index:
                    continue
                v = sign * (means.loc[(base, m), met] - means.loc[(base, "BR"), met])
                row = sig[(sig.base == base) & (sig.method == m) & (sig.metric == met)]
                filled = len(row) and row.p_holm.values[0] < 0.05
                ax.plot([0, v], [i + dy, i + dy], color=color, lw=2, alpha=0.5)
                ax.plot(v, i + dy, "o", ms=13, color=color, mfc=color if filled else BG, mew=2.5,
                        label=name if i == 0 else None)
        ax.set_yticks(range(len(methods)))
        ax.set_yticklabels([METHOD_LABEL[m] for m in methods])
        ax.set_title(lab, loc="left", color=INK)
        ax.grid(axis="x")
        ax.tick_params(axis="y", length=0)
        style.comma_axes(ax, which="x")
    axes[0].invert_yaxis()  # eixo y compartilhado: inverter uma única vez
    h, l = axes[0].get_legend_handles_labels()
    from matplotlib.lines import Line2D
    h += [Line2D([], [], marker="o", ls="", ms=13, color=INK2, mfc=INK2),
          Line2D([], [], marker="o", ls="", ms=13, color=INK2, mfc=BG, mew=2.5)]
    l += ["significativo (Holm)", "não significativo"]
    fig.legend(h, l, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.9), w_pad=3)
    save(fig, "delta_metodos")


def fig_rakel():
    df = pd.read_csv(ROOT / "results" / "rakel_k_by_fold.csv")
    ks = np.array(sorted(df.k.unique()))
    nf = df.fold.nunique()
    fig, axes = plt.subplots(1, 2, figsize=(16.0, 6.0))
    for ax, (met, lab, sign) in zip(axes, [("auc_macro", "ganho de AUC-ROC macro sobre o BR (k = 1)", 1),
                                           ("ranking_loss", "redução da ranking loss sobre o BR (k = 1)", -1)]):
        piv = df.pivot(index="fold", columns="k", values=met)
        delta = sign * piv.sub(piv[1], axis=0)
        m = delta.mean().values
        se = delta.std(ddof=1).values / np.sqrt(nf)
        ax.axhline(0, color="#8A8985", lw=1.4, ls=(0, (4, 3)))
        ax.fill_between(ks, m - se, m + se, color=BLUE, alpha=0.18, lw=0)
        ax.plot(ks, m, color=BLUE, lw=3, marker="o", ms=10)
        ib = int(np.argmax(m))
        ax.plot(ks[ib], m[ib], marker="o", ms=20, mfc="none", mec=ORANGE, mew=3)
        ax.set_xscale("log")
        ax.set_xticks(ks)
        ax.set_xticklabels([str(k) for k in ks])
        ax.minorticks_off()
        ax.set_xlabel("tamanho k dos subconjuntos de rótulos")
        ax.set_title(lab, loc="left", color=INK)
        ax.grid(axis="y")
        ax.annotate("BR", (ks[0], m[0]), textcoords="offset points", xytext=(8, -28), fontsize=20, color=INK)
        ax.annotate("LP", (ks[-1], m[-1]), textcoords="offset points", xytext=(-36, -30), fontsize=20, color=INK)
        style.comma_axes(ax, which="y")
    fig.tight_layout(w_pad=3)
    save(fig, "rakel_k")


def fig_chains():
    cc = pd.read_csv(ROOT / "results" / "cc_orders.csv")
    fig, axes = plt.subplots(1, 2, figsize=(16.0, 6.0))
    for ax, base, title in zip(axes, ["RF", "LR"], ["Random Forest", "Regressão Logística"]):
        d = cc[(cc.base == base) & (cc.metric == "auc_macro")].sort_values("fold")
        x = d.fold.values + 1
        ax.vlines(x, d.cc_min - d.br, d.cc_max - d.br, color=BLUE, lw=9, alpha=0.35,
                  label="10 cadeias, ordens aleatórias (min–máx)")
        ax.plot(x, d.cc_mean - d.br, "o", color=BLUE, ms=9, label="média das cadeias")
        ax.plot(x, d.ens - d.br, "D", color=ORANGE, ms=12, label="combinação das 10 (regra da média)")
        ax.axhline(0, color="#8A8985", lw=1.4, ls=(0, (4, 3)))
        ax.set_xticks(x)
        ax.set_xlabel("dobra")
        ax.set_title(title, loc="left", color=INK)
        ax.grid(axis="y")
        style.comma_axes(ax, which="y")
    axes[0].set_ylabel("Δ AUC-ROC vs. BR")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.88), w_pad=3)
    save(fig, "chains")


if __name__ == "__main__":
    which = sys.argv[1:] or ["eda"]
    if "eda" in which:
        fig_prevalence()
        fig_cardinality()
        fig_phi()
    if "results" in which:
        fig_delta_methods()
        fig_rakel()
        fig_chains()
    print("ok", sorted(p.name for p in OUT.glob("*.png")))
