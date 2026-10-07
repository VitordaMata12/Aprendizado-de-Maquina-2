"""Agrega os resultados por dobra e gera tabelas (LaTeX/CSV) e figuras do artigo."""
import json
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform

import style
from evaluation import corrected_ttest, holm

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
FIG = ROOT / "figures"
style.setup()

METHODS = ["Prior", "BR", "LP", "CC", "ECC-nb", "EBR", "ECC", "RAkEL", "Stacked-BR", "ML-kNN"]
SHOW = {"Prior": "Prior (base trivial)", "BR": "BR", "LP": "LP", "CC": "CC (1 ordem)",
        "ECC-nb": "CC$\\times$10 (média)", "EBR": "EBR", "ECC": "ECC", "RAkEL": "RAkEL ($k{=}3$)",
        "Stacked-BR": "BR empilhado", "ML-kNN": "ML-kNN ($k{=}10$)"}
METRICS = [("hamming_loss", "HL", False), ("subset_acc", "Ac.~exata", True),
           ("jaccard", "Jaccard", True), ("f1_micro", "F1-mi", True), ("f1_macro", "F1-ma", True),
           ("auc_macro", "AUC-ma", True), ("ranking_loss", "RL", False)]
N, N_FOLDS = 1427, 10


def load(base):
    folds = []
    for p in sorted((RES / "main").glob(f"fold_*_{base}.pkl")):
        with open(p, "rb") as fh:
            folds.append(pickle.load(fh))
    return folds


def long_table(folds, base):
    rows = []
    for o in folds:
        for m, d in o["metrics"].items():
            rows.append({"base": base, "fold": o["fold"], "method": m, **d})
    return pd.DataFrame(rows)


def fmt(x, nd=3):
    return f"{x:.{nd}f}".replace(".", ",")


def main_table(df):
    """Tabela LaTeX: média (dp) por método; negrito = melhor da coluna dentro do bloco;
    † = diferença significativa contra o BR do mesmo classificador base
    (t corrigido de Nadeau-Bengio, Holm, alpha = 0,05)."""
    n_test = N / N_FOLDS
    n_train = N - n_test
    lines, sig_rows = [], []
    for base in ["RF", "LR"]:
        d = df[df.base == base]
        if d.empty:
            continue
        methods = [m for m in METHODS if m in set(d.method)]
        means = d.groupby("method")[[m for m, _, _ in METRICS]].mean()
        sds = d.groupby("method")[[m for m, _, _ in METRICS]].std()
        # testes contra BR
        pv = {}
        for met, _, _ in METRICS:
            br = d[d.method == "BR"].sort_values("fold")[met].values
            others = [m for m in methods if m not in ("BR",)]
            ps = []
            for m in others:
                v = d[d.method == m].sort_values("fold")[met].values
                if len(v) != len(br) or len(br) < 3:
                    ps.append(1.0)
                    continue
                t, p = corrected_ttest(v, br, n_train, n_test)
                ps.append(p)
            for m, pa, praw in zip(others, holm(ps), ps):
                pv[(m, met)] = pa
                sig_rows.append({"base": base, "method": m, "metric": met,
                                 "diff_vs_BR": means.loc[m, met] - means.loc["BR", met],
                                 "p_raw": praw, "p_holm": pa})
        lines.append(f"\\multicolumn{{8}}{{l}}{{\\textit{{Classificador base: "
                     f"{'Random Forest' if base == 'RF' else 'Regressão Logística'}}}}}\\\\")
        for m in methods:
            if base == "LR" and m == "ML-kNN":
                continue
            cells = [SHOW[m]]
            for met, _, hib in METRICS:
                col = means.loc[[x for x in methods if x != "Prior"
                                 and not (base == "LR" and x == "ML-kNN")], met]
                best = col.max() if hib else col.min()
                s = fmt(means.loc[m, met])
                if np.isclose(means.loc[m, met], best) and m != "Prior":
                    s = f"\\textbf{{{s}}}"
                if m != "BR" and pv.get((m, met), 1) < 0.05:
                    better = (means.loc[m, met] > means.loc["BR", met]) == hib
                    s += "$^{\\blacktriangle}$" if better else "$^{\\triangledown}$"
                cells.append(s)
            lines.append(" & ".join(cells) + " \\\\")
        lines.append("\\midrule")
    lines = lines[:-1]
    (RES / "tab_main.tex").write_text("\n".join(lines) + "\n")
    pd.DataFrame(sig_rows).to_csv(RES / "significance_vs_BR.csv", index=False)


def summary_csv(df):
    agg = df.groupby(["base", "method"]).agg(["mean", "std"])
    agg.to_csv(RES / "summary_main.csv")
    return agg


def cc_order_analysis(folds, base):
    """Variabilidade entre ordens da cadeia e ganho da combinação."""
    rows = []
    for o in folds:
        cc = pd.DataFrame([o["metrics"][f"CC_o{i}"] for i in range(10)])
        for met, _, hib in METRICS:
            rows.append({"base": base, "fold": o["fold"], "metric": met,
                         "cc_min": cc[met].min(), "cc_max": cc[met].max(), "cc_mean": cc[met].mean(),
                         "ens": o["metrics"]["ECC-nb"][met], "br": o["metrics"]["BR"][met]})
    return pd.DataFrame(rows)


def diversity_table(all_folds):
    rows = []
    for base, folds in all_folds.items():
        for ens in ["ECC-nb", "EBR", "ECC"]:
            dis = np.mean([o["div"][ens][0] for o in folds])
            q = np.mean([o["div"][ens][1] for o in folds])
            if ens == "ECC-nb":
                mem = [np.mean([o["metrics"][f"CC_o{i}"][k] for i in range(10)])
                       for o in folds for k in ["auc_macro"]]
                mem_f1 = [np.mean([o["metrics"][f"CC_o{i}"]["f1_micro"] for i in range(10)]) for o in folds]
                mem_hl = [np.mean([o["metrics"][f"CC_o{i}"]["hamming_loss"] for i in range(10)]) for o in folds]
            else:
                mem = [np.mean([mm["auc_macro"] for mm in o["member_metrics"][ens]]) for o in folds]
                mem_f1 = [np.mean([mm["f1_micro"] for mm in o["member_metrics"][ens]]) for o in folds]
                mem_hl = [np.mean([mm["hamming_loss"] for mm in o["member_metrics"][ens]]) for o in folds]
            ens_auc = [o["metrics"][ens]["auc_macro"] for o in folds]
            ens_f1 = [o["metrics"][ens]["f1_micro"] for o in folds]
            ens_hl = [o["metrics"][ens]["hamming_loss"] for o in folds]
            rows.append({"base": base, "ensemble": ens, "desacordo": dis, "Q": q,
                         "auc_membro": np.mean(mem), "auc_ens": np.mean(ens_auc),
                         "f1_membro": np.mean(mem_f1), "f1_ens": np.mean(ens_f1),
                         "hl_membro": np.mean(mem_hl), "hl_ens": np.mean(ens_hl)})
    d = pd.DataFrame(rows)
    d.to_csv(RES / "diversity.csv", index=False)
    name = {"ECC-nb": "CC$\\times$10 (só ordem)", "EBR": "EBR (só bootstrap)", "ECC": "ECC (ordem + bootstrap)"}
    lines = []
    for _, r in d.iterrows():
        lines.append(" & ".join([r.base, name[r.ensemble], fmt(r.desacordo), fmt(r.Q, 2),
                                 fmt(r.auc_membro), fmt(r.auc_ens), f"{'+' if r.auc_ens >= r.auc_membro else ''}{fmt(r.auc_ens - r.auc_membro)}",
                                 fmt(r.f1_membro), fmt(r.f1_ens)]) + " \\\\")
    (RES / "tab_div.tex").write_text("\n".join(lines) + "\n")
    return d


def fig_dependencies(folds_rf, label_names):
    """(a) correlação phi entre rótulos; (b) pesos do meta-nível do BR empilhado."""
    D = np.load(ROOT / "data" / "features.npz")
    Y = D["Y"].astype(float)
    L = Y.shape[1]
    phi = np.corrcoef(Y.T)
    dist = 1 - phi
    np.fill_diagonal(dist, 0)
    leaf = leaves_list(linkage(squareform(dist, checks=False), "average"))
    W = np.mean([o["meta_coef"] for o in folds_rf], axis=0)  # (alvo j, entrada i)
    names = [style.LABELS_PT[n] for n in label_names]
    seq = LinearSegmentedColormap.from_list(
        "seq", ["#f7f6f3", "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"])
    div = LinearSegmentedColormap.from_list(
        "div", ["#9c1f1f", "#e34948", "#f4a3a2", style.MID, "#9ec5f4", "#3987e5", "#104281"])
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.55))
    P = phi[np.ix_(leaf, leaf)].copy()
    np.fill_diagonal(P, np.nan)
    Wl = W[np.ix_(leaf, leaf)].copy()
    off = Wl.copy()
    np.fill_diagonal(off, np.nan)
    vmax = np.nanpercentile(np.abs(off), 98)
    ims = [axes[0].imshow(P, cmap=seq, vmin=0, vmax=0.55),
           axes[1].imshow(off, cmap=div, vmin=-vmax, vmax=vmax)]
    titles = ["(a) Correlação $\\phi$ entre rótulos", "(b) Pesos fora da diagonal do meta-nível"]
    for k, ax in enumerate(axes):
        ax.set_xticks(range(L))
        ax.set_yticks(range(L))
        ax.set_xticklabels([names[i] for i in leaf], rotation=90, fontsize=4.6)
        ax.set_yticklabels([names[i] for i in leaf] if k == 0 else [], fontsize=4.6)
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(titles[k], loc="left", fontsize=7.5)
        cb = fig.colorbar(ims[k], ax=ax, fraction=0.045, pad=0.02)
        cb.ax.tick_params(labelsize=5.5)
        cb.outline.set_visible(False)
        style.comma_axes(cb.ax, which="y")
    axes[1].set_ylabel("rótulo predito (linha) ← escore de entrada (coluna)", fontsize=5.5)
    fig.tight_layout(w_pad=0.6)
    fig.savefig(FIG / "fig_dep.pdf")
    fig.savefig(FIG / "fig_dep.png", dpi=220)
    plt.close(fig)
    # correlação entre phi e pesos fora da diagonal
    iu = np.triu_indices(L, 1)
    Wsym = (W + W.T) / 2
    r = np.corrcoef(phi[iu], Wsym[iu])[0, 1]
    diag = np.diag(W)
    return {"corr_phi_meta": float(r), "meta_diag_mean": float(diag.mean()),
            "meta_off_mean": float(W[~np.eye(L, dtype=bool)].mean()),
            "meta_off_frac_pos": float((W[~np.eye(L, dtype=bool)] > 0).mean())}


def fig_rakel():
    files = sorted((RES / "rakel_k").glob("fold_*_RF.pkl"))
    if not files:
        return None
    rows = []
    for p in files:
        with open(p, "rb") as fh:
            r = pickle.load(fh)
        f = int(p.stem.split("_")[1])
        for k, d in r.items():
            rows.append({"fold": f, "k": k, **d})
    df = pd.DataFrame(rows)
    g = df.groupby("k").agg(["mean", "std"])
    g.to_csv(RES / "rakel_k_summary.csv")
    df.to_csv(RES / "rakel_k_by_fold.csv", index=False)
    ks = np.array(sorted(df.k.unique()))
    nf = df.fold.nunique()
    panels = [("auc_macro", "AUC-ROC macro (maior é melhor)"), ("hamming_loss", "Hamming loss (menor é melhor)"),
              ("ranking_loss", "Ranking loss (menor é melhor)")]
    fig, axes = plt.subplots(1, 3, figsize=(6.4, 2.15))
    for ax, (met, lab) in zip(axes, panels):
        piv = df.pivot(index="fold", columns="k", values=met)
        delta = piv.sub(piv[1], axis=0)  # diferença pareada em relação a k = 1 (BR)
        m = delta.mean().values
        se = delta.std(ddof=1).values / np.sqrt(nf)
        ax.axhline(0, color=style.GRAY, lw=0.7, ls=(0, (3, 2)))
        ax.fill_between(ks, m - se, m + se, color=style.BLUE, alpha=0.18, lw=0)
        ax.plot(ks, m, color=style.BLUE, lw=1.6, marker="o", ms=3.5)
        ax.set_xscale("log")
        ax.set_xticks(ks)
        ax.set_xticklabels([str(k) for k in ks], fontsize=6)
        ax.minorticks_off()
        ax.set_xlabel("tamanho $k$ do conjunto de rótulos")
        ax.set_title(lab, loc="left", fontsize=7)
        ax.grid(axis="y")
        ax.annotate("BR", (ks[0], m[0]), textcoords="offset points", xytext=(2, 4), fontsize=6.5, color=style.INK2)
        ax.annotate("LP", (ks[-1], m[-1]), textcoords="offset points", xytext=(-10, 4), fontsize=6.5, color=style.INK2)
        ib = int(np.argmax(m) if met == "auc_macro" else np.argmin(m))
        ax.plot(ks[ib], m[ib], marker="o", ms=6, mfc="none", mec=style.ORANGE, mew=1.2)
        style.comma_axes(ax, which="y")
    axes[0].set_ylabel("diferença para $k=1$ (BR)")
    fig.tight_layout(w_pad=0.8)
    fig.savefig(FIG / "fig_rakel_k.pdf")
    fig.savefig(FIG / "fig_rakel_k.png", dpi=220)
    plt.close(fig)
    return g


def fig_chains(cc_df):
    """Ganho de cada cadeia individual e do ensemble em relação ao BR, por dobra."""
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.3))
    for ax, base in zip(axes, ["RF", "LR"]):
        d = cc_df[(cc_df.base == base) & (cc_df.metric == "auc_macro")].sort_values("fold")
        if d.empty:
            continue
        x = d.fold.values
        ax.vlines(x, d.cc_min - d.br, d.cc_max - d.br, color=style.BLUE, lw=2.2, alpha=0.45,
                  label="amplitude das 10 ordens de CC")
        ax.plot(x, d.cc_mean - d.br, "o", color=style.BLUE, ms=3.5, label="média das cadeias")
        ax.plot(x, d.ens - d.br, "D", color=style.ORANGE, ms=4, label="combinação (média das 10)")
        ax.axhline(0, color=style.GRAY, lw=0.7, ls=(0, (3, 2)))
        ax.set_xticks(x)
        ax.set_xticklabels([str(i + 1) for i in x])
        ax.set_xlabel("dobra")
        ax.set_title(f"({'a' if base == 'RF' else 'b'}) {'Random Forest' if base == 'RF' else 'Regressão Logística'}",
                     loc="left", fontsize=7.5)
        ax.grid(axis="y")
        style.comma_axes(ax, which="y")
    axes[0].set_ylabel("$\\Delta$ AUC-ROC macro vs. BR")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.02), fontsize=6.5)
    fig.tight_layout(rect=(0, 0.08, 1, 1), w_pad=1.0)
    fig.savefig(FIG / "fig_chains.pdf")
    fig.savefig(FIG / "fig_chains.png", dpi=220)
    plt.close(fig)


def per_label(folds, label_names, methods=("BR", "ECC-nb", "Stacked-BR", "Prior")):
    """AUC e F1 por rótulo, calculados sobre as predições fora-da-dobra concatenadas."""
    from sklearn.metrics import f1_score, roc_auc_score
    D = np.load(ROOT / "data" / "features.npz")
    Y = D["Y"]
    out = {}
    for m in methods:
        P = np.zeros(Y.shape)
        for o in folds:
            P[o["test_idx"]] = o["proba"][m]
        out[m] = {"auc": [roc_auc_score(Y[:, j], P[:, j]) if m != "Prior" else 0.5 for j in range(Y.shape[1])],
                  "f1": [f1_score(Y[:, j], P[:, j] >= 0.5, zero_division=0) for j in range(Y.shape[1])]}
    rows = []
    for j, n in enumerate(label_names):
        r = {"rotulo": style.LABELS_PT[n], "prev": Y[:, j].mean()}
        for m in methods:
            r[f"auc_{m}"] = out[m]["auc"][j]
            r[f"f1_{m}"] = out[m]["f1"][j]
        rows.append(r)
    d = pd.DataFrame(rows).sort_values("prev")
    d.to_csv(RES / "per_label.csv", index=False)
    return d


def novelty(all_folds):
    """Fração das predições (limiar 0,5) cujo conjunto de rótulos não existe no treino."""
    D = np.load(ROOT / "data" / "features.npz")
    Y = D["Y"]
    rows = []
    for base, folds in all_folds.items():
        for o in folds:
            te = np.asarray(o["test_idx"])
            tr = np.setdiff1d(np.arange(len(Y)), te)
            seen = {tuple(r) for r in Y[tr]}
            for m, P in o["proba"].items():
                pred = (np.asarray(P) >= 0.5).astype(np.int8)
                rows.append({"base": base, "fold": o["fold"], "method": m,
                             "novel_pred": float(np.mean([tuple(r) not in seen for r in pred])),
                             "card_pred": float(pred.sum(1).mean())})
    d = pd.DataFrame(rows).groupby(["base", "method"])[["novel_pred", "card_pred"]].mean()
    d.to_csv(RES / "novelty.csv")
    return d


def lp_decoding():
    p = RES / "lp_decoding.pkl"
    if not p.exists():
        return None
    with open(p, "rb") as fh:
        out = pickle.load(fh)
    rows = []
    for o in out:
        for dec in ["mode", "marg"]:
            rows.append({"base": o["base"], "decod": dec, "fold": o["fold"], **o[dec],
                         "novel": o[f"novel_{dec}"], "n_classes": o["n_classes"]})
    d = pd.DataFrame(rows)
    g = d.groupby(["base", "decod"]).mean(numeric_only=True)
    g.to_csv(RES / "lp_decoding_summary.csv")
    return g


def write_numbers(df, div, g, nov, lpd, extra):
    """Gera results/numbers.tex com macros \\val{grupo}{chave}{métrica} (vírgula decimal)."""
    lines = ["% gerado automaticamente por src/analyze.py -- não editar",
             "\\makeatletter",
             "\\newcommand{\\val}[3]{\\@ifundefined{v@#1@#2@#3}{\\textbf{??}}{\\csname v@#1@#2@#3\\endcsname}}"]

    def put(a, b, c, x, nd=3, pct=False):
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return
        v = f"{100 * x:.{nd}f}" if pct else f"{x:.{nd}f}"
        v = v.replace(".", "{,}").replace("-", "\\ensuremath{-}")
        lines.append(f"\\expandafter\\def\\csname v@{a}@{b}@{c}\\endcsname{{{v}}}")

    means = df.groupby(["base", "method"]).mean(numeric_only=True)
    for (base, m), r in means.iterrows():
        for met in [x for x, _, _ in METRICS]:
            put(base, m, met, r[met], 3)
            put(base, m, met + "4", r[met], 4)
    sig = pd.read_csv(RES / "significance_vs_BR.csv")
    for _, r in sig.iterrows():
        put("diff", f"{r.base}-{r.method}", r.metric, r.diff_vs_BR, 3)
        put("p", f"{r.base}-{r.method}", r.metric, r.p_holm, 3)
    for _, r in div.iterrows():
        for c in ["desacordo", "Q", "auc_membro", "auc_ens", "f1_membro", "f1_ens", "hl_membro", "hl_ens"]:
            put("div", f"{r.base}-{r.ensemble}", c, r[c], 3)
        put("div", f"{r.base}-{r.ensemble}", "Q2", r["Q"], 2)
        put("div", f"{r.base}-{r.ensemble}", "ganho", r["auc_ens"] - r["auc_membro"], 3)
    if g is not None:
        for k, r in g.iterrows():
            for met in [x for x, _, _ in METRICS]:
                put("rakel", f"k{k}", met, r[(met, "mean")], 3)
                put("rakel", f"k{k}", met + "4", r[(met, "mean")], 4)
    p = RES / "rakel_k_by_fold.csv"
    if p.exists():
        rk = pd.read_csv(p)
        for met in [x for x, _, _ in METRICS]:
            piv = rk.pivot(index="fold", columns="k", values=met)
            delta = piv.sub(piv[1], axis=0).mean()
            for k, v in delta.items():
                put("rkd", f"k{k}", met, v, 4)
                put("rkd", f"k{k}", met + "3", v, 3)
    # perda média de AUC dos membros bootstrap em relação ao BR
    for base in df.base.unique():
        br = means.loc[(base, "BR"), "auc_macro"]
        for _, r in div[div.base == base].iterrows():
            put("divbr", f"{base}-{r.ensemble}", "membro_menos_br", r["auc_membro"] - br, 3)
            put("divbr", f"{base}-{r.ensemble}", "perda_membro", br - r["auc_membro"], 3)
            put("divbr", f"{base}-{r.ensemble}", "ens_menos_br", r["auc_ens"] - br, 3)
    for (base, m), r in nov.iterrows():
        put("nov", f"{base}-{m}", "novel", r["novel_pred"], 0, pct=True)
        put("nov", f"{base}-{m}", "card", r["card_pred"], 1)
    if lpd is not None:
        for (base, dec), r in lpd.iterrows():
            for met in ["hamming_loss", "subset_acc", "jaccard", "f1_micro", "f1_macro"]:
                put("lpdec", f"{base}-{dec}", met, r[met], 3)
            put("lpdec", f"{base}-{dec}", "novel", r["novel"], 0, pct=True)
            put("lpdec", f"{base}-{dec}", "ncls", r["n_classes"], 0)
    for k, v in extra.items():
        if isinstance(v, bool):
            continue
        if isinstance(v, int):
            put("x", k, "v", float(v), 0)
        elif isinstance(v, float):
            put("x", k, "v", v, 3)
    lines.append("\\makeatother")
    (RES / "numbers.tex").write_text("\n".join(lines) + "\n")


def main():
    D = np.load(ROOT / "data" / "features.npz")
    label_names = list(D["label_names"])
    all_folds = {b: load(b) for b in ["RF", "LR"]}
    all_folds = {b: f for b, f in all_folds.items() if f}
    df = pd.concat([long_table(f, b) for b, f in all_folds.items()])
    df.to_csv(RES / "metrics_by_fold.csv", index=False)
    summary_csv(df)
    main_table(df)
    cc = pd.concat([cc_order_analysis(f, b) for b, f in all_folds.items()])
    cc.to_csv(RES / "cc_orders.csv", index=False)
    fig_chains(cc)
    div = diversity_table(all_folds)
    extra = {}
    if "RF" in all_folds:
        extra.update(fig_dependencies(all_folds["RF"], label_names))
        pl = per_label(all_folds["RF"], label_names)
        extra["per_label_ecc_gt_br_auc"] = int((pl["auc_ECC-nb"] > pl["auc_BR"]).sum())
    g = fig_rakel()
    nov = novelty(all_folds)
    lpd = lp_decoding()
    # resumo das ordens de CC
    for b in all_folds:
        d = cc[(cc.base == b) & (cc.metric == "auc_macro")]
        extra[f"{b}_cc_order_range_auc_mean"] = float((d.cc_max - d.cc_min).mean())
        extra[f"{b}_ens_minus_best_order_auc"] = float((d.ens - d.cc_max).mean())
        extra[f"{b}_folds_ens_beats_all_orders"] = int((d.ens > d.cc_max).sum())
        dF = cc[(cc.base == b) & (cc.metric == "f1_micro")]
        extra[f"{b}_cc_order_range_f1mi_mean"] = float((dF.cc_max - dF.cc_min).mean())
        tm = {k: float(np.mean([o["time"].get(k, np.nan) for o in all_folds[b]]))
              for k in ["BR", "LP", "CC_o0", "EBR", "ECC", "RAkEL", "Stacked-BR", "ML-kNN"]}
        extra[f"{b}_time"] = tm
        extra[f"{b}_n_folds"] = len(all_folds[b])
    write_numbers(df, div, g, nov, lpd, extra)
    with open(RES / "analysis_extra.json", "w") as fh:
        json.dump(extra, fh, indent=2, ensure_ascii=False)
    pd.set_option("display.width", 200)
    print(df.groupby(["base", "method"])[[m for m, _, _ in METRICS]].mean().round(4).to_string())
    print(div.round(4).to_string())
    print(json.dumps(extra, indent=1, ensure_ascii=False))
    if g is not None:
        print(g.round(4).to_string())
    print(nov.round(3).to_string())
    if lpd is not None:
        print(lpd.round(4).to_string())


if __name__ == "__main__":
    main()
