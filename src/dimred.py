"""Part B: PCA scree, reconstruction error, 2-D visualisation, loadings.
PCA is always fitted on the standardized features only (no labels)."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from datasets import MODS, FEATURE_NAMES


def run_partB(Xs, y, fig_dir, tab_dir, log):
    res = {}
    n, d = Xs.shape

    # ---- B6: scree / cumulative -------------------------------------------
    pca = PCA().fit(Xs)                                  # features only
    evr = pca.explained_variance_ratio_
    cum = np.cumsum(evr)
    n90 = int(np.argmax(cum >= 0.90) + 1)
    n95 = int(np.argmax(cum >= 0.95) + 1)
    res.update(B6_evr=evr.tolist(), B6_cum=cum.tolist(), B6_n90=n90, B6_n95=n95)
    log(f"[B6] EVR = {np.round(evr, 4)}\n     cumulative = {np.round(cum, 4)}\n"
        f"     components for 90% = {n90}, for 95% = {n95}")

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    comp = np.arange(1, d + 1)
    ax[0].bar(comp, evr)
    ax[0].set_xlabel("component")
    ax[0].set_ylabel("explained variance ratio")
    ax[0].set_title("Scree plot")
    ax[1].plot(comp, cum, "o-")
    ax[1].axhline(0.90, ls="--", c="gray")
    ax[1].axhline(0.95, ls=":", c="gray")
    ax[1].set_xlabel("# components")
    ax[1].set_ylabel("cumulative EVR")
    ax[1].set_title("Cumulative explained variance")
    fig.tight_layout()
    fig.savefig(fig_dir / "B6_scree.png", dpi=150)
    plt.close(fig)

    # ---- B7: reconstruction error ------------------------------------------
    lam = pca.explained_variance_              # eigenvalues, denominator N-1
    rows = []
    for q in range(1, d + 1):
        p = PCA(n_components=q).fit(Xs)
        Xh = p.inverse_transform(p.transform(Xs))
        mse = float(np.mean(np.sum((Xs - Xh) ** 2, axis=1)))   # Eq. (6)
        pred = float((n - 1) / n * lam[q:].sum())              # Eq. (7)
        rows.append(dict(q=q, MSE_measured=mse, MSE_predicted=pred,
                         abs_diff=abs(mse - pred)))
    df = pd.DataFrame(rows)
    df.to_csv(tab_dir / "B7_reconstruction_mse.csv", index=False)
    res["B7_table"] = rows
    log("[B7] reconstruction MSE:\n" + df.to_string(index=False, float_format=lambda v: f"{v:.6g}"))

    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    ax.plot(df.q, df.MSE_measured, "o-", label="measured")
    ax.plot(df.q, df.MSE_predicted, "x--", label=r"$\frac{N-1}{N}\sum_{i>q}\lambda_i$")
    ax.set_xlabel("q")
    ax.set_ylabel("reconstruction MSE")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "B7_reconstruction_mse.png", dpi=150)
    plt.close(fig)

    # ---- B8: 2-D projection --------------------------------------------------
    pca2 = PCA(n_components=2).fit(Xs)                    # labels NOT used
    Z2 = pca2.transform(Xs)
    fig, ax = plt.subplots(figsize=(6, 5))
    for k, m in enumerate(MODS):
        ax.scatter(Z2[y == k, 0], Z2[y == k, 1], s=8, alpha=0.6, label=m)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("PCA 2-D (labels used ONLY for colour)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "B8_pca_scatter.png", dpi=150)
    plt.close(fig)

    load = pca2.components_[0]
    order = np.argsort(-np.abs(load))
    ldf = pd.DataFrame({"feature": np.array(FEATURE_NAMES)[order], "PC1_loading": load[order]})
    ldf.to_csv(tab_dir / "B8_pc1_loadings.csv", index=False)
    res["B8_pc1_top"] = ldf.head(3).to_dict("records")
    res["B8_pc1_all"] = ldf.to_dict("records")
    res["B8_evr_2d"] = float(pca2.explained_variance_ratio_.sum())
    # class-centroid distances in PC space (helps identify overlapping pairs)
    cent = np.array([Z2[y == k].mean(0) for k in range(len(MODS))])
    spread = np.array([Z2[y == k].std(0).mean() for k in range(len(MODS))])
    cd = {}
    for i in range(len(MODS)):
        for j in range(i + 1, len(MODS)):
            cd[f"{MODS[i]}-{MODS[j]}"] = float(np.linalg.norm(cent[i] - cent[j]) /
                                              (0.5 * (spread[i] + spread[j])))
    res["B8_centroid_dist_in_std"] = cd
    log(f"[B8] top |loadings| on PC1:\n{ldf.head(5).to_string(index=False)}")
    log(f"     centroid separation / mean within-class std (2-D): {cd}")
    return res, Z2
