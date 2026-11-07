"""Parts C and D: k-means, model selection, PCA-before-clustering, SNR sweep,
GMM / agglomerative comparison, make_moons failure case.

Discipline: every .fit / .fit_predict receives features ONLY. The labels `y`
appear only inside evaluate() (ARI / NMI), contingency tables, and plot colours.
"""
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
from sklearn.datasets import make_moons
from sklearn.decomposition import PCA
from sklearn.metrics import (adjusted_rand_score, normalized_mutual_info_score,
                             silhouette_score)
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from datasets import MODS, get_features

SEED = 0


def km(X, k, seed=SEED):
    return KMeans(n_clusters=k, n_init=10, random_state=seed).fit(X)


def evaluate(X, labels, y):
    """Internal: silhouette.  External (labels used for EVALUATION only): ARI, NMI."""
    return dict(silhouette=float(silhouette_score(X, labels)),
                ARI=float(adjusted_rand_score(y, labels)),
                NMI=float(normalized_mutual_info_score(y, labels)))


# ----------------------------------------------------------------------- C9
def run_C9(Xs, y, tab_dir, log):
    res = {}
    m = km(Xs, 4)
    ev = evaluate(Xs, m.labels_, y)
    res["C9_inertia"] = float(m.inertia_)
    res.update({f"C9_{k}": v for k, v in ev.items()})
    log(f"[C9a,b] k=4: inertia={m.inertia_:.2f}, silhouette={ev['silhouette']:.4f}, "
        f"ARI={ev['ARI']:.4f}, NMI={ev['NMI']:.4f}  (ARI/NMI: labels used for evaluation only)")
    ct = pd.crosstab(pd.Series(np.array(MODS)[y], name="true class"),
                     pd.Series(m.labels_, name="cluster"))
    ct = ct.loc[MODS]
    ct.to_csv(tab_dir / "C9_contingency.csv")
    res["C9_contingency"] = ct.to_dict()
    log("[C9c] contingency table:\n" + ct.to_string())

    # ARI of purely random assignments (for part d)
    rng = np.random.default_rng(SEED)
    r = np.array([adjusted_rand_score(y, rng.integers(0, 4, len(y))) for _ in range(500)])
    res["C9_ARI_random_mean"], res["C9_ARI_random_std"] = float(r.mean()), float(r.std())
    log(f"[C9d] ARI of 500 random 4-way assignments: mean={r.mean():.5f}, std={r.std():.5f}, "
        f"min={r.min():.4f}, max={r.max():.4f}")
    return res, m


# ---------------------------------------------------------------------- C10
def run_C10(Xs, y, fig_dir, tab_dir, log):
    rows = []
    for k in range(2, 9):
        m = km(Xs, k)
        e = evaluate(Xs, m.labels_, y)
        rows.append(dict(k=k, inertia=float(m.inertia_), **e))
    df = pd.DataFrame(rows)
    df.to_csv(tab_dir / "C10_k_sweep.csv", index=False)
    log("[C10] k sweep (select k from inertia/silhouette ONLY; ARI shown for evaluation):\n"
        + df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    best_sil = int(df.loc[df.silhouette.idxmax(), "k"])
    best_ari = int(df.loc[df.ARI.idxmax(), "k"])
    log(f"     k with max silhouette = {best_sil};  k with max ARI = {best_ari}")

    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    ax[0].plot(df.k, df.inertia, "o-")
    ax[0].set_title("Elbow: inertia vs k")
    ax[1].plot(df.k, df.silhouette, "o-", c="C1")
    ax[1].set_title("Silhouette vs k")
    ax[2].plot(df.k, df.ARI, "o-", c="C2")
    ax[2].set_title("ARI vs k (evaluation only)")
    for a in ax:
        a.set_xlabel("k")
    fig.tight_layout()
    fig.savefig(fig_dir / "C10_k_selection.png", dpi=150)
    plt.close(fig)
    return dict(C10_table=rows, C10_best_sil_k=best_sil, C10_best_ari_k=best_ari)


# ---------------------------------------------------------------------- C11
def run_C11(Xs, y, Z2, tab_dir, log):
    # Z2 is the 2-D PCA representation of Problem 8 (fit on features only)
    rows = []
    for name, X in (("12-D standardized", Xs), ("2-D PCA", Z2)):
        m = km(X, 4)
        rows.append(dict(representation=name, **evaluate(X, m.labels_, y)))
    df = pd.DataFrame(rows)
    df.to_csv(tab_dir / "C11_pca_before_clustering.csv", index=False)
    log("[C11]\n" + df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    return dict(C11_table=rows)


# ---------------------------------------------------------------------- C12
def run_C12(fig_dir, tab_dir, log, snrs=(20, 15, 10, 5, 0)):
    rows = []
    for snr in snrs:
        Xs, y, _ = get_features(snr_db=snr, seed=SEED)
        m = km(Xs, 4)
        e = evaluate(Xs, m.labels_, y)
        # Supervised reference (stand-in for the Lab-3 curve, which is not available here):
        # RBF-SVM on a stratified 70/30 split of the SAME features. Labels are legitimately
        # used here because this is a supervised model.
        Xtr, Xte, ytr, yte = train_test_split(Xs, y, test_size=0.3, stratify=y, random_state=SEED)
        sc = StandardScaler().fit(Xtr)
        acc = float(SVC(kernel="rbf", random_state=SEED).fit(sc.transform(Xtr), ytr)
                    .score(sc.transform(Xte), yte))
        rows.append(dict(snr_db=snr, ARI=e["ARI"], NMI=e["NMI"],
                         silhouette=e["silhouette"], supervised_acc=acc))
    df = pd.DataFrame(rows)
    df.to_csv(tab_dir / "C12_snr_sweep.csv", index=False)
    log("[C12]\n" + df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ax[0].plot(df.snr_db, df.ARI, "o-")
    ax[0].axhline(0, ls=":", c="gray")
    ax[0].set_xlabel("SNR (dB)")
    ax[0].set_ylabel("ARI (k-means, k=4)")
    ax[0].set_title("Unsupervised: ARI vs SNR")
    ax[0].invert_xaxis()
    ax[1].plot(df.snr_db, df.supervised_acc, "s-", c="C3", label="supervised SVM accuracy")
    ax[1].plot(df.snr_db, df.ARI, "o-", label="k-means ARI")
    ax[1].set_xlabel("SNR (dB)")
    ax[1].set_title("Supervised vs unsupervised")
    ax[1].legend()
    ax[1].invert_xaxis()
    fig.tight_layout()
    fig.savefig(fig_dir / "C12_snr_sweep.png", dpi=150)
    plt.close(fig)
    return dict(C12_table=rows)


# ---------------------------------------------------------------------- C13
def _three_algos(Xs, seed):
    out = {}
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out["KMeans"] = KMeans(n_clusters=4, n_init=10, random_state=seed).fit_predict(Xs)
        gmm = GaussianMixture(n_components=4, covariance_type="full", n_init=5,
                              random_state=seed)
        out["GMM (full)"] = gmm.fit_predict(Xs)
        out["Agglomerative (ward)"] = AgglomerativeClustering(
            n_clusters=4, linkage="ward").fit_predict(Xs)
    return out, [str(x.message) for x in w]


def run_D13(Xs, y, tab_dir, log, n_repeat=10):
    res = {}
    labs, warns = _three_algos(Xs, SEED)
    rows = []
    for name, l in labs.items():
        rows.append(dict(method=name,
                         ARI=float(adjusted_rand_score(y, l)),
                         NMI=float(normalized_mutual_info_score(y, l))))
    df = pd.DataFrame(rows)
    df.to_csv(tab_dir / "D13_algorithm_comparison.csv", index=False)
    log("[D13] single run, 20 dB, dataset seed 0, algo seed 0 (GMM n_init=5, ward linkage):\n"
        + df.to_string(index=False, float_format=lambda v: f"{v:.4f}")
        + f"\n     warnings: {warns or 'none'}")
    res["D13_table"] = rows
    res["D13_warnings"] = warns

    # repeat over independent datasets to judge whether differences are meaningful
    rep = {n: {"ARI": [], "NMI": []} for n in labs}
    for s in range(n_repeat):
        Xs_s, y_s, _ = get_features(snr_db=20, seed=100 + s)
        L, _ = _three_algos(Xs_s, s)
        for n, l in L.items():
            rep[n]["ARI"].append(adjusted_rand_score(y_s, l))
            rep[n]["NMI"].append(normalized_mutual_info_score(y_s, l))
    rrows = [dict(method=n, ARI_mean=float(np.mean(v["ARI"])), ARI_std=float(np.std(v["ARI"])),
                  NMI_mean=float(np.mean(v["NMI"])), NMI_std=float(np.std(v["NMI"])))
             for n, v in rep.items()]
    rdf = pd.DataFrame(rrows)
    rdf.to_csv(tab_dir / "D13_repeated_runs.csv", index=False)
    log(f"[D13+] mean +/- std over {n_repeat} independent datasets (seeds 100..{99+n_repeat}):\n"
        + rdf.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    res["D13_repeat"] = rrows
    return res


# ---------------------------------------------------------------------- D14
def run_D14(fig_dir, log):
    X, y = make_moons(n_samples=500, noise=0.05, random_state=0)
    lk = KMeans(n_clusters=2, n_init=10, random_state=SEED).fit_predict(X)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        ls = SpectralClustering(n_clusters=2, affinity="nearest_neighbors",
                                random_state=SEED).fit_predict(X)
    res = dict(D14_kmeans_ARI=float(adjusted_rand_score(y, lk)),
               D14_spectral_ARI=float(adjusted_rand_score(y, ls)),
               D14_warnings=[str(x.message) for x in w])
    log(f"[D14] moons: k-means ARI={res['D14_kmeans_ARI']:.4f}, "
        f"spectral ARI={res['D14_spectral_ARI']:.4f} (labels used for ARI only); "
        f"warnings: {res['D14_warnings'] or 'none'}")
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for a, l, nm in zip(ax, (lk, ls), ("k-means (k=2)", "Spectral (kNN affinity)")):
        a.scatter(X[:, 0], X[:, 1], c=l, s=10, cmap="coolwarm")
        a.set_title(nm)
    fig.tight_layout()
    fig.savefig(fig_dir / "D14_moons.png", dpi=150)
    plt.close(fig)
    return res
