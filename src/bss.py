"""Part A: blind source separation. ICA vs PCA, and Gaussian non-identifiability."""
import warnings
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, stats
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import FastICA, PCA

A_MIX = np.array([[1.0, 1.0, 0.6],
                  [0.6, 1.8, 1.0],
                  [1.4, 0.7, 1.3]])
N = 2000
SEED_NOISE = 0      # np.random.seed for the illustrative sources
SEED_GAUSS = 1      # default_rng seed for Gaussian sources
SEED_ROT = 2        # default_rng seed for random rotations
SEED_ICA = 0        # FastICA random_state
N_ROT = 200


# ----------------------------------------------------------------- helpers
def matched_abs_corr(S, Z):
    """|Pearson| matrix between columns of S (true) and Z (estimated), plus a
    one-to-one matching that maximises total correlation (Hungarian algorithm).
    Handles ICA's permutation / sign / scale ambiguities."""
    k = S.shape[1]
    C = np.abs(np.corrcoef(S.T, Z.T)[:k, k:])
    r, c = linear_sum_assignment(-C)
    return C, r, c, C[r, c]


def whiten(X):
    """Centre + whiten with ddof=1 covariance:  Xw = Xc E Lambda^{-1/2}."""
    Xc = X - X.mean(axis=0)
    Sigma = np.cov(Xc, rowvar=False)           # ddof = 1 everywhere in this lab
    lam, E = np.linalg.eigh(Sigma)
    Xw = Xc @ E @ np.diag(lam ** -0.5)
    return Xw, Sigma, lam, E


def D_matrix(Z):
    """D_ij = E[zi^2 zj^2] - E[zi^2]E[zj^2] on centred, unit-variance outputs
    (unit variance so that PCA and ICA outputs are on the same scale)."""
    Z = (Z - Z.mean(0)) / Z.std(0)
    return np.cov(Z ** 2, rowvar=False, bias=True)


def offdiag_mean_abs(D):
    return float(np.abs(D[~np.eye(D.shape[0], dtype=bool)]).mean())


def make_sources():
    np.random.seed(SEED_NOISE)
    t = np.linspace(0, 8, N)
    s1 = np.sin(2 * np.pi * 1.0 * t)
    s2 = signal.square(2 * np.pi * 0.7 * t)
    s3 = signal.sawtooth(2 * np.pi * 1.3 * t)
    S = np.c_[s1, s2, s3]
    S += 0.05 * np.random.randn(*S.shape)
    S /= S.std(axis=0)
    return t, S


def _fmt(M):
    return np.array2string(np.asarray(M), precision=4, suppress_small=True)


# -------------------------------------------------------------------- main
def run_partA(fig_dir, tab_dir, log):
    res = {}
    names = ["sine", "square", "sawtooth"]
    t, S = make_sources()
    X = S @ A_MIX.T

    # ---- A1 ---------------------------------------------------------------
    fig, ax = plt.subplots(3, 2, figsize=(11, 6), sharex=True)
    for i in range(3):
        ax[i, 0].plot(t, S[:, i], lw=0.8)
        ax[i, 0].set_ylabel(f"s{i+1} ({names[i]})")
        ax[i, 1].plot(t, X[:, i], lw=0.8, color="C1")
        ax[i, 1].set_ylabel(f"x{i+1}")
    ax[0, 0].set_title("True sources")
    ax[0, 1].set_title("Observed mixtures")
    ax[2, 0].set_xlabel("t (s)")
    ax[2, 1].set_xlabel("t (s)")
    fig.tight_layout()
    fig.savefig(fig_dir / "A1_sources_mixtures.png", dpi=150)
    plt.close(fig)
    res["A1_cond_A"] = float(np.linalg.cond(A_MIX))
    res["A1_det_A"] = float(np.linalg.det(A_MIX))
    log(f"[A1] cond(A) = {res['A1_cond_A']:.3f}, det(A) = {res['A1_det_A']:.3f}")

    # ---- A2 ---------------------------------------------------------------
    Xw, Sigma, lam, E = whiten(X)
    covw = np.cov(Xw, rowvar=False)
    res["A2_cov_Xw"] = covw.tolist()
    res["A2_max_dev_from_I"] = float(np.abs(covw - np.eye(3)).max())
    res["A2_eigvals"] = lam.tolist()
    log(f"[A2] cov(X_w) =\n{_fmt(covw)}\n     max |cov - I| = {res['A2_max_dev_from_I']:.2e}")

    # ---- A3 ---------------------------------------------------------------
    ica = FastICA(n_components=3, whiten="unit-variance",
                  random_state=SEED_ICA, max_iter=1000)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        S_ica = ica.fit_transform(X)                 # X only: no S, no A supplied
    res["A3_warnings"] = [str(x.message) for x in w]
    C, r, c, m = matched_abs_corr(S, S_ica)
    res["A3_corr_matrix"] = C.tolist()
    res["A3_match"] = {names[i]: int(c[i]) for i in range(3)}
    res["A3_matched_corr"] = m.tolist()
    res["A3_mean_matched"] = float(m.mean())
    res["A3_n_iter"] = int(ica.n_iter_)
    log(f"[A3] |corr| matrix (rows=true, cols=ICA):\n{_fmt(C)}")
    log(f"     matching true->ICA comp: {res['A3_match']}")
    log(f"     matched |corr| = {_fmt(m)}  mean = {m.mean():.4f}  (n_iter={ica.n_iter_})")

    # aligned plot: permute, fix sign, standardise (for display only)
    fig, ax = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
    for i in range(3):
        z = S_ica[:, c[i]]
        z = z * np.sign(np.corrcoef(S[:, i], z)[0, 1])
        z = z / z.std()
        ax[i].plot(t, S[:, i], lw=1.0, label=f"true s{i+1} ({names[i]})")
        ax[i].plot(t, z, lw=0.8, ls="--", label=f"ICA comp {c[i]} (aligned)")
        ax[i].legend(loc="upper right", fontsize=8)
    ax[2].set_xlabel("t (s)")
    fig.suptitle("A3: FastICA components (permutation/sign/scale aligned for display)")
    fig.tight_layout()
    fig.savefig(fig_dir / "A3_ica_vs_true.png", dpi=150)
    plt.close(fig)

    # ---- A4 ---------------------------------------------------------------
    Z_pca = PCA(n_components=3).fit_transform(X)
    Cp, _, cp, mp = matched_abs_corr(S, Z_pca)
    res["A4_pca_corr_matrix"] = Cp.tolist()
    res["A4_pca_matched_corr"] = mp.tolist()
    res["A4_pca_mean_matched"] = float(mp.mean())
    log(f"[A4a] PCA |corr| matrix:\n{_fmt(Cp)}\n      matched = {_fmt(mp)} mean = {mp.mean():.4f}"
        f"   (ICA mean = {m.mean():.4f})")

    cov_pca = np.cov(Z_pca, rowvar=False)
    cov_ica = np.cov(S_ica, rowvar=False)
    res["A4_cov_pca"] = cov_pca.tolist()
    res["A4_cov_ica"] = cov_ica.tolist()
    log(f"[A4b] cov(PCA outputs) =\n{_fmt(cov_pca)}\n      cov(ICA outputs) =\n{_fmt(cov_ica)}")

    D_pca, D_ica = D_matrix(Z_pca), D_matrix(S_ica)
    res["A4_D_pca"] = D_pca.tolist()
    res["A4_D_ica"] = D_ica.tolist()
    res["A4_D_pca_offdiag_mean_abs"] = offdiag_mean_abs(D_pca)
    res["A4_D_ica_offdiag_mean_abs"] = offdiag_mean_abs(D_ica)
    log(f"[A4c] D (PCA) =\n{_fmt(D_pca)}\n      D (ICA) =\n{_fmt(D_ica)}")
    log(f"      mean |off-diag D|: PCA = {res['A4_D_pca_offdiag_mean_abs']:.4f}, "
        f"ICA = {res['A4_D_ica_offdiag_mean_abs']:.4f}")

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.2), sharey=True)
    for a_, Z, nm in zip(ax, (Z_pca, S_ica), ("PCA outputs", "ICA outputs")):
        a_.plot(t, Z[:, :], lw=0.5)
        a_.set_title(nm)
        a_.set_xlabel("t (s)")
    fig.tight_layout()
    fig.savefig(fig_dir / "A4_pca_vs_ica_outputs.png", dpi=150)
    plt.close(fig)

    # ---- A5 ---------------------------------------------------------------
    rng = np.random.default_rng(SEED_GAUSS)
    S_g = rng.standard_normal((N, 3))
    S_g /= S_g.std(axis=0)
    X_g = S_g @ A_MIX.T
    Xgw, *_ = whiten(X_g)

    rng_r = np.random.default_rng(SEED_ROT)
    scores = []
    for _ in range(N_ROT):
        Q, R_ = np.linalg.qr(rng_r.standard_normal((3, 3)))
        Q = Q * np.sign(np.diag(R_))                 # Haar-distributed rotation
        scores.append(matched_abs_corr(S_g, Xgw @ Q)[3].mean())
    scores = np.array(scores)
    res["A5_rot_mean"] = float(scores.mean())
    res["A5_rot_p5"] = float(np.percentile(scores, 5))
    res["A5_rot_p95"] = float(np.percentile(scores, 95))
    log(f"[A5a] random-rotation baseline ({N_ROT} rotations): mean={scores.mean():.4f}, "
        f"5th={res['A5_rot_p5']:.4f}, 95th={res['A5_rot_p95']:.4f}")

    ica_g = FastICA(n_components=3, whiten="unit-variance",
                    random_state=SEED_ICA, max_iter=1000)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        Zg = ica_g.fit_transform(X_g)
    res["A5_warnings"] = [str(x.message) for x in w]
    Cg, _, _, mg = matched_abs_corr(S_g, Zg)
    res["A5_ica_gauss_corr_matrix"] = Cg.tolist()
    res["A5_ica_gauss_mean"] = float(mg.mean())
    res["A5_ica_nongauss_mean"] = res["A3_mean_matched"]
    log(f"[A5b] FastICA on Gaussian mixtures: matched = {_fmt(mg)}, mean = {mg.mean():.4f}; "
        f"warnings: {res['A5_warnings'] or 'none'}")
    log(f"      FastICA on illustrative signals (A3): mean = {res['A3_mean_matched']:.4f}")

    # kurtosis (Fisher, excess)
    k_non = stats.kurtosis(S, axis=0, fisher=True)
    k_g = stats.kurtosis(S_g, axis=0, fisher=True)
    res["A5_kurt_nongauss"] = k_non.tolist()
    res["A5_kurt_gauss"] = k_g.tolist()
    log(f"[A5d] excess kurtosis: illustrative = {_fmt(k_non)}, Gaussian = {_fmt(k_g)}")

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.hist(scores, bins=20, alpha=0.7, label="random rotations (Gaussian)")
    ax.axvline(mg.mean(), color="r", label=f"FastICA Gaussian = {mg.mean():.3f}")
    ax.axvline(res["A3_mean_matched"], color="g", label=f"FastICA illustrative = {res['A3_mean_matched']:.3f}")
    ax.set_xlabel("mean matched |corr|")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "A5_gaussian_baseline.png", dpi=150)
    plt.close(fig)
    return res
