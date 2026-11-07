"""Part B: synthetic modulation-classification dataset + feature extraction.

Labels `y` returned here are ground truth. They are NEVER passed to any fit()
of an unsupervised model in this project (only used for colouring / evaluation).
"""
import numpy as np
from sklearn.preprocessing import StandardScaler

MODS = ["BPSK", "QPSK", "8PSK", "16QAM"]

FEATURE_NAMES = [
    "std|r|", "mean|r|^2", "std|r|^2", "kurt|r|", "skew|r|",
    "|C20|", "|C40|", "|C42|", "|C40|/C21^2", "|C42|/C21^2",
    "std(phase)", "mean|phase|",
]


def symbols(mod, n, rng):
    if mod == "BPSK":
        return (2 * rng.integers(0, 2, n) - 1).astype(complex)
    if mod == "QPSK":
        return np.exp(1j * (np.pi / 4 + rng.integers(0, 4, n) * np.pi / 2))
    if mod == "8PSK":
        return np.exp(1j * rng.integers(0, 8, n) * np.pi / 4)
    if mod == "16QAM":
        lv = np.array([-3, -1, 1, 3])
        return (rng.choice(lv, n) + 1j * rng.choice(lv, n)) / np.sqrt(10)
    raise ValueError(mod)


def make_dataset(n_per_class=250, n_sym=256, snr_db=20, seed=0):
    rng = np.random.default_rng(seed)
    X, y = [], []
    for k, m in enumerate(MODS):
        for _ in range(n_per_class):
            s = symbols(m, n_sym, rng)
            s = s * np.exp(1j * rng.uniform(0, 2 * np.pi))  # unknown carrier phase
            p = np.mean(np.abs(s) ** 2)
            nz = np.sqrt(p / (10 ** (snr_db / 10)) / 2) * (
                rng.normal(size=n_sym) + 1j * rng.normal(size=n_sym))
            X.append(s + nz)
            y.append(k)
    return np.array(X), np.array(y)


def feats(R):
    out = []
    for r in R:
        r = r / (np.sqrt(np.mean(np.abs(r) ** 2)) + 1e-12)  # unit power
        a = np.abs(r)
        ph = np.angle(r)
        C20 = np.mean(r ** 2)
        C21 = np.mean(np.abs(r) ** 2)
        C40 = np.mean(r ** 4) - 3 * C20 ** 2
        C42 = np.mean(np.abs(r) ** 4) - np.abs(C20) ** 2 - 2 * C21 ** 2
        am, asd = a.mean(), a.std() + 1e-12
        out.append([a.std(), np.mean(a ** 2), np.std(a ** 2),
                    np.mean(((a - am) / asd) ** 4), np.mean(((a - am) / asd) ** 3),
                    np.abs(C20), np.abs(C40), np.abs(C42),
                    np.abs(C40) / C21 ** 2, np.abs(C42) / C21 ** 2,
                    ph.std(), np.mean(np.abs(ph))])
    return np.array(out)


def get_features(snr_db=20, seed=0, n_per_class=250, n_sym=256):
    """Full pipeline: signals -> 12 features -> StandardScaler.
    Returns (X_standardized, y, raw_features)."""
    R, y = make_dataset(n_per_class=n_per_class, n_sym=n_sym, snr_db=snr_db, seed=seed)
    F = feats(R)
    Xs = StandardScaler().fit_transform(F)  # unsupervised: features only
    return Xs, y, F
