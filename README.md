# ECM-402 Lab 7 - Unsupervised Learning (ICA, PCA, Clustering)

Roll number: 523194

## Run
```bash
pip install -r requirements.txt
python src/main.py          # reproduces ALL numbers, figures (results/figures), tables (results/tables),
                            # results/results.json and results/log.txt
python src/make_report.py   # rebuilds report.pdf from results/
```
CPU only; whole pipeline takes about a minute. No datasets are committed - everything is generated from fixed seeds.

## Layout
| file | content |
|---|---|
| `src/bss.py` | Part A: sources, mixing, whitening, FastICA vs PCA, higher-order dependence D_ij, Gaussian non-identifiability |
| `src/datasets.py` | Part B: modulation signals (BPSK/QPSK/8PSK/16QAM), 12 AMC features, standardisation |
| `src/dimred.py` | Part B: scree, reconstruction-MSE identity, 2-D PCA, loadings |
| `src/clustering.py` | Parts C, D: k-means, k selection, PCA-before-clustering, SNR sweep, GMM/agglomerative, moons + spectral |
| `src/main.py` | runs everything |
| `src/make_report.py` | builds `report.pdf` |

## Seeds
`np.random.seed(0)` (A sources) | `default_rng(1)` Gaussian sources | `default_rng(2)` random rotations |
FastICA `random_state=0` | dataset `seed=0` | KMeans `random_state=0, n_init=10` |
GMM/Spectral `random_state=0` (GMM `n_init=5`) | Agglomerative `linkage="ward"` | D13 repeat study: dataset seeds 100-109.

## Label discipline
Ground-truth labels are used **only** for colouring plots and for ARI/NMI/contingency evaluation.
They are never passed to any `fit()` and never used to choose k or the number of components
(k is chosen from inertia/silhouette; ARI is reported afterwards).

## Notes
* Dataset used: the synthetic modulation dataset (not `load_digits`).
* The Lab-3 results were not available, so the supervised curve in C12 is an RBF-SVM on the same features
  (70/30 stratified split) used as a stand-in.
