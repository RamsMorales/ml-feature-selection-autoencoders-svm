# Autoencoder vs. Concrete Autoencoder for Feature Extraction/Selection

**Goal:** Compare SVM on (1) original features, (2) PyTorch AE 50-dim latent, (3) Concrete AE 50 selected features.

## Project Structure
- `notebooks/`: exploration, SVM baseline, AE (PyTorch), Concrete AE (TF 2.12)
- `src/`: small helpers (optional)
- `data/`: not tracked; see `data/README.md`
- `reports/figures/`: saved plots

## Reproducibility
- Python 3.10
- Kernels: **AE (PyTorch)** and **CAE (TF 2.12)**
- Requirements in `envs/requirements_ae.txt` and `envs/requirements_cae.txt`

## How to Run
1. Create envs (see `envs/requirements_*.txt`) and select the matching Jupyter kernel.
2. Run notebooks in order: `00_exploration` → `10_svm_baseline` → `20_ae_pytorch` → `30_concrete_ae_tf`.
3. Outputs (confusion matrices, ROC/PR) will save to `reports/figures/`.

## Notes
- Dataset is excluded from the repo. See `data/README.md` for download/source.

