# 🎯 Internship Placement Readiness Predictor

A Streamlit app that scores how competitive a student's profile is for internship placements, shown as
**"84% — Placement Ready"** or **"Needs Upskilling"**, and suggests the single change (one more project,
better soft skills, an internship, ...) that would help most.

> **Educational project.** Predictions come from a small ML model and are not a guarantee of any hiring outcome.

## Project structure

```
internship-readiness-predictor/
├── app.py                  # Streamlit app
├── model.joblib            # saved bundle: pipeline + feature order + threshold + UI ranges + metadata
├── requirements.txt        # runtime dependencies (used by Streamlit Community Cloud)
├── README.md
├── .streamlit/config.toml  # theme
└── training/
    ├── Internship_Readiness_Training.ipynb   # data -> EDA -> 3 models -> evaluation -> model.joblib
    ├── requirements-training.txt             # extra packages for the notebook
    └── data/                                 # put the Kaggle CSV here
```

## ⚠️ About the data

The bundled `model.joblib` was trained on a **synthetic stand-in dataset (2,100 rows)** because the Kaggle file
can't be downloaded without an account. The app's sidebar shows this. For a real portfolio project:

1. Download *Campus Recruitment & Employability Factors* from Kaggle and save it as
   `training/data/campus_recruitment.csv`.
2. Open the notebook and run all cells. If your column headers differ, edit the `ALIASES` dictionary in the
   *Cleaning* section (the notebook raises a clear error naming the missing column).
3. The notebook overwrites `model.joblib`, and the sidebar will then report `kaggle` as the data source.

## Method

- **Features (exact order):** `cgpa, aptitude_score, soft_skills, projects, internship, specialization`
- **Preprocessing (inside the sklearn Pipeline):** median imputation + scaling for numeric features, one-hot
  encoding for specialization. No leakage between train and test.
- **Models compared:** Logistic Regression, Random Forest, Gradient Boosting (`GradientBoostingClassifier`),
  each tuned with 5-fold stratified `GridSearchCV` on ROC-AUC.
- **Metrics:** accuracy, precision, recall, F1, ROC-AUC on a stratified 20% hold-out set. The best model is
  chosen by cross-validated ROC-AUC on training data only, then refit on all data and saved.
- **Feedback engine:** the app re-scores the student after one small change at a time (+1 project, +1 soft-skills
  point, +10 aptitude points, completing an internship, +0.5 CGPA) and reports the lever that crosses the 50%
  threshold, or the one with the biggest uplift.

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

To retrain: `pip install -r training/requirements-training.txt`, then `jupyter notebook training/`.

## Deploy on Streamlit Community Cloud

1. Push this folder to a **public GitHub repository** (include `model.joblib`; the Kaggle CSV is git-ignored).
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **Create app** → select the repo and branch, and set **Main file path** to `app.py`.
4. Under **Advanced settings**, choose Python 3.12 (or whichever version matches your local environment).
5. Click **Deploy**. Streamlit installs `requirements.txt` and serves the app at a public URL.

**Tip:** `scikit-learn` is pinned to the version used for training because `joblib` files are not guaranteed to
load across versions. If you retrain with a different version, update the pin in `requirements.txt`.

## Troubleshooting

| Symptom | Fix |
|---|---|
| "Model file not found" | Run the training notebook so `model.joblib` sits next to `app.py` |
| "Model file could not be loaded" | Retrain with the same scikit-learn version as `requirements.txt` |
| Notebook says it can't find a column | Adjust `ALIASES` in the cleaning cell to match your CSV headers |
