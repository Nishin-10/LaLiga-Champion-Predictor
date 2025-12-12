# -------------------------------------------------------------------
# LALIGA CHAMPION PREDICTION – ENHANCED
# - Stratified K-Fold CV reporting
# - Combined ROC & PR plots for all models
# - Confusion heatmaps, RF feature importance
# - PR-AUC + ROC-AUC (better for imbalanced)
# -------------------------------------------------------------------

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score
)

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

# -------------------------
# Config
# -------------------------
DATA_PATH = "MP2 DATASET.csv"
RANDOM_STATE = 42
TEST_SIZE = 0.3
K_FOLDS = 5
SAVE_FIGS = False  # set True to save plots to disk
OUT_DIR = "figs"
if SAVE_FIGS:
    os.makedirs(OUT_DIR, exist_ok=True)

# -------------------------
# Load data (with dummy fallback)
# -------------------------
try:
    df = pd.read_csv(DATA_PATH)
except FileNotFoundError:
    print(f"Warning: {DATA_PATH} not found — generating dummy data for demo.")
    n = 100
    df = pd.DataFrame({
        'Short Passes pg': np.random.randint(300, 600, n),
        'Goals': np.random.randint(30, 100, n),
        'Shots pg': np.random.uniform(10, 20, n),
        'Through Balls pg': np.random.randint(1, 10, n),
        'Aerials Won': np.random.uniform(10, 20, n),
        'Champion': [1]*10 + [0]*90
    })

df.columns = df.columns.str.strip()
df = df.fillna(0)

features = ['Short Passes pg', 'Goals', 'Shots pg', 'Through Balls pg', 'Aerials Won']
X = df[features]
y = df['Champion'].astype(int)

print("Original class counts:\n", y.value_counts())

# -------------------------
# Train/Test split (stratified)
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
)
print(f"Train: {len(X_train)}, Test: {len(X_test)}")

# Plot before/after SMOTE (viz only)
plt.figure(figsize=(6,4))
sns.countplot(x=y_train)
plt.title("Before SMOTE - train")
if SAVE_FIGS: plt.savefig(os.path.join(OUT_DIR, "before_smote.png"))
plt.show()

smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=2)
X_res, y_res = smote.fit_resample(X_train, y_train)

plt.figure(figsize=(6,4))
sns.countplot(x=y_res)
plt.title("After SMOTE (viz)")
if SAVE_FIGS: plt.savefig(os.path.join(OUT_DIR, "after_smote.png"))
plt.show()

# -------------------------
# Models & Pipelines
# -------------------------
models = {
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
    "SVM": SVC(probability=True, random_state=RANDOM_STATE),
}

# Stacking ensemble using the *base* models
stacking_clf = StackingClassifier(
    estimators=[
        ('rf', RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)),
        ('svm', SVC(probability=True, random_state=RANDOM_STATE)),
        ('nb', GaussianNB())
    ],
    final_estimator=LogisticRegression(),
    cv=5,
    n_jobs=-1
)
models["Stacked Ensemble"] = stacking_clf

pipelines = {
    name: ImbPipeline([
        ('scaler', StandardScaler()),
        ('smote', smote),            # SMOTE inside pipeline -> no leakage in CV
        ('model', model)
    ])
    for name, model in models.items()
}

# -------------------------
# Utility: safe probability extraction
# -------------------------
def get_positive_scores(estimator, X):
    """Return score-like values in [0,1] for 'positive' class."""
    try:
        probs = estimator.predict_proba(X)[:, 1]
        return probs
    except Exception:
        try:
            scores = estimator.decision_function(X)
            # scale to (0,1) via min-max for ROC/PR plotting
            smin, smax = scores.min(), scores.max()
            if smax == smin:
                return np.full(len(scores), 0.5)
            return (scores - smin) / (smax - smin)
        except Exception:
            return estimator.predict(X)

# -------------------------
# Cross-validated metrics (on training set)
# -------------------------
skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_STATE)

cv_summary = []
print("\nRunning Cross-Validation...")
for name, pipe in pipelines.items():
    # Use cross_val_score on F1 (with pipeline so SMOTE is inside)
    f1_cv = cross_val_score(pipe, X_train, y_train, scoring='f1', cv=skf, n_jobs=-1)
    auc_cv = cross_val_score(pipe, X_train, y_train, scoring='roc_auc', cv=skf, n_jobs=-1)
    cv_summary.append({
        "Model": name,
        "F1_mean": f1_cv.mean(), "F1_std": f1_cv.std(),
        "ROC_AUC_mean": auc_cv.mean(), "ROC_AUC_std": auc_cv.std()
    })
cv_df = pd.DataFrame(cv_summary).sort_values(by="F1_mean", ascending=False)
print("\nCross-validated summary (train):\n", cv_df.round(4))

# -------------------------
# Train on full train set, evaluate on test set
# -------------------------
results = []
for_plot_rocs = []

# Setup Combined PR Plot
plt_pr = plt.figure(figsize=(8,6))
ax_pr = plt_pr.add_subplot(111)

for name, pipe in pipelines.items():
    print(f"\nTraining & evaluating: {name}")
    pipe.fit(X_train, y_train)              # SMOTE applied internally
    preds = pipe.predict(X_test)
    scores = get_positive_scores(pipe, X_test)

    # metrics
    acc = accuracy_score(y_test, preds)
    pre = precision_score(y_test, preds, zero_division=0)
    rec = recall_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)
    roc_auc = roc_auc_score(y_test, scores)
    pr_auc = average_precision_score(y_test, scores) 

    results.append({
        "Model": name,
        "Accuracy": acc, "Precision": pre, "Recall": rec, "F1": f1,
        "ROC_AUC": roc_auc, "PR_AUC": pr_auc
    })

    print(f"Accuracy: {acc:.3f}  Precision: {pre:.3f}  Recall: {rec:.3f}  F1: {f1:.3f}")
    print(f"ROC AUC: {roc_auc:.3f}  PR AUC: {pr_auc:.3f}")
    
    # Confusion Matrix Text & Heatmap
    cm = confusion_matrix(y_test, preds)
    print("Confusion Matrix:\n", cm)
    
    plt.figure(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f"Confusion Matrix: {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    if SAVE_FIGS: plt.savefig(os.path.join(OUT_DIR, f"cm_{name}.png"))
    plt.show()

    # Collect for ROC Plot
    for_plot_rocs.append((name, y_test, scores))

    # Add to PR Plot
    precision, recall, _ = precision_recall_curve(y_test, scores)
    ax_pr.plot(recall, precision, label=f"{name} (AUC = {pr_auc:.2f})")

# Finalize Combined PR Plot
ax_pr.set_title("Precision-Recall Curves")
ax_pr.set_xlabel("Recall")
ax_pr.set_ylabel("Precision")
ax_pr.legend()
ax_pr.grid(True, alpha=0.3)
if SAVE_FIGS: plt_pr.savefig(os.path.join(OUT_DIR, "pr_curves.png"))
plt.show()

# Finalize Combined ROC Plot
plt.figure(figsize=(8,6))
for name, y_true, y_scores in for_plot_rocs:
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    auc_val = roc_auc_score(y_true, y_scores)
    plt.plot(fpr, tpr, label=f"{name} (AUC = {auc_val:.2f})")

plt.plot([0, 1], [0, 1], 'k--', label="Chance")
plt.title("ROC Curves")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid(True, alpha=0.3)
if SAVE_FIGS: plt.savefig(os.path.join(OUT_DIR, "roc_curves.png"))
plt.show()

# Feature Importance (Random Forest only)
rf_pipe = pipelines.get("Random Forest")
if rf_pipe:
    rf_model = rf_pipe.named_steps['model']
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(8,4))
    plt.title("Feature Importances (Random Forest)")
    plt.bar(range(X.shape[1]), importances[indices], align="center")
    plt.xticks(range(X.shape[1]), X.columns[indices], rotation=45)
    plt.tight_layout()
    if SAVE_FIGS: plt.savefig(os.path.join(OUT_DIR, "rf_features.png"))
    plt.show()

# Final Leaderboard
results_df = pd.DataFrame(results).sort_values(by="F1", ascending=False)
print("\nFinal Leaderboard (Test Set):\n", results_df.round(3))