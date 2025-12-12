#full stack code with pre processing add ons 
# -------------------------------------------------------------------
# STACKED ENSEMBLE MODEL – LaLiga FYP (REMEDIATED VERSION)
#
# AUDIT IMPLEMENTATION:
# 1. Temporal Integrity: Time-Series splitting instead of random split.
# 2. Distributional Robustness: RobustScaler replaces StandardScaler.
# 3. Data Hygiene: Explicit sorting before imputation.
# 4. Evaluation Focus: Recall/F1 for minority class (Champion).
# -------------------------------------------------------------------

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

# -------------------------------
# 1. LOAD AND CLEAN DATA
# -------------------------------
print("Loading and Cleaning data...")

try:
    df = pd.read_csv("MP2 DATASET.csv")
except FileNotFoundError:
    raise SystemExit("CRITICAL ERROR: Dataset not found.")

df.columns = df.columns.str.strip()

# Force ordering for forward filling
df = df.sort_index()

# FIX: Forward fill missing values AFTER sorting
df = df.ffill()

# Extract Season Year (assuming column exists called 'Year')
if "Year" not in df.columns:
    raise ValueError("Dataset missing required 'Year' column.")

df["Year"] = df["Year"].astype(str).str.extract(r"(\d{4})").astype(int)

# Define usable feature columns
# (Replace below with your real selected feature list)
features = [col for col in df.columns if col not in ["Team", "Actual_Champion", "Year"]]

df = df.dropna(subset=features)

# -------------------------------
# 2. INTEGRATE TARGET (CHAMPIONS)
# -------------------------------
print("Integrating Champion Labels...")

try:
    actual = pd.read_excel("actual_champions.xlsx")
    actual.columns = ["Season", "Champion"]
    actual["Season"] = actual["Season"].astype(str).str.extract(r"(\d{4})").astype(int)
    df = df.merge(actual, left_on="Year", right_on="Season", how="left")
    df = df.dropna(subset=["Champion"])
except Exception as e:
    raise SystemExit(f"ERROR loading champions file: {e}")

df["Champion"] = df["Champion"].astype(str)
df["Team"] = df["Team"].astype(str)

y = (df["Team"] == df["Champion"]).astype(int)
X = df[features + ["Year"]]

print("\nTarget distribution:")
print(y.value_counts())

# -------------------------------
# 3. TIME-SERIES SPLIT
# -------------------------------
unique_years = sorted(df["Year"].unique())
split_idx = int(len(unique_years) * 0.80)
split_year = unique_years[split_idx]

train_mask = df["Year"] <= split_year
test_mask = df["Year"] > split_year

X_train_full = X[train_mask].copy()
y_train = y[train_mask]
X_test_full = X[test_mask].copy()
y_test = y[test_mask]

# Remove Year from training features
X_train = X_train_full.drop(columns=["Year"])
X_test = X_test_full.drop(columns=["Year"])

# -------------------------------
# 4. MODELS & PIPELINES
# -------------------------------
base_models = [
    ("nb", GaussianNB()),
    ("rf", RandomForestClassifier()),
    ("svm", SVC(probability=True))
]

meta_model = LogisticRegression()

stacked_model = StackingClassifier(
    estimators=base_models,
    final_estimator=meta_model,
    passthrough=True,
    cv=3
)

pipelines = {
    "Naive Bayes": Pipeline([
        ("smote", SMOTE(k_neighbors=2)),
        ("scaler", RobustScaler()),
        ("model", GaussianNB())
    ]),
    
    "Random Forest": Pipeline([
        ("smote", SMOTE(k_neighbors=2)),
        ("scaler", RobustScaler()),
        ("model", RandomForestClassifier())
    ]),

    "SVM": Pipeline([
        ("smote", SMOTE(k_neighbors=2)),
        ("scaler", RobustScaler()),
        ("model", SVC(probability=True))
    ]),

    "Stacked Ensemble": Pipeline([
        ("smote", SMOTE(k_neighbors=2)),
        ("scaler", RobustScaler()),
        ("model", stacked_model)
    ]),
}

# -------------------------------
# 5. HYPERPARAMETER TUNING
# -------------------------------
tscv = TimeSeriesSplit(n_splits=3)

param_grids = {
    "Naive Bayes": {},

    "Random Forest": {
        'model__n_estimators': [100, 200],
        'model__max_depth': [5, 10, None],
        'model__min_samples_leaf': [1, 2]
    },

    "SVM": {
        'model__C': [1.0, 10.0],
        'model__gamma': ['scale', 0.1]
    },

    "Stacked Ensemble": {}
}

model_results = []

for name, pipeline in pipelines.items():
    print(f"\n--- Training {name} ---")

    grid = param_grids[name]

    gs = GridSearchCV(
        estimator=pipeline,
        param_grid=grid,
        cv=tscv,
        scoring="f1",
        n_jobs=-1
    )

    gs.fit(X_train, y_train)
    best_model = gs.best_estimator_

    preds = best_model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    pre = precision_score(y_test, preds, zero_division=0)
    rec = recall_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)

    model_results.append({
        "Model": name,
        "Accuracy": acc,
        "Precision": pre,
        "Recall": rec,
        "F1-Score": f1
    })

    print(f"Best Params: {gs.best_params_}")
    print(confusion_matrix(y_test, preds))
    print(classification_report(y_test, preds, target_names=["Others", "Champion"]))

comparison_df = pd.DataFrame(model_results).set_index("Model").sort_values(by="F1-Score", ascending=False)

print("\n=============================================")
print("🏆 FINAL PERFORMANCE REPORT (TEMPORAL VALIDATION) 🏆")
print("=============================================")
print(comparison_df.to_markdown(floatfmt=".4f"))
