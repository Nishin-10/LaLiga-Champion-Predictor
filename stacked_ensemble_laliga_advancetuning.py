# -------------------------------------------------------------------
# STACKED ENSEMBLE MODEL – LaLiga FYP (Advanced Tuning Script)
#
# AIM:
# 1. Use SMOTE to handle severe class imbalance.
# 2. Use GridSearchCV to find the best hyperparameters for RF and SVM.
# 3. Compare all four models using F1, Precision, and Recall.
#
# -------------------------------------------------------------------

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    classification_report,
    confusion_matrix
)
# --- NEW IMPORTS ---
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

# -------------------------------
# 1. LOAD AND CLEAN DATA (Unchanged)
# -------------------------------
print("Loading data...")
df = pd.read_csv("MP2 DATASET.csv")

# Clean column names (e.g., " Year " -> "Year")
df.columns = df.columns.str.strip()

# Fix missing 'Year' values
print("Fixing missing 'Year' values...")
df['Year'] = df['Year'].ffill()

# Extract 4-digit Year
df["Year"] = df["Year"].astype(str).str.extract(r"(\d{4})").astype("Int64")

# Drop missing key features
df = df.dropna(subset=[
    "Team", "Short Passes pg", "Goals", "Shots pg", 
    "Through Balls pg", "Aerials Won"
])

# -------------------------------
# 2. LOAD ACTUAL CHAMPIONS FILE (Unchanged)
# -------------------------------
actual = pd.read_excel("actual_champions.xlsx")
actual["Year"] = actual["Season"].str.split("-").str[1].astype(int) + 2000  
df = df.merge(actual[["Year", "Actual_Champion"]], on="Year", how="left")
df = df.dropna(subset=["Actual_Champion"])

# -------------------------------
# 3. DEFINE FEATURES AND TARGET (Unchanged)
# -------------------------------
features = [
    "Short Passes pg", 
    "Goals", 
    "Shots pg", 
    "Through Balls pg", 
    "Aerials Won"
]
X = df[features]
y = (df["Team"] == df["Actual_Champion"]).astype(int)

print(f"\nTarget variable 'y' created.")
print(y.value_counts())

# -------------------------------
# 4. TRAIN/TEST SPLIT (Unchanged)
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.4, 
    random_state=42, 
    stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\nData split and scaled:")
print(f"X_train shape: {X_train_scaled.shape}")
print(f"Champions in y_train: {y_train.sum()}")
print(f"Champions in y_test: {y_test.sum()}")


# -------------------------------------------------------
# 5. DEFINE ALL MODELS, PIPELINES, AND GRIDS (!!! NEW !!!)
# -------------------------------------------------------

# --- Step 1: Define base models (NO class_weight) ---
# We remove 'class_weight' because SMOTE will handle the balance.

base_models = [
    ('nb', GaussianNB()),
    ('rf', RandomForestClassifier(n_estimators=200, random_state=42)),
    ('svm', SVC(kernel="rbf", probability=True, random_state=42))
]

meta_model = LogisticRegression()

stacked_model = StackingClassifier(
    estimators=base_models, 
    final_estimator=meta_model, 
    passthrough=True
)

# --- Step 2: Create a Pipeline for each model ---
# A Pipeline chains steps: 1. Apply SMOTE, 2. Train the model.
# This ensures SMOTE is *only* applied to the training data.

pipelines = {
    "Naive Bayes": Pipeline([
        ('smote', SMOTE(random_state=42, k_neighbors=3)),  # <-- FIX
        ('model', GaussianNB())
    ]),
    
    "Random Forest": Pipeline([
        ('smote', SMOTE(random_state=42, k_neighbors=3)),  # <-- FIX
        ('model', RandomForestClassifier(random_state=42))
    ]),
    
    "SVM": Pipeline([
        ('smote', SMOTE(random_state=42, k_neighbors=3)),  # <-- FIX
        ('model', SVC(kernel="rbf", probability=True, random_state=42))
    ]),
    
    "Stacked Ensemble": Pipeline([
        ('smote', SMOTE(random_state=42, k_neighbors=3)),  # <-- FIX
        ('model', stacked_model)
    ])
}

# --- Step 3: Define Hyperparameter Grids for Tuning ---
# We will tune RF and SVM. We use 'model__' to access
# the parameters inside the pipeline.
# We'll use smaller grids to make it run faster.

param_grids = {
    "Naive Bayes": {},  # Naive Bayes has no params to tune

    "Random Forest": {
        'model__n_estimators': [100, 200],
        'model__max_depth': [None, 10, 20],
        'model__min_samples_leaf': [1, 2]
    },

    "SVM": {
        'model__C': [1.0, 10.0, 50.0],
        'model__gamma': ['scale', 0.1, 0.01]
    },

    "Stacked Ensemble": {} # Tuning a stacker is very slow,
                            # so we'll just use its default SMOTE score.
}


# -----------------------------------------------
# 6. TRAIN AND EVALUATE MODELS (!!! NEW !!!)
# -----------------------------------------------

# Store results for final comparison
model_results = []

# Use cv=3 because our minority class (8 champions) is very small
# n_jobs=-1 uses all your computer's cores to speed up the search.
CV_FOLDS = 3 

for name, pipeline in pipelines.items():
    print(f"\n--- Training {name} ---")
    
    # Get the parameter grid for this model
    grid = param_grids[name]
    
    # Create the GridSearchCV object
    # We score using 'f1_weighted' to handle the imbalance
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=grid,
        cv=CV_FOLDS,
        scoring='f1_weighted', # Focus on the F1-Score
        n_jobs=-1,
        verbose=1 # Shows progress
    )
    
    # Train the grid search
    grid_search.fit(X_train_scaled, y_train)
    
    # Get the best model found by the search
    best_model = grid_search.best_estimator_
    
    # Make predictions on the test set
    preds = best_model.predict(X_test_scaled)
    
    # Calculate all 4 metrics
    acc = accuracy_score(y_test, preds)
    pre = precision_score(y_test, preds, zero_division=0)
    rec = recall_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)
    
    # Store metrics
    model_results.append({
        "Model": name,
        "Accuracy": acc,
        "Precision": pre,
        "Recall": rec,
        "F1-Score": f1
    })
    
    # Print detailed report for this model
    print(f"Results for {name}:")
    if grid: # If we did a search
        print(f"Best Params: {grid_search.best_params_}")
        
    print(f"Accuracy: {acc*100:.2f}%")
    print(f"Precision: {pre:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, preds))
    
    print("\nClassification Report:")
    print(classification_report(y_test, preds, zero_division=0, target_names=["Not Champion (0)", "Champion (1)"]))

# -------------------------------
# 7. FINAL COMPARISON (Unchanged)
# -------------------------------

comparison_df = pd.DataFrame(model_results)
comparison_df = comparison_df.set_index("Model")
comparison_df = comparison_df.sort_values(by="F1-Score", ascending=False)

print("\n=============================================")
print("🏆 FINAL MODEL PERFORMANCE COMPARISON 🏆")
print("=============================================")
print(comparison_df.to_markdown(floatfmt=".4f"))