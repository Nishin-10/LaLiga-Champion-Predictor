import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

# ----------------------------------------------------
# 1. LOAD DATA
# ----------------------------------------------------
# Make sure the file exists in your LaligaFYP folder
df = pd.read_excel(r"C:\Users\AZ\OneDrive\Desktop\LaligaFYP\laliga_2009_2025.xlsx")
df.columns = df.columns.str.strip()

# ----------------------------------------------------
# 2. CREATE TARGET VARIABLE (Champion = 1 if Rank == 1)
# ----------------------------------------------------
df["Champion"] = (df["Rank"] == 1).astype(int)

# ----------------------------------------------------
# 3. DEFINE FEATURES AND TARGET
# ----------------------------------------------------
features = ["Short Passes pg", "Goals", "Shots pg", "Through Balls pg", "Aerials Won"]
target = "Champion"

# ----------------------------------------------------
# 4. DEFINE MODELS
# ----------------------------------------------------
gnb = GaussianNB()
rf = RandomForestClassifier(n_estimators=300, random_state=42)
svm_pipe = make_pipeline(StandardScaler(), SVC(kernel="rbf", probability=True, random_state=42))

# Stacked ensemble combines the above 3 models
stack = StackingClassifier(
    estimators=[
        ("gnb", gnb),
        ("rf", rf),
        ("svm", svm_pipe)
    ],
    final_estimator=LogisticRegression(random_state=42)
)

models = {
    "GaussianNB": gnb,
    "RandomForest": rf,
    "SVM": svm_pipe,
    "StackedEnsemble": stack
}

# ----------------------------------------------------
# 5. ROLLING-YEAR PREDICTION (TRAIN ON PAST, TEST NEXT YEAR)
# ----------------------------------------------------

# Clean and standardize the 'Year' column
df["Year"] = df["Year"].astype(str).str.extract(r"(\d{4})")  # extract 4-digit year
df = df.dropna(subset=["Year"])
df["Year"] = df["Year"].astype(int)

# Sort years
years = sorted(df["Year"].unique())

results = []  # store all evaluation metrics

print("🚀 Starting rolling-year training...")

for test_year in years[1:]:
    # Train on all previous years
    train = df[df["Year"] < test_year]
    test = df[df["Year"] == test_year]

    X_train = train[features]
    y_train = train[target]
    X_test = test[features]
    y_test = test[target]

    # Skip if not enough variation in training
    if len(y_train.unique()) < 2:
        print(f"⚠️ Skipping {test_year}: training data has only one class.")
        continue

    for model_name, model in models.items():
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)

            results.append({
                "Year": test_year,
                "Model": model_name,
                "Accuracy": acc,
                "Precision": prec,
                "Recall": rec,
                "F1_Score": f1
            })
        except Exception as e:
            print(f"⚠️ Error with {model_name} for {test_year}: {e}")

# Convert results to DataFrame
if len(results) == 0:
    print("⚠️ No valid results produced — please check data consistency.")
else:
    results_df = pd.DataFrame(results)

    # Compute average performance per model
    metrics_summary = results_df.groupby("Model")[["Accuracy", "Precision", "Recall", "F1_Score"]].mean().reset_index()

    print("\n✅ Average Model Performance Across Years:")
    print(metrics_summary)

    # Save to CSV
    results_df.to_csv("model_yearly_performance.csv", index=False)
    metrics_summary.to_csv("model_summary.csv", index=False)
    print("\n📁 Results saved as 'model_yearly_performance.csv' and 'model_summary.csv'")

# ----------------------------------------------------
# 6. CALCULATE AVERAGE METRICS PER MODEL
# ----------------------------------------------------
results_df = pd.DataFrame(results)
metrics_summary = results_df.groupby("Model")[["Accuracy", "Precision", "Recall", "F1_Score"]].mean().reset_index()

print("\n📊 Average Performance per Model (Rolling-Year Evaluation):")
print(metrics_summary.sort_values(by="Accuracy", ascending=False))

# ----------------------------------------------------
# 7. SAVE RESULTS
# ----------------------------------------------------
results_df.to_csv("model_performance_each_year.csv", index=False)
metrics_summary.to_csv("average_model_performance.csv", index=False)

print("\n✅ Saved results as:")
print(" - model_performance_each_year.csv")
print(" - average_model_performance.csv")

