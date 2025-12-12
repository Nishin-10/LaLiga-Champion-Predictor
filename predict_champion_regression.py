# ----------------------------------------------------
# Predicting La Liga Champion using Regression Approach
# ----------------------------------------------------

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# -------------------------------
# 1. LOAD AND CLEAN DATA
# -------------------------------
import pandas as pd
import re

# Load the dataset
df = pd.read_excel("laliga_2009_2025.xlsx")

# Remove extra spaces from column names
df.columns = df.columns.str.strip()

# Function to convert season strings to numeric year
def parse_season(season):
    if pd.isna(season):
        return pd.NA
    match = re.findall(r'\d{4}', str(season))
    if len(match) == 0:
        return pd.NA
    elif len(match) == 1:
        return int(match[0])
    else:
        # For ranges like "2022-23", calculate mid-season
        start_year = int(match[0])
        end_year = int(match[1])
        if end_year < 100:
            end_year += (start_year // 100) * 100
        return start_year + (end_year - start_year) / 2

# Apply the function to 'Year' column
df["Year"] = df["Year"].apply(parse_season).astype("Float64")

# List of numeric columns for regression
numeric_cols = ["Rank", "Short Passes pg", "Goals", "Shots pg", 
                "Through Balls pg", "Aerials Won"]

# Convert to numeric with NaN handling
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce").astype("Float64")

# Impute missing values with median
for col in numeric_cols:
    median_value = df[col].median(skipna=True)
    df[col] = df[col].fillna(median_value)

# Optional: drop rows where 'Year' is missing (critical for time-series)
df = df.dropna(subset=["Year"]).reset_index(drop=True)

# Get sorted unique years
years = sorted(df["Year"].unique())

print("Cleaned years:", years)
print("Dataset ready for regression with shape:", df.shape)


# -------------------------------
# 2. DEFINE FEATURES AND TARGET
# -------------------------------
features = ["Short Passes pg", "Goals", "Shots pg", "Through Balls pg", "Aerials Won"]
target = "Rank"  # We’ll predict the rank numerically (1 = best)

# -------------------------------
# 3. ROLLING-YEAR PREDICTION
# -------------------------------
years = sorted(df["Year"].unique())
results = []

print("🚀 Starting regression-based rolling-year prediction...\n")

for test_year in years[1:]:
    train = df[df["Year"] < test_year]
    test = df[df["Year"] == test_year]

    if train.empty or test.empty:
        continue

    X_train = train[features]
    y_train = train[target]
    X_test = test[features]
    y_test = test[target]

    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict ranks
    y_pred = model.predict(X_test)

    # Evaluate regression performance
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Identify predicted champion (lowest predicted rank)
    predicted_champion = test.iloc[np.argmin(y_pred)]["Team"]
    actual_champion = test.loc[test["Rank"].idxmin()]["Team"]

    correct = predicted_champion == actual_champion

    results.append({
        "Test_Year": test_year,
        "Predicted_Champion": predicted_champion,
        "Actual_Champion": actual_champion,
        "Correct": correct,
        "MSE": mse,
        "R2_Score": r2
    })

# -------------------------------
# 4. SAVE AND DISPLAY RESULTS
# -------------------------------
results_df = pd.DataFrame(results)

if results_df.empty:
    print("⚠️ No valid results. Check if your dataset covers multiple years.")
else:
    accuracy = results_df["Correct"].mean() * 100
    print("\n🏆 Prediction Results by Year:\n")
    print(results_df[["Test_Year", "Predicted_Champion", "Actual_Champion", "Correct"]])
    print("\n📈 Average Champion Prediction Accuracy: {:.2f}%".format(accuracy))

    # Save to CSV
    output_file = "champion_predictions_regression.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\n💾 Results saved to: {output_file}")
