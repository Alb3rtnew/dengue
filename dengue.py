# =============================================================================
# DENGUE CASES PREDICTION - RANDOM FOREST MODEL
# Dataset: dengue_cases_cleaned_datasets.xlsx
# Target Variable: Dengue_Cases
# Features: Month, Year, Region
# Train/Test Split: 70/30
# Evaluation Metrics: RMSE, R², MAE
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    mean_absolute_error
)

# =============================================================================
# 1. LOAD DATASET
# =============================================================================

file_path = "dengue cases cleaned datasets.xlsx"  # Make sure this file is in the same folder
df = pd.read_excel(file_path)

print("=" * 60)
print("DENGUE CASES - RANDOM FOREST REGRESSION MODEL")
print("=" * 60)
print("\n📋 Dataset Overview:")
print(f"   Shape      : {df.shape[0]} rows × {df.shape[1]} columns")
print(f"   Columns    : {list(df.columns)}")
print(f"\n   First 5 rows:")
print(df.head())
print(f"\n   Missing Values:\n{df.isnull().sum()}")
print(f"\n   Unique Regions ({df['Region'].nunique()}): {sorted(df['Region'].unique())}")

# =============================================================================
# 2. FEATURE ENGINEERING
# =============================================================================

# Encode 'Month' as a numeric value (Jan=1 ... Dec=12)
month_order = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December"
]
df["Month_Num"] = df["Month"].apply(lambda m: month_order.index(m) + 1)

# Encode 'Region' using Label Encoding
le = LabelEncoder()
df["Region_Encoded"] = le.fit_transform(df["Region"])

print("\n📐 Region Label Encoding Map:")
for orig, enc in zip(le.classes_, le.transform(le.classes_)):
    print(f"   {enc:2d} → {orig}")

# Define features (X) and target (y)
X = df[["Month_Num", "Year", "Region_Encoded"]]
y = df["Dengue_Cases"]

print(f"\n✅ Features used : {list(X.columns)}")
print(f"✅ Target variable: Dengue_Cases")

# =============================================================================
# 3. TRAIN / TEST SPLIT (70/30)
# =============================================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.30,
    random_state=42
)

print(f"\n🔀 Train/Test Split (70/30):")
print(f"   Training set : {X_train.shape[0]} samples ({X_train.shape[0]/len(X)*100:.1f}%)")
print(f"   Testing set  : {X_test.shape[0]} samples  ({X_test.shape[0]/len(X)*100:.1f}%)")

# =============================================================================
# 4. TRAIN RANDOM FOREST MODEL
# =============================================================================

rf_model = RandomForestRegressor(
    n_estimators=100,    # Number of trees
    max_depth=None,      # Trees grow fully (best for tabular data)
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1            # Use all CPU cores
)

print("\n⚙️  Training Random Forest model...")
rf_model.fit(X_train, y_train)
print("   ✅ Training complete!")

# =============================================================================
# 5. MAKE PREDICTIONS
# =============================================================================

y_pred_train = rf_model.predict(X_train)
y_pred_test  = rf_model.predict(X_test)

# =============================================================================
# 6. EVALUATION METRICS
# =============================================================================

# --- Training Metrics ---
rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
r2_train   = r2_score(y_train, y_pred_train)
mae_train  = mean_absolute_error(y_train, y_pred_train)

# --- Testing Metrics ---
rmse_test  = np.sqrt(mean_squared_error(y_test, y_pred_test))
r2_test    = r2_score(y_test, y_pred_test)
mae_test   = mean_absolute_error(y_test, y_pred_test)

print("\n" + "=" * 60)
print("📊 EVALUATION METRICS")
print("=" * 60)
print(f"\n{'Metric':<30} {'Training':>12} {'Testing':>12}")
print("-" * 56)
print(f"{'RMSE (Root Mean Sq. Error)':<30} {rmse_train:>12.4f} {rmse_test:>12.4f}")
print(f"{'R² (Coefficient of Det.)':<30} {r2_train:>12.4f} {r2_test:>12.4f}")
print(f"{'MAE (Mean Absolute Error)':<30} {mae_train:>12.4f} {mae_test:>12.4f}")
print("-" * 56)

print("\n📌 Interpretation:")
print(f"   RMSE (Test) = {rmse_test:.2f}  → Average prediction error in dengue cases")
print(f"   R²   (Test) = {r2_test:.4f} → Model explains {r2_test*100:.2f}% of variance in dengue cases")
print(f"   MAE  (Test) = {mae_test:.2f}  → On average, predictions are off by {mae_test:.0f} cases")

# =============================================================================
# 7. FEATURE IMPORTANCES
# =============================================================================

feature_names = ["Month", "Year", "Region"]
importances   = rf_model.feature_importances_

print("\n" + "=" * 60)
print("🌟 FEATURE IMPORTANCES")
print("=" * 60)
for name, imp in sorted(zip(feature_names, importances), key=lambda x: -x[1]):
    print(f"   {name:<12}: {imp:.4f}  ({imp*100:.2f}%)")

# =============================================================================
# 8. VISUALIZATIONS
# =============================================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Dengue Cases — Random Forest Model Results", fontsize=15, fontweight="bold")

# --- Plot 1: Actual vs Predicted (Test Set) ---
ax1 = axes[0, 0]
ax1.scatter(y_test, y_pred_test, alpha=0.5, color="steelblue", edgecolors="white", linewidth=0.3)
lims = [min(y_test.min(), y_pred_test.min()), max(y_test.max(), y_pred_test.max())]
ax1.plot(lims, lims, "r--", linewidth=1.5, label="Perfect Prediction")
ax1.set_xlabel("Actual Dengue Cases")
ax1.set_ylabel("Predicted Dengue Cases")
ax1.set_title("Actual vs Predicted (Test Set)")
ax1.legend()
ax1.text(0.05, 0.92, f"R² = {r2_test:.4f}\nRMSE = {rmse_test:.2f}\nMAE = {mae_test:.2f}",
         transform=ax1.transAxes, fontsize=9,
         bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.8))

# --- Plot 2: Residuals Plot ---
ax2 = axes[0, 1]
residuals = y_test.values - y_pred_test
ax2.scatter(y_pred_test, residuals, alpha=0.5, color="coral", edgecolors="white", linewidth=0.3)
ax2.axhline(0, color="black", linewidth=1.2, linestyle="--")
ax2.set_xlabel("Predicted Dengue Cases")
ax2.set_ylabel("Residuals (Actual − Predicted)")
ax2.set_title("Residuals Plot (Test Set)")

# --- Plot 3: Feature Importances ---
ax3 = axes[1, 0]
colors = ["#4C72B0", "#DD8452", "#55A868"]
bars = ax3.barh(feature_names, importances, color=colors, edgecolor="black", linewidth=0.5)
ax3.set_xlabel("Importance Score")
ax3.set_title("Feature Importances")
for bar, val in zip(bars, importances):
    ax3.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
             f"{val:.4f}", va="center", fontsize=9)
ax3.set_xlim(0, max(importances) * 1.2)

# --- Plot 4: Metrics Comparison (Train vs Test) ---
ax4 = axes[1, 1]
metrics = ["RMSE", "MAE"]
train_vals = [rmse_train, mae_train]
test_vals  = [rmse_test,  mae_test]
x = np.arange(len(metrics))
width = 0.35
ax4.bar(x - width/2, train_vals, width, label="Training", color="#4C72B0", edgecolor="black", linewidth=0.5)
ax4.bar(x + width/2, test_vals,  width, label="Testing",  color="#DD8452", edgecolor="black", linewidth=0.5)
for i, (tv, tev) in enumerate(zip(train_vals, test_vals)):
    ax4.text(i - width/2, tv + 5, f"{tv:.0f}", ha="center", fontsize=8)
    ax4.text(i + width/2, tev + 5, f"{tev:.0f}", ha="center", fontsize=8)
ax4.set_xticks(x)
ax4.set_xticklabels(metrics)
ax4.set_ylabel("Error Value")
ax4.set_title("Train vs Test Error Comparison")
ax4.legend()

plt.tight_layout()
plt.savefig("dengue_rf_results.png", dpi=150, bbox_inches="tight")
plt.show()
print("\n📁 Plot saved as: dengue_rf_results.png")

print("\n" + "=" * 60)
print("RANDOM FOREST MODELING COMPLETE")
print("=" * 60)