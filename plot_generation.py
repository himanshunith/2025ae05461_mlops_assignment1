import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# 1. Locate and load the dataset
csv_paths = [
    "heart_disease_dataset/heart.csv",
    "heart.csv",
    "heart_disease_dataset/heart_disease.csv",
    "data/heart.csv",
]

df = None
for path in csv_paths:
    if os.path.exists(path):
        df = pd.read_csv(path)
        print(f"Loaded dataset from: {path}")
        break

if df is None:
    # Fallback: Load directly from UCI repository if local file isn't found
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    columns = [
        "age",
        "sex",
        "cp",
        "trestbps",
        "chol",
        "fbs",
        "restecg",
        "thalach",
        "exang",
        "oldpeak",
        "slope",
        "ca",
        "thal",
        "target",
    ]
    df = pd.read_csv(url, names=columns, na_values="?")
    df["target"] = df["target"].apply(lambda x: 1 if x > 0 else 0)
    print("Loaded dataset from online UCI repository.")

# Clean missing values if any
df = df.dropna()

# Normalize target column name if needed
target_col = "target" if "target" in df.columns else df.columns[-1]

# 2. Set up the figure layout (1 row x 3 columns or 2x2 grid)
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.25)

# --- Subplot 1: Class Balance ---
ax1 = fig.add_subplot(gs[0, 0])
palette = ["#2b5c8f", "#d95f02"]
counts = df[target_col].value_counts()
sns.barplot(x=counts.index, y=counts.values, palette=palette, ax=ax1)
ax1.set_title("1. Target Class Balance", fontsize=14, weight="bold")
ax1.set_xlabel("Heart Disease Presence (0 = No, 1 = Yes)", fontsize=11)
ax1.set_ylabel("Patient Count", fontsize=11)
for i, v in enumerate(counts.values):
    ax1.text(i, v + 2, f"{v} ({v/len(df)*100:.1f}%)", ha="center", fontweight="bold")

# --- Subplot 2: Feature Histograms (Key Numerical Features) ---
ax2 = fig.add_subplot(gs[0, 1])
key_features = [col for col in ["age", "chol", "thalach", "trestbps"] if col in df.columns]
for feature in key_features[:3]:
    sns.kdeplot(df[feature], ax=ax2, label=feature.upper(), fill=True, alpha=0.3)
ax2.set_title("2. Key Numerical Distributions (KDE / Histogram)", fontsize=14, weight="bold")
ax2.set_xlabel("Value", fontsize=11)
ax2.set_ylabel("Density", fontsize=11)
ax2.legend(title="Features")

# --- Subplot 3: Correlation Heatmap ---
ax3 = fig.add_subplot(gs[1, :])
corr = df.corr()
mask = None  # or np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(
    corr,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    square=False,
    linewidths=0.5,
    cbar_kws={"shrink": 0.8},
    ax=ax3,
)
ax3.set_title("3. Feature Correlation Heatmap", fontsize=14, weight="bold")

# Save output
os.makedirs("reports", exist_ok=True)
output_path = "reports/eda_plots.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight")
plt.close()

print(f"\nEDA plot generated successfully and saved to: {output_path}")