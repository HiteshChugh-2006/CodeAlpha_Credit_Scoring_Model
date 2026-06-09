import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def perform_eda(df, target_col, output_dir="reports"):

    os.makedirs(output_dir, exist_ok=True)

    print("\nDataset Shape:")
    print(df.shape)

    print("\nMissing Values:")
    print(df.isnull().sum())

    print("\nData Types:")
    print(df.dtypes)

    # Target Distribution
    plt.figure(figsize=(6,4))
    sns.countplot(x=target_col, data=df)
    plt.title("Target Distribution")
    plt.savefig(f"{output_dir}/target_distribution.png")
    plt.close()

    # Correlation Matrix
    numeric_df = df.select_dtypes(include=['int64','float64'])

    if len(numeric_df.columns) > 1:
        plt.figure(figsize=(12, 10))

        sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    linewidths=0.5
)

plt.title("Correlation Matrix", fontsize=16)
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)

plt.tight_layout()

plt.savefig(
    "reports/correlation_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nEDA Completed")