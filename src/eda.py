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
        plt.figure(figsize=(10,8))
        sns.heatmap(
            numeric_df.corr(),
            annot=False,
            cmap="coolwarm"
        )
        plt.title("Correlation Matrix")
        plt.savefig(f"{output_dir}/correlation_matrix.png")
        plt.close()

    print("\nEDA Completed")