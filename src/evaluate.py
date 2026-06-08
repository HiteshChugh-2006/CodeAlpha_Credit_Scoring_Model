import os

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)


def evaluate_model(
    model,
    X_test,
    y_test,
    output_dir="reports"
):

    os.makedirs(output_dir, exist_ok=True)

    y_pred = model.predict(X_test)

    y_prob = model.predict_proba(X_test)[:, 1]

    print("\nModel Results")
    print("-" * 30)

    print(
        "Accuracy:",
        accuracy_score(y_test, y_pred)
    )

    print(
        "Precision:",
        precision_score(y_test, y_pred)
    )

    print(
        "Recall:",
        recall_score(y_test, y_pred)
    )

    print(
        "F1 Score:",
        f1_score(y_test, y_pred)
    )

    print(
        "ROC AUC:",
        roc_auc_score(y_test, y_prob)
    )

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    plt.figure(figsize=(6, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues"
    )

    plt.title("Confusion Matrix")

    plt.savefig(
        f"{output_dir}/confusion_matrix.png"
    )

    plt.close()