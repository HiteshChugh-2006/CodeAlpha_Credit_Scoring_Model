import os
import joblib

from sklearn.datasets import fetch_openml

from sklearn.model_selection import (
    train_test_split
)

from src.eda import perform_eda
from src.preprocessing import (
    get_preprocessor
)
from src.feature_engineering import (
    engineer_features
)
from src.train import train_models
from src.evaluate import evaluate_model


print("Loading Dataset...")

data = fetch_openml(
    name="credit-g",
    version=1,
    as_frame=True
)

df = data.frame

df["class"] = df["class"].map(
    {
        "good": 0,
        "bad": 1
    }
)

perform_eda(
    df,
    target_col="class"
)

df = engineer_features(df)

X = df.drop(
    "class",
    axis=1
)

y = df["class"]

X_train, X_test, y_train, y_test = (
    train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
)

preprocessor = get_preprocessor(
    X_train
)

best_model = train_models(
    X_train,
    y_train,
    preprocessor
)

evaluate_model(
    best_model,
    X_test,
    y_test
)

os.makedirs(
    "models",
    exist_ok=True
)

joblib.dump(
    best_model,
    "models/best_credit_model.pkl"
)

print("\nModel Saved Successfully")
with open("reports/metrics.txt", "w") as f:
    f.write(f"Accuracy: {accuracy_score(y_test, y_pred)}\n")
    f.write(f"Precision: {precision_score(y_test, y_pred)}\n")
    f.write(f"Recall: {recall_score(y_test, y_pred)}\n")
    f.write(f"F1: {f1_score(y_test, y_pred)}\n")
    f.write(f"ROC AUC: {roc_auc_score(y_test, y_prob)}\n")