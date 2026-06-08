import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score

from xgboost import XGBClassifier


def train_models(X_train, y_train, preprocessor):

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),

        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=42
        ),

        "XGBoost": XGBClassifier(
            random_state=42,
            eval_metric="logloss"
        )
    }

    best_model = None
    best_score = 0

    for name, model in models.items():

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", model)
            ]
        )

        scores = cross_val_score(
            pipeline,
            X_train,
            y_train,
            cv=5,
            scoring="roc_auc"
        )

        mean_score = scores.mean()

        print(f"{name}: {mean_score:.4f}")

        if mean_score > best_score:
            best_score = mean_score
            best_model = pipeline

    best_model.fit(X_train, y_train)

    print(f"\nBest ROC-AUC: {best_score:.4f}")

    return best_model