import pandas as pd


def engineer_features(df: pd.DataFrame):

    df = df.copy()

    if (
        "credit_amount" in df.columns
        and "duration" in df.columns
    ):
        df["amount_per_month"] = (
            df["credit_amount"]
            / (df["duration"] + 1)
        )

    return df