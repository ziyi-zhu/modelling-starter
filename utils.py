import pandas as pd

from features.common import generate_features, generate_targets


def load_dataset(path):
    return pd.read_csv(path)


def preprocess(df):
    df = generate_features(df)
    df = generate_targets(df)
    return df.dropna()


def extract_feature_target_columns(df):
    X = df.filter(regex="^X_", axis=1)
    X.columns = X.columns.str.removeprefix("X_")
    y = df.filter(regex="^y_", axis=1)
    return X, y
