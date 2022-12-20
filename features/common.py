from functools import partial
from tqdm import tqdm

from . import features as feat


FEATURES = [
    feat.KMID,
    partial(feat.SMA, window=10),
]


TARGETS = [
    partial(feat.pct_return, window=5),
]


def generate_features(df):
    for func in tqdm(FEATURES):
        feat_name = get_feature_name(func)
        df[feat_name] = func(df)
    return df


def get_feature_name(func, prefix="X"):
    if isinstance(func, partial):
        feat_name = f"{prefix}_{func.func.__name__}_"
        feat_name += "_".join([f"{k}_{v}" for k, v in func.keywords.items()])
    else:
        feat_name = f"{prefix}_{func.__name__}"
    return feat_name


def generate_targets(df):
    for func in TARGETS:
        feat_name = get_feature_name(func, prefix="y")
        df[feat_name] = func(df)
    return df
