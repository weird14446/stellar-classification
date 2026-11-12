# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas>=2.0", "scikit-learn>=1.4"]
# ///
"""Preprocessing for Evolutionary Deep Learning (EDL).

Shares the exact same data policy as DL:
1. Drop sentinel row (u/g/z = -9999) -> 99,999 rows.
2. RobustScaler for photometric magnitudes + derived color indices.
3. log1p + StandardScaler for redshift.
4. Deterministic 64k/16k/20k train/val/test stratified split.
"""

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT / "star_classification.csv"
if not CSV_PATH.exists():
    CSV_PATH = ROOT.parent / "star_classification.csv"

PHOTOMETRIC = ["u", "g", "r", "i", "z"]
FEATURES = [*PHOTOMETRIC, "redshift"]
TARGET = "class"
SENTINEL = -9999.0
COLOR_INDICES = {"u-g": ("u", "g"), "g-r": ("g", "r"), "r-i": ("r", "i"), "i-z": ("i", "z")}


def load_clean(csv_path: Path = CSV_PATH) -> pd.DataFrame:
    """Load the dataset, drop sentinel rows, keep selected features + target."""
    df = pd.read_csv(csv_path)
    bad = (df[PHOTOMETRIC] == SENTINEL).any(axis=1)
    return df.loc[~bad, [*FEATURES, TARGET]].reset_index(drop=True)


def add_color_indices(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with the four color-index columns appended."""
    out = df.copy()
    for name, (a, b) in COLOR_INDICES.items():
        out[name] = out[a] - out[b]
    return out


def build_preprocessor(color_indices: bool = True):
    """ColumnTransformer implementing the scaling policy."""
    import numpy as np
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import FunctionTransformer, RobustScaler, StandardScaler

    redshift_pipe = make_pipeline(
        FunctionTransformer(np.log1p, feature_names_out="one-to-one"),
        StandardScaler(),
    )
    robust_cols = [*PHOTOMETRIC, *COLOR_INDICES] if color_indices else PHOTOMETRIC
    return ColumnTransformer(
        [
            ("robust", RobustScaler(), robust_cols),
            ("redshift", redshift_pipe, ["redshift"]),
        ],
        verbose_feature_names_out=False,
    )


def make_splits(color_indices: bool = True, return_preprocessor: bool = False):
    """Deterministic train/val/test split (64k/16k/20k), scaled features."""
    from sklearn.model_selection import train_test_split

    df = load_clean()
    if color_indices:
        df = add_color_indices(df)
    features = [*FEATURES, *COLOR_INDICES] if color_indices else FEATURES
    classes = sorted(df[TARGET].unique())  # GALAXY, QSO, STAR
    y_all = df[TARGET].map({c: i for i, c in enumerate(classes)}).to_numpy()

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        df[features], y_all, test_size=0.2, random_state=42, stratify=y_all
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.2, random_state=42, stratify=y_trainval
    )
    pre = build_preprocessor(color_indices=color_indices)
    splits = {
        "train": (pre.fit_transform(X_train), y_train),
        "val": (pre.transform(X_val), y_val),
        "test": (pre.transform(X_test), y_test),
    }
    if return_preprocessor:
        return splits, classes, pre
    return splits, classes


if __name__ == "__main__":
    splits, classes = make_splits(color_indices=True)
    print(f"Classes: {classes}")
    for name, (X, y) in splits.items():
        print(f"Split {name}: X {X.shape}, y {y.shape}")

