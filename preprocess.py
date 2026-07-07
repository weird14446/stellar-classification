# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas>=2.0", "scikit-learn>=1.4"]
# ///
"""Preprocessing per the policy in memory/2026-07-07.md.

1. Drop the sentinel row (u/g/z = -9999) -> 99,999 rows.
2. Keep outliers (redshift tail is QSO signal; photometric tails are
   physically valid objects).
3. Features: u, g, r, i, z, redshift. IDs and alpha/delta excluded.
4. Scaling: redshift -> log1p + StandardScaler, photometric -> RobustScaler.

The scaler must be fit on the training split only. Run this file
directly for a demo: uv run preprocess.py
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT / "star_classification.csv"

PHOTOMETRIC = ["u", "g", "r", "i", "z"]
FEATURES = [*PHOTOMETRIC, "redshift"]
TARGET = "class"
SENTINEL = -9999.0
# Adjacent-band color indices, candidate derived features.
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


def build_preprocessor(color_indices: bool = False):
    """ColumnTransformer implementing the scaling policy.

    Fit on the training split only; sklearn is imported lazily so that
    figure scripts can import load_clean without it. With
    color_indices=True the four derived columns (added beforehand via
    add_color_indices) are RobustScaler-scaled like the magnitudes.
    """
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


def main() -> None:
    from sklearn.model_selection import train_test_split

    df = load_clean()
    print(f"rows after cleaning: {len(df):,}")
    print(df[TARGET].value_counts().to_string())

    X_train, X_test, y_train, y_test = train_test_split(
        df[FEATURES], df[TARGET], test_size=0.2, random_state=42, stratify=df[TARGET]
    )
    pre = build_preprocessor()
    Xt_train = pd.DataFrame(pre.fit_transform(X_train), columns=pre.get_feature_names_out())
    Xt_test = pd.DataFrame(pre.transform(X_test), columns=pre.get_feature_names_out())

    print(f"\ntrain {Xt_train.shape}  test {Xt_test.shape}")
    print("\n=== scaled train feature stats ===")
    print(
        Xt_train.describe().T[["mean", "std", "min", "max"]].to_string(
            float_format=lambda x: f"{x:.4f}"
        )
    )


if __name__ == "__main__":
    main()
