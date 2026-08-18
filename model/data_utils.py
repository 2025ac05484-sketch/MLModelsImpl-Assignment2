import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42
TEST_SIZE = 0.20

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)

RAW_CSV_PATH = os.path.join(
    REPO_ROOT,
    "data",
    "mushrooms.csv"
)

TEST_CSV_PATH = os.path.join(
    REPO_ROOT,
    "test_data.csv"
)


def load_raw_data(csv_path=RAW_CSV_PATH):
    """Load and encode the original Mushroom dataset."""

    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Dataset not found: {csv_path}"
        )

    df = pd.read_csv(csv_path)

    if "class" not in df.columns:
        raise ValueError(
            "Expected a 'class' column in the mushroom dataset."
        )

    # Convert target:
    # edible = 0
    # poisonous = 1
    y = df["class"].map({
        "e": 0,
        "p": 1
    })

    if y.isna().any():
        raise ValueError(
            "Found class values other than 'e'/'p'."
        )

    X_raw = df.drop(columns=["class"])

    # One-hot encode all categorical features.
    X = pd.get_dummies(
        X_raw,
        columns=X_raw.columns,
        drop_first=False
    )

    return X, y


def get_train_test_data(csv_path=RAW_CSV_PATH):
    """
    Create the fixed train/test split.

    """

    X, y = load_raw_data(csv_path)

    (
        X_train_raw,
        X_test_raw,
        y_train,
        y_test
    ) = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    # Fit scaler ONLY on training data.
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train_raw
    )

    X_test_scaled = scaler.transform(
        X_test_raw
    )

    return {
        "X_train_scaled": X_train_scaled,
        "X_test_scaled": X_test_scaled,

        "X_train_raw": X_train_raw.values,
        "X_test_raw": X_test_raw.values,

        "y_train": y_train,
        "y_test": y_test,

        "feature_columns": list(X.columns),

        "scaler": scaler
    }


def write_test_csv(csv_path=RAW_CSV_PATH):
    """
    Create test_data.csv containing the original categorical
    feature values plus the true target.

    """

    df = pd.read_csv(csv_path)

    if "class" not in df.columns:
        raise ValueError(
            "Expected a 'class' column."
        )

    y = df["class"].map({
        "e": 0,
        "p": 1
    })

    X_raw = df.drop(columns=["class"])

    (
        _,
        X_test_raw,
        _,
        y_test
    ) = train_test_split(
        X_raw,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    test_df = X_test_raw.copy()

    test_df["target"] = y_test.values

    test_df.to_csv(
        TEST_CSV_PATH,
        index=False
    )

    return TEST_CSV_PATH


def encode_like_training(
    X_raw,
    feature_columns
):
    """
    Apply the same one-hot encoding used during training
    and align the resulting columns with the training features.
    """

    X_encoded = pd.get_dummies(
        X_raw,
        columns=X_raw.columns,
        drop_first=False
    )

    X_aligned = X_encoded.reindex(
        columns=feature_columns,
        fill_value=0
    )

    return X_aligned


if __name__ == "__main__":
    output_path = write_test_csv()

    print(
        f"Test data written to: {output_path}"
    )