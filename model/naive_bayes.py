import os
import joblib

from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
)

from data_utils import get_train_test_data


MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "naive_bayes.joblib"
)


def train_model(X_train, y_train):

    model = MultinomialNB()

    model.fit(
        X_train,
        y_train
    )

    return model


def evaluate(model, X_test, y_test):

    y_pred = model.predict(X_test)

    y_prob = model.predict_proba(
        X_test
    )[:, 1]

    return {
        "Accuracy": accuracy_score(
            y_test,
            y_pred
        ),

        "AUC": roc_auc_score(
            y_test,
            y_prob
        ),

        "Precision": precision_score(
            y_test,
            y_pred,
            zero_division=0
        ),

        "Recall": recall_score(
            y_test,
            y_pred,
            zero_division=0
        ),

        "F1": f1_score(
            y_test,
            y_pred,
            zero_division=0
        ),

        "MCC": matthews_corrcoef(
            y_test,
            y_pred
        ),
    }


def save_model(model):

    joblib.dump(
        model,
        MODEL_PATH
    )

    print(
        f"Saved Naive Bayes model to {MODEL_PATH}"
    )


def load_model():

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Saved Naive Bayes model not found: {MODEL_PATH}"
        )

    return joblib.load(
        MODEL_PATH
    )


if __name__ == "__main__":

    data = get_train_test_data()

    # MultinomialN is using the raw, non-negative,
    # one-hot encoded features.
    model = train_model(
        data["X_train_raw"],
        data["y_train"]
    )

    metrics = evaluate(
        model,
        data["X_test_raw"],
        data["y_test"]
    )

    print("Naive Bayes results:")

    for name, value in metrics.items():
        print(
            f"  {name}: {value:.4f}"
        )

    save_model(model)