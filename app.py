import os
import sys
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
)

# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_DIR = os.path.join(BASE_DIR, "model")
TEST_DATA_PATH = os.path.join(BASE_DIR, "test_data.csv")

sys.path.append(MODEL_DIR)

from data_utils import (
    load_raw_data,
    get_train_test_data,
    encode_like_training,
)


# ---------------------------------------------------------
# Page setup
# ---------------------------------------------------------

st.set_page_config(
    page_title="Mushroom Classification",
    layout="wide",
)


# ---------------------------------------------------------
# Simple styling
# ---------------------------------------------------------

st.markdown(
    """
    <style>

    .main-title {
        font-size: 32px;
        font-weight: 600;
        margin-bottom: 5px;
    }

    .subtitle {
        color: #666666;
        font-size: 16px;
        margin-bottom: 25px;
    }

    .section-title {
        font-size: 22px;
        font-weight: 600;
        margin-top: 25px;
        margin-bottom: 15px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# Model paths
# ---------------------------------------------------------

MODEL_PATHS = {
    "Logistic Regression": os.path.join(
        MODEL_DIR,
        "logistic_regression.joblib"
    ),

    "Decision Tree": os.path.join(
        MODEL_DIR,
        "decision_tree.joblib"
    ),

    "kNN": os.path.join(
        MODEL_DIR,
        "knn.joblib"
    ),

    "Naive Bayes": os.path.join(
        MODEL_DIR,
        "naive_bayes.joblib"
    ),

    "Random Forest": os.path.join(
        MODEL_DIR,
        "random_forest.joblib"
    ),
}


# Naive Bayes uses the raw one-hot encoded data.
# The other models use the scaled data.
RAW_MODELS = {
    "Naive Bayes"
}


# ---------------------------------------------------------
# Load models
# ---------------------------------------------------------

@st.cache_resource
def load_models():

    models = {}

    for name, path in MODEL_PATHS.items():

        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Could not find {name} model: {path}"
            )

        models[name] = joblib.load(path)

    return models


# ---------------------------------------------------------
# Load test data
# ---------------------------------------------------------

@st.cache_data
def load_test_data():

    if not os.path.exists(TEST_DATA_PATH):
        raise FileNotFoundError(
            "test_data.csv was not found."
        )

    df = pd.read_csv(TEST_DATA_PATH)

    if "target" not in df.columns:
        raise ValueError(
            "test_data.csv must contain a 'target' column."
        )

    return df


# ---------------------------------------------------------
# Calculate metrics
# ---------------------------------------------------------

def calculate_metrics(y_true, y_pred, y_prob):

    return {
        "Accuracy": accuracy_score(
            y_true,
            y_pred
        ),

        "AUC": roc_auc_score(
            y_true,
            y_prob
        ),

        "Precision": precision_score(
            y_true,
            y_pred,
            zero_division=0
        ),

        "Recall": recall_score(
            y_true,
            y_pred,
            zero_division=0
        ),

        "F1 Score": f1_score(
            y_true,
            y_pred,
            zero_division=0
        ),

        "MCC": matthews_corrcoef(
            y_true,
            y_pred
        ),
    }


# ---------------------------------------------------------
# Get predictions
# ---------------------------------------------------------

def predict(model_name, model, X_raw, X_scaled):

    if model_name in RAW_MODELS:
        X = X_raw.values
    else:
        X = X_scaled

    predictions = model.predict(X)

    probabilities = model.predict_proba(X)[:, 1]

    return predictions, probabilities


# ---------------------------------------------------------
# Main app
# ---------------------------------------------------------

def main():

    # -----------------------------------------------------
    # Title
    # -----------------------------------------------------

    st.markdown(
        '<div class="main-title">'
        'Mushroom Classification'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Comparison of five machine learning models'
        '</div>',
        unsafe_allow_html=True
    )

    st.write(
        "This application compares different machine learning "
        "models for classifying mushrooms as edible or poisonous."
    )

    # -----------------------------------------------------
    # Load everything
    # -----------------------------------------------------

    try:

        models = load_models()
        test_df = load_test_data()

    except Exception as e:

        st.error(
            f"Error loading the application: {e}"
        )

        return

    # -----------------------------------------------------
    # Prepare test data
    # -----------------------------------------------------

    y_test = test_df["target"]

    X_test_raw_original = test_df.drop(
        columns=["target"]
    )

    # Get the same feature columns used during training
    X_full, _ = load_raw_data()

    feature_columns = list(
        X_full.columns
    )

    # Apply the same one-hot encoding
    X_test_encoded = encode_like_training(
        X_test_raw_original,
        feature_columns
    )

    # Recreate the scaler used during training
    training_data = get_train_test_data()

    scaler = training_data["scaler"]

    X_test_scaled = scaler.transform(
        X_test_encoded
    )

    # -----------------------------------------------------
    # Sidebar
    # -----------------------------------------------------

    st.sidebar.header("Model Selection")

    selected_model = st.sidebar.selectbox(
        "Select a model",
        list(models.keys())
    )

    st.sidebar.markdown("---")

    st.sidebar.write(
        "Test dataset information"
    )

    st.sidebar.write(
        f"Number of test samples: {len(test_df)}"
    )

    st.sidebar.write(
        f"Number of features: {len(X_test_raw_original.columns)}"
    )

    # -----------------------------------------------------
    # Selected model
    # -----------------------------------------------------

    model = models[selected_model]

    y_pred, y_prob = predict(
        selected_model,
        model,
        X_test_encoded,
        X_test_scaled
    )

    metrics = calculate_metrics(
        y_test,
        y_pred,
        y_prob
    )

    # -----------------------------------------------------
    # Model name
    # -----------------------------------------------------

    st.header(selected_model)

    st.write(
        "The following results are calculated using the "
        "predefined test dataset."
    )

    # -----------------------------------------------------
    # Metrics
    # -----------------------------------------------------

    st.subheader("Evaluation Metrics")

    metric_data = pd.DataFrame(
        {
            "Metric": list(metrics.keys()),
            "Value": list(metrics.values())
        }
    )

    metric_data["Value"] = metric_data["Value"].apply(
    lambda x: f"{x:.4f}"
)

    st.dataframe(
        metric_data,
        use_container_width=True,
        hide_index=True
    )

    # -----------------------------------------------------
    # Confusion matrix
    # -----------------------------------------------------

    st.subheader("Confusion Matrix")

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    fig, ax = plt.subplots(
        figsize=(6, 4)
    )

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=[
            "Edible",
            "Poisonous"
        ],
        yticklabels=[
            "Edible",
            "Poisonous"
        ],
        ax=ax
    )

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(
        f"Confusion Matrix - {selected_model}"
    )

    st.pyplot(fig)

    plt.close(fig)

    # -----------------------------------------------------
    # Classification report
    # -----------------------------------------------------

    st.subheader("Classification Report")

    report = classification_report(
        y_test,
        y_pred,
        target_names=[
            "Edible",
            "Poisonous"
        ],
        output_dict=True,
        zero_division=0
    )

    report_df = pd.DataFrame(
        report
    ).transpose()

    # Keep more decimal places
    st.dataframe(
        report_df,
        use_container_width=True
    )

    # -----------------------------------------------------
    # Comparison of all models
    # -----------------------------------------------------

    st.header("Comparison of All Models")

    comparison = []

    for name, current_model in models.items():

        predictions, probabilities = predict(
            name,
            current_model,
            X_test_encoded,
            X_test_scaled
        )

        current_metrics = calculate_metrics(
            y_test,
            predictions,
            probabilities
        )

        row = {
            "Model": name,
            "Accuracy": current_metrics["Accuracy"],
            "AUC": current_metrics["AUC"],
            "Precision": current_metrics["Precision"],
            "Recall": current_metrics["Recall"],
            "F1 Score": current_metrics["F1 Score"],
            "MCC": current_metrics["MCC"],
        }

        comparison.append(row)

    comparison_df = pd.DataFrame(
        comparison
    )

    # Show complete floating point values
    st.dataframe(
        comparison_df,
        use_container_width=True,
        hide_index=True
    )

    # -----------------------------------------------------
    # Best model
    # -----------------------------------------------------

    best_model = comparison_df.loc[
        comparison_df["Accuracy"].idxmax()
    ]

    st.subheader("Best Model Based on Accuracy")

    st.write(
        f"Model: {best_model['Model']}"
    )

    st.write(
        f"Accuracy: {best_model['Accuracy']}"
    )

    # -----------------------------------------------------
    # Dataset information
    # -----------------------------------------------------

    st.header("Test Dataset")

    st.write(
        "The application uses the predefined test_data.csv "
        "file."
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Test Samples",
        len(test_df)
    )

    col2.metric(
        "Features",
        len(X_test_raw_original.columns)
    )

    col3.metric(
        "Models",
        len(models)
    )


if __name__ == "__main__":
    main()