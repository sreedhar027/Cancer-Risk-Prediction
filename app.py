
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Cancer Risk Predictor",
    page_icon="🩺",
    layout="centered"
)

BASE_DIR = Path(__file__).resolve().parent

# --------------------------------------------------
# 2. LOAD TRAINED MODEL AND ARTIFACTS
# --------------------------------------------------

@st.cache_resource
def load_artifacts():

    model_path = BASE_DIR / "final_xgb_class_weighted.pkl"
    encoder_path = BASE_DIR / "label_encoder.pkl"
    features_path = BASE_DIR / "feature_names.pkl"

    for path in [model_path, encoder_path, features_path]:
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path.name}")

    model = joblib.load(model_path)
    le = joblib.load(encoder_path)
    feature_names = list(joblib.load(features_path))

    if not hasattr(model, "predict_proba"):
        raise ValueError("Loaded model does not support predict_proba().")

    return model, le, feature_names


try:
    model, le, FEATURE_NAMES = load_artifacts()
except Exception as e:
    st.error(f"Error loading model artifacts: {e}")
    st.stop()


# --------------------------------------------------
# 3. PAGE TITLE
# --------------------------------------------------

st.title("Cancer Risk Level Predictor")

st.write(
    "Predict the risk category (Low / Medium / High) "
    "using patient features and a trained XGBoost model."
)

st.warning(
    "Educational project only. This prediction is not a medical "
    "diagnosis and must not replace professional medical evaluation."
)

# --------------------------------------------------
# 4. INPUT PREPROCESSING
# --------------------------------------------------

def preprocess_input(df):

    missing = [
        col for col in FEATURE_NAMES
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required feature columns: {missing}"
        )

    X = df[FEATURE_NAMES].copy()

    X = X.apply(pd.to_numeric, errors="coerce")

    if X.isnull().any().any():
        bad_columns = X.columns[
            X.isnull().any()
        ].tolist()

        raise ValueError(
            f"Missing or non-numeric values in: {bad_columns}"
        )

    return X


# --------------------------------------------------
# 5. PREDICTION FUNCTION
# --------------------------------------------------

def make_predictions(X):

    predictions_encoded = np.asarray(
        model.predict(X)
    ).astype(int)

    probabilities = np.asarray(
        model.predict_proba(X)
    )

    predictions = le.inverse_transform(
        predictions_encoded
    )

    # Match probability columns to model.classes_
    model_classes = np.asarray(
        model.classes_
    ).astype(int)

    probability_classes = le.inverse_transform(
        model_classes
    )

    if probabilities.shape[1] != len(probability_classes):
        raise ValueError("Model class and probability dimensions do not match.")

    return predictions, probabilities, probability_classes


# --------------------------------------------------
# 6. SELECT PREDICTION MODE
# --------------------------------------------------

option = st.radio(
    "Select Prediction Mode",
    [
        "Upload CSV (Batch Prediction)",
        "Manual Input (Single Prediction)"
    ]
)


# --------------------------------------------------
# 7. CSV BATCH PREDICTION
# --------------------------------------------------

if option == "Upload CSV (Batch Prediction)":

    st.subheader("Upload Patient Dataset")

    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=["csv"]
    )

    if uploaded_file is not None:

        try:
            input_df = pd.read_csv(uploaded_file)

            if input_df.empty:
                raise ValueError("Uploaded CSV contains no records.")

            X = preprocess_input(input_df)

            predictions, probabilities, classes = make_predictions(X)

            result = X.copy()

            result["Predicted_Risk_Level"] = predictions

            for i, cls in enumerate(classes):
                result[f"score_{cls}"] = probabilities[:, i]

            st.success("Predictions generated successfully!")

            st.subheader("Prediction Results")

            st.dataframe(
                result,
                use_container_width=True
            )

            csv_output = result.to_csv(index=False)

            st.download_button(
                label="Download Results CSV",
                data=csv_output,
                file_name="cancer_risk_predictions.csv",
                mime="text/csv"
            )

        except Exception as e:
            st.error(f"CSV prediction error: {e}")


# --------------------------------------------------
# 8. MANUAL SINGLE PATIENT PREDICTION
# --------------------------------------------------

else:

    st.subheader("Enter Patient Features")

    st.sidebar.header("Patient Information")

    st.sidebar.caption(
        "Enter numeric values using the same feature "
        "format and encoding used during model training."
    )

    input_data = {}

    for feature in FEATURE_NAMES:

        value = st.sidebar.number_input(
            feature,
            value=0.0
        )

        input_data[feature] = value

    if st.sidebar.button("Predict Risk Level"):

        try:
            X_single = pd.DataFrame([input_data])

            X_proc = preprocess_input(X_single)

            predictions, probabilities, classes = make_predictions(
                X_proc
            )

            predicted_label = predictions[0]

            st.success("Prediction generated successfully!")

            st.subheader("Prediction Result")

            st.write(
                f"### Predicted Risk Level: {predicted_label}"
            )

            prob_df = pd.DataFrame({
                "Risk Level": classes,
                "Model Score": probabilities[0]
            })

            prob_df = prob_df.sort_values(
                "Model Score",
                ascending=False
            ).reset_index(drop=True)

            st.subheader("Class Scores")

            st.dataframe(
                prob_df.style.format({
                    "Model Score": "{:.2%}"
                }),
                use_container_width=True
            )

            if "High" in classes:
                high_index = list(classes).index("High")
                high_score = probabilities[0][high_index]

                st.metric(
                    "Model Score for High Risk",
                    f"{high_score:.2%}"
                )

                st.caption(
                    "This is the model's output score, "
                    "not a clinically validated risk estimate."
                )

        except Exception as e:
            st.error(f"Prediction error: {e}")


# --------------------------------------------------
# 9. FOOTER
# --------------------------------------------------

st.markdown("---")

st.caption(
    "Model: Optuna-tuned, class-weighted XGBoost"
)

st.caption(
    "Cancer Risk Prediction | Machine Learning Project"
)
