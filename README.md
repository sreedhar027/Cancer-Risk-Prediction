# Cancer Risk Prediction Using Machine Learning

## Project Overview

Cancer Risk Prediction is a machine learning project that predicts a patient's cancer risk category based on selected risk factors and health-related features.

The project uses a multiclass classification approach to predict one of three categories:

- **High**
- **Medium**
- **Low**

The final model is an Optuna-tuned, class-weighted XGBoost classifier. A Streamlit web application provides an interface for individual predictions and batch predictions using CSV files.

> **Disclaimer:** This project is developed for academic and educational purposes only. It is not a medical diagnostic tool. Predictions must not be used as a substitute for professional medical advice or clinical assessment.

## Technologies Used

- Python
- Pandas and NumPy
- Scikit-learn
- XGBoost
- Optuna
- Imbalanced-learn
- Joblib
- Streamlit

## Machine Learning Workflow

1. Load and explore the dataset.
2. Prepare the input features and target variable.
3. Encode the target labels.
4. Split the dataset into training and testing sets.
5. Experiment with machine learning classification models.
6. Tune XGBoost hyperparameters using Optuna.
7. Apply class weighting to address class imbalance.
8. Evaluate the final model using classification metrics.
9. Save the trained model and supporting artifacts.
10. Deploy the model using Streamlit.

## Final Model

**Algorithm:** XGBoost Classifier  
**Optimization:** Optuna hyperparameter tuning  
**Class imbalance handling:** Class weighting  
**Optimization objective:** Macro F1-score  
**Classification type:** Multiclass classification

### Model Input Features

The model uses 17 input features:

1. Age
2. Gender
3. Smoking
4. Alcohol_Use
5. Obesity
6. Family_History
7. Diet_Red_Meat
8. Diet_Salted_Processed
9. Fruit_Veg_Intake
10. Physical_Activity
11. Air_Pollution
12. Occupational_Hazards
13. BRCA_Mutation
14. H_Pylori_Infection
15. Calcium_Intake
16. BMI
17. Physical_Activity_Level

### Evaluation Metrics

The project report records the following results for the final class-weighted XGBoost model:

| Metric | Score |
|---|---:|
| Accuracy | 0.88 |
| Macro F1-score | 0.72 |
| Weighted F1-score | 0.87 |
| High-risk recall | 0.45 |
| Low-risk recall | 0.78 |
| Medium-risk recall | 0.92 |

These are the reported test results and should be checked against the final notebook outputs before formal submission.

## Web Application Features

The Streamlit application provides:

- Individual patient-feature input.
- Prediction of Low, Medium, or High risk.
- Class probability display.
- Batch predictions through CSV upload.
- Downloadable prediction results.

## Project Structure

```text
Cancer-Risk-Prediction/
│
├── app.py
├── main.py
├── README.md
├── requirements.txt
├── pyproject.toml
├── uv.lock
│
├── final_xgb_class_weighted.pkl
├── label_encoder.pkl
└── feature_names.pkl
```

The dataset and original Jupyter notebook are not currently included in this repository.

## Installation and Execution

### 1. Clone the repository

```bash
git clone https://github.com/sreedhar027/Cancer-Risk-Prediction.git
```

### 2. Navigate to the project directory

```bash
cd Cancer-Risk-Prediction
```

### 3. Create a virtual environment

```bash
python -m venv .venv
```

### 4. Activate the environment

**Windows PowerShell:**

```powershell
.venv\Scripts\Activate.ps1
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the Streamlit application

```bash
streamlit run app.py
```

The application will open in your browser.

## Model Artifacts

The following files are required by the application:

- `final_xgb_class_weighted.pkl` – trained XGBoost model.
- `label_encoder.pkl` – target label encoder.
- `feature_names.pkl` – expected model input feature names.

## Future Improvements

- Improve recall for the High-risk class.
- Perform additional model validation.
- Improve input validation and user experience.
- Add visualizations for prediction results.
- Explore model explainability techniques.

## Author

**Sreedhar**

B.Tech – Artificial Intelligence and Data Science  
S.J.C. Institute of Technology (SJCIT)

---

*Developed as an academic machine learning project.*
