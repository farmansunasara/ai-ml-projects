# PharmaEase

PharmaEase is a Streamlit-based pharmacy analytics demo that bundles trained machine learning models for four workflows:

- Inventory demand forecasting
- Prescription-to-drug recommendation
- Sales prediction
- Staff requirement estimation

## Project Structure

- `app.py`: Streamlit dashboard entrypoint
- `modules/`: Inference helpers for each business area
- `models/`: Serialized ML and time-series model artifacts
- `data/raw/`: Source datasets used for training and experimentation
- `notebooks/`: Jupyter notebooks for model development

## Modules

### Inventory Forecasting

- Uses per-drug ARIMA models stored in `models/*_arima.pkl`
- Forecasts demand for a selected medicine over a configurable number of days

### Prescription Tracking

- Uses a TF-IDF vectorizer and a classifier to recommend a likely drug from symptom text

### Sales Prediction

- Predicts revenue from calendar and transaction features

### Employee Management

- Predicts required staff count from workload features

## Run Locally

1. Create and activate a Python virtual environment.
2. Install dependencies with `pip install -r requirements.txt`.
3. Start the app with `streamlit run app.py`.

## Notes

- Model files are loaded from the repository `models/` directory.
- The dashboard is intentionally lightweight and focused on inference rather than model training.

