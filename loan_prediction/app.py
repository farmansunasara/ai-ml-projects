from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

model = joblib.load('random_forest_model.pkl')
scaler = joblib.load('script/scaler.pkl')
feature_columns = joblib.load('script/feature_columns.pkl')
categorical_features = joblib.load('script/categorical_features.pkl')

app = FastAPI(title="Loan Prediction API", version="2.0")

class LoanApplication(BaseModel):
    Gender: str
    Married: str
    Dependents: str
    Education: str
    Self_Employed: str
    ApplicantIncome: float
    CoapplicantIncome: float
    LoanAmount: float
    Loan_Amount_Term: float
    Credit_History: float
    Property_Area: str

@app.post("/predict/")
def predict_loan_status(data: LoanApplication):
    input_df = pd.DataFrame([data.dict()])

    input_df = pd.get_dummies(input_df, columns=categorical_features, drop_first=True)

    input_df = input_df.reindex(columns=feature_columns, fill_value=0)

    missing = set(feature_columns) - set(input_df.columns)
    extra = set(input_df.columns) - set(feature_columns)
    print(" Missing columns:", missing)
    print(" Extra columns:", extra)

    # Scale and predict
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]

    result = "Y" if prediction == 1 or prediction == "Y" else "N"

    return {"loan_status": result}
