
from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd

with open('random_forest_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('preprocessor.pkl', 'rb') as f:
    preprocessor = pickle.load(f)

app = FastAPI()

class HeartDiseaseInput(BaseModel):
    Age: int
    Sex: str
    ChestPainType: str
    RestingBP: int
    Cholesterol: int
    FastingBS: int
    RestingECG: str
    MaxHR: int
    ExerciseAngina: str
    Oldpeak: float
    ST_Slope: str

@app.post("/predict/")
def predict_heart_disease(data: HeartDiseaseInput):
    input_df = pd.DataFrame([data.dict()])

    input_processed = preprocessor.transform(input_df)

    prediction = model.predict(input_processed)

    return {"prediction": int(prediction[0])}