import pandas as pd

from modules.common import load_model


def predict_sales(day, month, weekday, quantity, transactions):
    model = load_model("sales_model.pkl")
    sample = pd.DataFrame({
        "day": [day],
        "month": [month],
        "day_of_week": [weekday],
        "total_quantity": [quantity],
        "num_transactions": [transactions],
    })
    pred = model.predict(sample)
    return pred[0]
