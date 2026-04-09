import pandas as pd

from modules.common import load_model


def predict_staff(day, transactions, quantity):
    model = load_model("staff_model.pkl")
    sample = pd.DataFrame(
        {
            "day_of_week": [day],
            "transactions": [transactions],
            "total_quantity": [quantity],
        }
    )
    pred = model.predict(sample)
    return round(pred[0])
