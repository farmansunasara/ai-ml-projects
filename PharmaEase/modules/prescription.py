from modules.common import load_model


def predict_drug(symptoms):
    model = load_model("prescription_model.pkl")
    vectorizer = load_model("tfidf_vectorizer.pkl")
    vec = vectorizer.transform([symptoms])
    pred = model.predict(vec)
    return pred[0]
