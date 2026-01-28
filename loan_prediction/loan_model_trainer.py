
import os
import joblib
import pandas as pd
import numpy as np
import logging
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

logging.basicConfig(
    filename='script/training_log.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(message)s')
console.setFormatter(formatter)
logging.getLogger().addHandler(console)

logging.info(" Starting Loan Model Training with Versioning...")

df = pd.read_csv('data/Loan_Data.csv')
logging.info(f" Loaded dataset with shape: {df.shape}")

df.drop('Loan_ID', axis=1, inplace=True, errors='ignore')

num_cols = df.select_dtypes(exclude=['object']).columns
cat_cols = df.select_dtypes(include=['object']).columns

for col in num_cols:
    df[col] = df[col].fillna(df[col].mean())
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])
logging.info(" Missing values handled successfully.")

df['Loan_Status'] = df['Loan_Status'].map({'Y': 1, 'N': 0})
logging.info(" Target encoded successfully.")

X = df.drop('Loan_Status', axis=1)
y = df['Loan_Status']
cat_cols = [c for c in cat_cols if c != 'Loan_Status']
X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

feature_cols = X.columns.tolist()


sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X, y)
logging.info(f" After SMOTE: {dict(pd.Series(y_res).value_counts())}")

X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42, stratify=y_res)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
joblib.dump(scaler, 'script/scaler.pkl')
logging.info(" Feature scaling completed and scaler saved.")

models = {
    'Logistic Regression': LogisticRegression(random_state=42),
    'SVM': SVC(random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42)
}

def evaluate(model):
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)
    return {
        'accuracy': accuracy_score(y_test, preds),
        'precision': precision_score(y_test, preds),
        'recall': recall_score(y_test, preds),
        'f1': f1_score(y_test, preds)
    }

results = {}
for name, model in models.items():
    res = evaluate(model)
    results[name] = res
    logging.info(f" {name}: Acc={res['accuracy']:.4f}, Prec={res['precision']:.4f}, Rec={res['recall']:.4f}, F1={res['f1']:.4f}")

best_name = max(results, key=lambda x: results[x]['accuracy'])
best_acc = results[best_name]['accuracy']
best_model = models[best_name]

logging.info(f" Best model: {best_name} (Acc={best_acc:.4f})")

os.makedirs("scripts/models", exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_version_name = f"model_v{timestamp}_{best_name.replace(' ', '_')}_acc{best_acc:.4f}.pkl"
version_path = os.path.join("models", model_version_name)
joblib.dump(best_model, version_path)
logging.info(f" Saved model version: {model_version_name}")

score_file = "script/best_model_score.txt"
old_acc = 0.0
if os.path.exists(score_file):
    with open(score_file) as f:
        txt = f.read().strip()
        old_acc = float(txt) if txt else 0.0

if best_acc > old_acc:
    joblib.dump(best_model, 'script/random_forest_model.pkl')  # used by API
    with open(score_file, "w") as f:
        f.write(str(best_acc))
    logging.info(f" New best model updated for API (Improved from {old_acc:.4f} → {best_acc:.4f})")
else:
    logging.info(f" Accuracy {best_acc:.4f} not better than {old_acc:.4f}. API model unchanged.")

csv_path = "script/model_versions.csv"
record = pd.DataFrame([{
    "timestamp": timestamp,
    "model_name": best_name,
    "accuracy": best_acc,
    "version_file": model_version_name,
    "is_best": best_acc > old_acc
}])
if os.path.exists(csv_path):
    pd.concat([pd.read_csv(csv_path), record], ignore_index=True).to_csv(csv_path, index=False)
else:
    record.to_csv(csv_path, index=False)

logging.info(" Model version history updated.")
logging.info(" Training pipeline complete.\n")
