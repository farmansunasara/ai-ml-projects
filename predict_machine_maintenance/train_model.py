import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import json
from pathlib import Path
from imblearn.over_sampling import SMOTE

import matplotlib.pyplot as plt
import seaborn as sns


df=pd.read_csv('dataset/Industrial_fault_detection.csv')
print("dataset rows and coloumn", df.shape)
print(df.head())
# remove missing value row
df = df.dropna()
# split dataset feature and target
X = df.drop('Fault_Type', axis=1)
y = df['Fault_Type']

print("Class distribution:\n", df['Fault_Type'].value_counts())

smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)

print("Class distribution after SMOTE:\n", y_res.value_counts())
    

# traning and testing data split
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42, stratify=y_res)
# mean and standardiviation for training model for vast different measure
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# train a model
model = RandomForestClassifier(
    # no of tree
    n_estimators=200,          
    random_state=42, 
    # depth of the tree          
    max_depth=None,   
#  -1 for all cpu core  usage       
    n_jobs=-1                 
)
model.fit(X_train_scaled, y_train)

# Save the model and scaler
model_dir = Path('ml_models')
model_dir.mkdir(exist_ok=True)

joblib.dump(model, model_dir / 'model.joblib')
joblib.dump(scaler, model_dir / 'scaler.joblib')

# Save feature names for reference
feature_order = {
    'features': list(X.columns)
}
with open(model_dir / 'feature_order.json', 'w') as f:
    json.dump(feature_order, f, indent=2)

# evaluate model with test data
y_pred = model.predict(X_test_scaled)

# finding accuracy score of model
accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", round(accuracy * 100, 2), "%\n")

print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# visualize the confusion matrix of model accuracy
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()


# feature of model importance
importances = model.feature_importances_
feature_names = X.columns

plt.figure(figsize=(10, 6))
sorted_idx = importances.argsort()
plt.barh(feature_names[sorted_idx], importances[sorted_idx])
plt.title("Feature Importance")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.show()


# ensure ml_models directory exists inside the prediction app
ml_dir = Path('prediction') / 'ml_models'
ml_dir.mkdir(parents=True, exist_ok=True)

model_path = ml_dir / 'fault_detection_model.pkl'
scaler_path = ml_dir / 'scaler.pkl'
feature_order_path = Path('prediction') / 'feature_order.json'

joblib.dump(model, model_path)
joblib.dump(scaler, scaler_path)

# save the feature order so the web/API can accept dict inputs
feature_order = list(X.columns)
with open(feature_order_path, 'w') as f:
    json.dump(feature_order, f, indent=2)

print(f"\nModel saved to: {model_path}\nScaler saved to: {scaler_path}\nFeature order saved to: {feature_order_path}")