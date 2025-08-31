import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import pickle
import os

# Extra ML libraries
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# --- File paths ---
train_file = "Training.xlsx"
test_file = "Testing.xlsx"
csv_file = "patient_data.csv"

# --- Load Data ---
dataframes = []
if os.path.exists(train_file):
    df_train = pd.read_excel(train_file, engine="openpyxl")
    dataframes.append(df_train)
if os.path.exists(test_file):
    df_test = pd.read_excel(test_file, engine="openpyxl")
    dataframes.append(df_test)
if os.path.exists(csv_file):
    df_csv = pd.read_csv(csv_file)
    dataframes.append(df_csv)
    print("Merging patient_data.csv...")

# Merge everything
full_df = pd.concat(dataframes, ignore_index=True)
print(f"✅ Total dataset shape: {full_df.shape}")

# --- Preprocessing ---
X = full_df.iloc[:, :-1]
y = full_df.iloc[:, -1]

# Drop rows where target is missing
mask = y.notna()
X, y = X[mask], y[mask]

# Encode target labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Save the label encoder
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

# One-hot encode categorical columns
X = pd.get_dummies(X, drop_first=True)

# Handle missing values
imputer = SimpleImputer(strategy="median")
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# --- Train/Test Split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# --- Apply SMOTE to balance classes ---
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# --- Train Multiple Models ---
models = {
    "Logistic Regression": LogisticRegression(max_iter=3000, class_weight="balanced"),
    "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced"),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, random_state=42),
    "XGBoost": XGBClassifier(n_estimators=300, learning_rate=0.05, eval_metric="logloss", random_state=42, use_label_encoder=False),
    "LightGBM": LGBMClassifier(n_estimators=300, learning_rate=0.05, random_state=42, class_weight="balanced")
}

best_model, best_acc = None, 0
results = {}

for name, clf in models.items():
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc
    print(f"✅ {name} Accuracy: {acc:.2f}")
    
    if acc > best_acc:
        best_acc = acc
        best_model = clf

# --- Save Best Model ---
with open("model.pkl", "wb") as f:
    pickle.dump(best_model, f)

# Save training columns
with open("model_columns.pkl", "wb") as f:
    pickle.dump(X.columns, f)

# --- Save Health Reports Mapping ---
health_reports = {
    "Healthy": {
        "problem": "No major issues detected.",
        "suggestion": "Maintain a balanced diet, exercise regularly, and get good sleep."
    },
    "At Risk": {
        "problem": "Some vital signs indicate early health risks.",
        "suggestion": "Adopt a healthier lifestyle: daily exercise, more fruits/vegetables, reduce stress, and consider a check-up."
    },
    "Sick": {
        "problem": "Your parameters indicate illness.",
        "suggestion": "Consult a doctor immediately, follow prescribed medication, and ensure proper rest."
    }
}

with open("health_reports.pkl", "wb") as f:
    pickle.dump(health_reports, f)

print("\n📊 Model Comparison Results:")
for name, acc in results.items():
    print(f"{name}: {acc:.2f}")

print(f"\n🏆 Best Model Selected: {type(best_model).__name__} with Accuracy {best_acc:.2f}")
print("📁 Model saved as model.pkl")
print("📁 Model columns saved as model_columns.pkl")
print("📁 Health reports mapping saved as health_reports.pkl")