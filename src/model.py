import os
import joblib
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
from src.data_preprocessing import preprocess_data


def train_xgb_model(csv_path):

    # 1️⃣ Preprocess
    X_train, X_test, y_train, y_test, scaler = preprocess_data(csv_path)

    # 2️⃣ Model
    model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42
    )

    # 3️⃣ Train
    model.fit(X_train, y_train)

    # 4️⃣ Evaluate
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    # 5️⃣ HARD ABSOLUTE PATH (NO RELATIVE PATHS)
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

    os.makedirs(MODELS_DIR, exist_ok=True)

    model_file = os.path.join(MODELS_DIR, "xgb_model.pkl")
    scaler_file = os.path.join(MODELS_DIR, "scaler.pkl")

    joblib.dump(model, model_file)
    joblib.dump(scaler, scaler_file)

    print("✅ Model saved at:", model_file)
    print("✅ Scaler saved at:", scaler_file)
