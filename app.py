import os
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

TRIBES_FILE = "tribal_names.csv"
DATA_FILE = "fraud_merged_data2.csv"
PIPELINE_FILE = "pipeline.pkl"

known_tribes = []
if os.path.exists(TRIBES_FILE):
    known_tribes = pd.read_csv(TRIBES_FILE)["tribe"].astype(str).str.lower().str.strip().tolist()

def amount_severity(amount):
    if amount <= 8000:
        return 0
    elif amount <= 50000:
        return 1
    elif amount <= 100000:
        return 2
    elif amount <= 500000:
        return 3
    else:
        return 4

def build_features_dataframe(df):
    df = df.copy()
    df["amount"] = df["amount"].astype(float)
    df["hour"] = df["hour"].astype(int)
    df["device"] = df["device"].astype(str).str.lower()
    if "last_name" in df.columns and pd.notna(df.at[0, "last_name"]) and df.at[0, "last_name"]:
        df["name_proc"] = df["last_name"].astype(str)
    elif "tribe" in df.columns and pd.notna(df.at[0, "tribe"]) and df.at[0, "tribe"]:
        df["name_proc"] = df["tribe"].astype(str)
    else:
        df["name_proc"] = ""
    df["name_proc"] = df["name_proc"].str.lower().str.strip()
    df["is_known_name"] = df["name_proc"].isin(known_tribes).astype(int)
    df["amount_log"] = np.log1p(df["amount"])
    df["amount_severity"] = df["amount"].apply(amount_severity)
    df["is_night"] = ((df["hour"] < 6) | (df["hour"] > 22)).astype(int)
    return df

if os.path.exists(PIPELINE_FILE):
    pipeline = joblib.load(PIPELINE_FILE)
else:
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"{DATA_FILE} not found; needed to train model.")
    df_raw = pd.read_csv(DATA_FILE)
    df_feat = build_features_dataframe(df_raw)
    X = df_feat[["amount_log", "amount_severity", "is_known_name", "is_night", "device"]]
    y = df_feat["fraud"]
    numeric_features = ["amount_log", "amount_severity", "is_known_name", "is_night"]
    numeric_transformer = StandardScaler()
    try:
        categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse=False)
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, ["device"]),
        ],
        remainder="drop",
        sparse_threshold=0
    )
    base_model = HistGradientBoostingClassifier(random_state=42)
    calibrated = CalibratedClassifierCV(base_model, method="isotonic", cv=3)
    pipeline = Pipeline([
        ("pre", preprocessor),
        ("clf", calibrated)
    ])
    pipeline.fit(X, y)
    joblib.dump(pipeline, PIPELINE_FILE)

app = Flask(__name__)

def classify_level(score):
    if score >= 0.75:
        return "High", "عالي"
    elif score >= 0.4:
        return "Medium", "متوسط"
    else:
        return "Low", "منخفض"

@app.route("/fraud-score", methods=["POST"])
def fraud_score():
    try:
        data = request.get_json(force=True)
        amount = float(data.get("amount", 0))
        hour = int(data.get("hour", 0))
        device_raw = str(data.get("device", "")).strip().lower()
        name_input = str(data.get("last_name", "") or data.get("tribe", "")).strip().lower()
    except Exception as e:
        return jsonify({"error": "Invalid input", "details": str(e)}), 400

    row = {
        "amount": amount,
        "hour": hour,
        "device": device_raw,
        "last_name": name_input,
        "tribe": name_input,
        "fraud": 0
    }
    df_row = pd.DataFrame([row])
    df_feat = build_features_dataframe(df_row)
    X_pred = df_feat[["amount_log", "amount_severity", "is_known_name", "is_night", "device"]]
    try:
        prob = pipeline.predict_proba(X_pred)[0][1]
    except Exception as e:
        return jsonify({"error": "Prediction failed", "details": str(e)}), 500

    level_en, level_ar = classify_level(prob)

    reasons = []
    if df_feat.at[0, "is_known_name"] == 0:
        reasons.append("اسم القبيلة غير معروف")
    if amount > 100000:
        reasons.append("مبلغ كبير جدًا")
    elif amount > 8000:
        reasons.append("المبلغ يتجاوز 8000 ريال")
    if df_feat.at[0, "is_night"] == 1:
        reasons.append("العملية خارج أوقات العمل")
    if device_raw == "new":
        reasons.append("الوصول من جهاز جديد")
    else:
        try:
            encoder = pipeline.named_steps["pre"].named_transformers_["cat"]
            known_devices = encoder.categories_[0]
            if device_raw not in known_devices:
                reasons.append("الوصول من جهاز غير معروف")
        except Exception:
            pass

    return jsonify({
        "risk_score": round(float(prob), 2),
        "risk_level": {"en": level_en, "ar": level_ar},
        "reasons": reasons
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
