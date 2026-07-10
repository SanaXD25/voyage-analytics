
import os
import joblib
import pandas as pd
from flask import Flask, request, jsonify

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "flight_price_model.joblib")

app = Flask(__name__)
model = None


def load_model():
    global model
    if model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}. Run `python src/train_price_model.py` first."
            )
        model = joblib.load(MODEL_PATH)
    return model


REQUIRED_FIELDS = ["from", "to", "flightType", "agency", "distance", "time", "month", "day_of_week"]


@app.route("/health", methods=["GET"])
def health():
    status = "ok" if os.path.exists(MODEL_PATH) else "model_missing"
    return jsonify({"status": status}), 200


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)
    if data is None:
        return jsonify({"error": "Request body must be JSON"}), 400

    missing = [f for f in REQUIRED_FIELDS if f not in data]
    if missing:
        return jsonify({"error": f"Missing required fields: {missing}"}), 400

    try:
        m = load_model()
        X = pd.DataFrame([{
            "from": data["from"],
            "to": data["to"],
            "flightType": data["flightType"],
            "agency": data["agency"],
            "distance": float(data["distance"]),
            "time": float(data["time"]),
            "month": int(data["month"]),
            "day_of_week": int(data["day_of_week"]),
        }])
        pred = float(m.predict(X)[0])
        return jsonify({"predicted_price": round(pred, 2)}), 200
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 503
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {e}"}), 400


if __name__ == "__main__":
    load_model()
    app.run(host="0.0.0.0", port=5000)
