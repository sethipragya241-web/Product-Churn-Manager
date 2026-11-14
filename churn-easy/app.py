import logging
import joblib
import pandas as pd
from flask import Flask, request, jsonify

logging.basicConfig(filename="predictions.log", level=logging.INFO)
app = Flask(__name__)
model = joblib.load("model.pkl")
count = 0


@app.route("/predict", methods=["POST"])
def predict():
    global count
    data = request.get_json()
    df = pd.DataFrame([data])[model.feature_names_in_]
    result = int(model.predict(df)[0])
    count += 1
    logging.info("%s -> %s", data, result)
    return jsonify({"churn": result})


@app.route("/metrics")
def metrics():
    return "prediction_requests_total " + str(count) + "\n"


print("API running at http://localhost:5000", flush=True)
app.run(host="0.0.0.0", port=5000)
