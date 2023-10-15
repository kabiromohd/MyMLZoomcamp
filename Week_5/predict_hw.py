from flask import Flask
from flask import request
from flask import jsonify

import os
import joblib

# Access the environment variables
model_file = os.environ.get("MODEL_FILE")
dv_file = os.environ.get("DV_FILE")

if model_file is None or dv_file is None:
    raise ValueError("MODEL_FILE and DV_FILE environment variables are not set.")

# Load the model and data vectorizer
model = joblib.load(model_file)
dv = joblib.load(dv_file)
    
app = Flask('churn')

@app.route('/predict_hw', methods=['POST'])
def predict_hw():
    client = request.get_json()

    X = dv.transform([client])
    y_pred = model.predict_proba(X)[0, 1]
    churn = y_pred >= 0.5

    result = {
        'churn_probability': float(y_pred),
        'churn': bool(churn)
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9690)