from flask import Flask
from flask import request
from flask import jsonify

import os
import pickle

# Access the environment variables
model_file = os.environ.get("MODEL_FILE")
dv_file = os.environ.get("DV_FILE")

if model_file is None or dv_file is None:
    raise ValueError("MODEL_FILE and DV_FILE environment variables are not set.")

# Load the model and data vectorizer
with open(model_file, 'rb') as f_in: 
    model = pickle.load(f_in)
    
with open(dv_file, 'rb') as f_in2: 
    dv = pickle.load(f_in2)
    
app = Flask('credit_score')

@app.route('/predict_hw', methods=['POST'])
def predict_hw():
    client = request.get_json()

    X = dv.transform([client])
    y_pred = model.predict_proba(X)[0, 1]
    credit_score = y_pred >= 0.5

    result = {
        'Probability Client will get Credit': float(y_pred),
        'Get Credit': bool(credit_score)
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9690)