import pickle

from flask import Flask
from flask import request
from flask import jsonify

input_file = "model1.bin"
input_file2 = "dv.bin"

with open(input_file, 'rb') as f_in: 
    model = pickle.load(f_in)
    
with open(input_file2, 'rb') as f_in2: 
    dv = pickle.load(f_in2)
    
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