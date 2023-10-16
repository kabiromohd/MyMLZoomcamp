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
    
app = Flask('credit_score')

@app.route('/predict_hw2', methods=['POST'])
def predict_hw2():
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