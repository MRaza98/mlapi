from flask import Flask, request
import joblib
import pandas as pd

app = Flask(__name__)
pipe = joblib.load('iris.mdl')

@app.route('/hello', methods=['POST'])
def hello():
    data = request.get_json()
    name = data.get('name', '')
    message = f'Hello, {name}!'
    return {'message': message}, 200

@app.route('/predict', methods=['POST'])
def predict():
    column_names = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    data = request.get_json()
    data_vector = [data.get('sepal_length', 0)
                    , data.get('sepal_width', 0)
                    , data.get('petal_length', 0)
                    , data.get('petal_width', 0)]
    X_new = pd.DataFrame([data_vector], columns=column_names)
    y_pred = pipe.predict(X_new)[0]
    return {'prediction': y_pred}, 200

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=80)
