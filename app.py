

from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load the model
model = joblib.load('fish_model.pkl')

@app.route('/')
def home():
    return render_template('input.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        features = [float(x) for x in request.form.values()]
        prediction = model.predict([features])
        return render_template('result.html', prediction=prediction[0])
    
if __name__ == '__main__':
    app.run(debug=True)
