#!/usr/bin/env python
# coding: utf-8

# In[1]:


from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load trained model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input features from form
        features = [float(request.form[f'feature{i}']) for i in range(1, 5)]  # Adjust feature count
        
        # Transform features using scaler
        features = np.array(features).reshape(1, -1)
        features = scaler.transform(features)
        
        # Predict
        prediction = model.predict(features)[0]
        
        return render_template('result.html', prediction=prediction)
    except Exception as e:
        return render_template('result.html', prediction=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)

