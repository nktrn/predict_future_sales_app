from flask import render_template, request

from pfs import app, model
import pandas as pd


@app.route('/')
def home():
    return render_template('predict_form.html')


@app.route('/predict',methods=['POST'])
def predict():
    feature_list = request.form.to_dict()
    predict, img = model.predict(feature_list)

    return f"""
    <p>Prediction: {predict}</p>
    <div style="text-align: center">{img}</div>
    """