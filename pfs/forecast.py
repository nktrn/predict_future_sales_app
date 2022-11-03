from flask import render_template, request

from pfs import app, model

@app.route('/')
def home():
    return render_template('predict_form.html')


@app.route('/predict',methods=['POST'])
def predict():
    feature_list = request.form.to_dict()
    predict = model.predict(feature_list['item_id'])
    predict = round(predict)
    return f'{predict}'
