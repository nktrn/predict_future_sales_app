from flask import render_template, request

from pfs import app, model

@app.route('/')
def home():
    return render_template('predict_form.html')


@app.route('/predict',methods=['POST'])
def predict():
    feature_list = request.form.to_dict()
    predict_plot, pred = model.predict(feature_list)
    return f"""
  <p>prediction: {pred}</p>
  <img src="data:image/png;base64,{predict_plot}" alt="Red dot" />
    """
