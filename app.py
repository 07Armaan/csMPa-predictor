from flask import Flask,render_template,request
from outliers_handler import outliers_handling
import joblib
import pandas as pd

app = Flask(__name__)
final_pipeline = joblib.load("final_pipeline.pkl")
shap_feature_names = joblib.load("shap_feature_names.pkl")

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict',methods=["POST"])
def predict():
    cement = request.form.get("cement")
    slag = request.form.get("slag")
    flyash = request.form.get("flyash") 
    water = request.form.get("water")
    superplasticizer = request.form.get("superplasticizer")
    coarseaggregate = request.form.get("coarseaggregate")
    fineaggregate = request.form.get("fineaggregate")
    age = request.form.get("age")

    test_feature_names = ['cement', 'slag', 'flyash', 'water', 'superplasticizer',
       'coarseaggregate', 'fineaggregate', 'age']

    test = [[cement, slag, flyash, water, superplasticizer,
       coarseaggregate, fineaggregate, age]]
    test_df = pd.DataFrame(test,columns=test_feature_names)
    test_pred = final_pipeline.predict(test_df)
    return render_template("index.html",prediction=f"The value of csMPa predicted is: {test_pred[0]}")

if __name__=="__main__":
    app.run(debug=True)