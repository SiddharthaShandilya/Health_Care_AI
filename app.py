from flask import Flask, render_template, request
#from flask_cors import cross_origin
from src.utils.all_utils import read_yaml, create_directory, load_model, error_value
import os
import numpy as np
import json

#config_path="config/config.yaml"
config = read_yaml("config/config.yaml")

artifacts_dir = config["artifacts"]["artifacts_dir"]
model_dir = config["artifacts"]["model"]["model_dir"]
reports_dir = config["artifacts"]["reports"]["reports_dir"]

scores_filename = config["artifacts"]["reports"]["scores"]
saved_model_filename = config["artifacts"]["model"]["xgboost_reg"]

saved_model_file_path = os.path.join(artifacts_dir, model_dir, saved_model_filename)
scores_file_path = os.path.join(artifacts_dir, reports_dir, scores_filename)

model = load_model(saved_model_file_path)
print(" model loaded")

app = Flask(__name__)

#-------------------------------------------------------------------------------
#   fetch_reports
#-------------------------------------------------------------------------------
def fetch_reports(report_path: str):
    with open(report_path, "r") as f:
        report = json.load(f)
    print(f"reports are loaded from {report_path}")
    print(report)
    return report
#-------------------------------------------------------------------------------


@app.route("/aqt_predict",methods = ["GET", "POST"])
def aqt_predict():

    model_report =  fetch_reports(scores_file_path)

    if request.method == "POST":
        
        T = float(request.form["Average_temperature"] )
        TM = float(request.form["Maximum_temperature"])
        Tm = float(request.form["Minimum_temperature"])
        SLP = float(request.form["Atmospheric_pressure_at_sea_level"])
        H = float(request.form["Average_humidity"])
        VV = float(request.form["Average_visibility"])
        V = float(request.form["Average_wind_speed"])
        VM = float(request.form["Maximum_wind_speed"])

        data = [T,TM,Tm,SLP,H,VV,V,VM]
        Xnew = np.array(data).reshape((1,-1))
        print(Xnew)

        output = model.predict(Xnew)

        
        return "<p>model fetched from {} with detail reports of {}</p>" .format(output,model_report)


@app.route("/")
@cross_origin()
def home():
    return render_template("home.html")

if __name__ == "__main__":
    #aqt_predict()
    app.run(Debug=True)
    