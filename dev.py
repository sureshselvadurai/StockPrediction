import pandas as pd

def generate_report():
    data = pd.read_csv("data_output/model/report/report.csv")
    data = data[data["Tag"] == "Prediction"]

