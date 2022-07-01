from flask import Flask,render_template,request
from Cleaning_and_Imputing import Clean_and_Impute
from Preprocessing import Preprocessor
from flask_cors import cross_origin
import pandas as pd
import os

app = Flask(__name__)

@app.route("/", methods=['GET'])
@cross_origin()
def home():
    return render_template('index.html')

@app.route('/predict',methods=["POST"])
def index():
    try:
        #cleanImputeobj = Clean_and_Impute()
        #train = cleanImputeobj.readDataset(path='Training_file/SCMS_Delivery_History_Dataset.csv')
        #target = 'Freight Cost (USD)'
        #train[target] = cleanImputeobj.cleaning_feature(col=target, data=train)
        #train[target] = cleanImputeobj.cleaning_freight_col(col=target, data=train)
        #train['Weight (Kilograms)'] = cleanImputeobj.cleaning_feature(col='Weight (Kilograms)', data=train)
        #train['Weight (Kilograms)'] = cleanImputeobj.cleaning_weight_col(col='Weight (Kilograms)', data=train)
        #train = train.reset_index(drop=True)
        # train['Line Item Insurance (USD)'] = cleanImputeobj.imputing_Insurance_col(col='Line Item Insurance (USD)', data=train)
        #train['Shipment Mode'] = cleanImputeobj.impute_shipmentmode(col='Shipment Mode', data=train)
        #train['Dosage'] = cleanImputeobj.impute_dosage_col(col='Dosage', data=train)
        #train = cleanImputeobj.convertToDatetime(data=train)
        ## Preprocessing
        preprocessorobj = Preprocessor()
        train = preprocessorobj.readCleanedData(path='cleaned_data/cleaned_data.csv')
        train = preprocessorobj.removeOutliers(col='Line Item Value', data=train)
        train = preprocessorobj.encodeCatFeatures(data=train)
        train = preprocessorobj.extractingfromDatetime(data=train)
        train = preprocessorobj.removeCorrfeatures(data=train)
        preprocessorobj.trainModel(data=train)
        ## Prediction
        test = preprocessorobj.getTestDataset()
        predict_test = preprocessorobj.prediction(data=test)
        result = predict_test
        return render_template('index.html',results=result)
    except Exception as e:
        print("The exception is:", e)
        return "Something is wrong"



if __name__ == "__main__":
    app.run(port=7000,debug=True)