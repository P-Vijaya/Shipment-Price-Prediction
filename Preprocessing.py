import pandas as pd
import numpy as np
import pickle
from Application_logging.logger_class import App_Logger
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from flask import request


class Preprocessor:
    def __init__(self):
        try:
            self.logging = App_Logger()
            self.file_object =open("Logs/train_logs.txt",'a+')
            self.target = 'Freight Cost (USD)'
        except Exception as e:
            raise Exception(f"(__init__): Something went wrong on initiation process\n" + str(e))

    def readCleanedData(self,path):
        """
                This function helps to read the cleaned data
        """
        self.logging.log(self.file_object, 'Entered the readCleanedData method of the Preprocessor class')
        try:
            data = pd.read_csv(path)
            self.logging.log(self.file_object, 'Reading data is a success')
            return data
        except Exception as e:
            raise Exception(f"(readDataset): Something went wrong while reading the dataset\n"+str(e))

    def removeOutliers(self,col,data):
        """
                This function helps to remove outliers in the dataset
        """
        self.logging.log(self.file_object, 'Entered the removeOutliers method of the Preprocessor class')
        try:
            sorted_data = data[col].sort_values(ascending=True)
            quantile1, quantile3 = np.percentile(sorted_data, [25, 75])
            iqr = quantile3 - quantile1
            lower_bound = quantile1 - (1.5 * iqr)
            upper_bound = quantile3 + (1.5 * iqr)
            outliers = [i for i in sorted_data if i < lower_bound or i > upper_bound]
            index = []
            outlier_index = []
            for i in outliers:
                index.append(data['Line Item Value'][data['Line Item Value'] == i].index.tolist())
            for i in index:
                for j in i:
                    outlier_index.append(j)
            data = data.drop(outlier_index, axis=0)
            return data
        except Exception as e:
            raise Exception(f"(readDataset): Something went wrong while removing outliers in the dataset\n"+str(e))

    def encodeCatFeatures(self,data):
        """
                This function helps to encode the categorical features
        """
        self.logging.log(self.file_object, 'Entered the encodeCatFeatures method of the Preprocessor class')
        try:
            freq_encoding_columns = ['Project Code', 'Country', 'Vendor INCO Term', 'Shipment Mode', 'Product Group',
                                     'Sub Classification', 'Vendor', 'Molecule/Test Type', 'Brand', 'Dosage',
                                     'Dosage Form', 'Manufacturing Site']
            simple_mapping_columns = ['Managed By', 'Fulfill Via', 'First Line Designation']
            for i in freq_encoding_columns:
                freq = data.groupby(i).size() / len(data)
                data[i + '_fq_en'] = data[i].map(freq)
            data['First Line_Designation_mapping'] = data['First Line Designation'].apply(
                lambda x: 1 if x == 'Yes' else (0 if x == 'No' else None))
            data['Fulfill Via_mapping'] = data['Fulfill Via'].apply(
                lambda x: 1 if x == 'Direct Drop' else (0 if x == 'From RDC' else None))
            data['Managed By_mapping'] = data['Managed By'].apply(
                lambda x: 1 if x == 'PMO - US' else (0 if x == 'Haiti Field Office' else None))
            data = data.drop(
                ['Project Code', 'PQ #', 'PO / SO #', 'ASN/DN #', 'Country', 'Vendor INCO Term', 'Shipment Mode',
                 'PQ First Sent to Client Date', 'PO Sent to Vendor Date', 'Product Group', 'Sub Classification',
                 'Vendor', 'Item Description', 'Molecule/Test Type', 'Brand', 'Dosage', 'Dosage Form',
                 'Manufacturing Site', 'Managed By', 'Fulfill Via', 'First Line Designation'], axis=1)
            return data
        except Exception as e:
            raise Exception(f"(readDataset): Something went wrong while encoding the categorical features\n"+str(e))

    def extractingfromDatetime(self,data):
        """
                This function helps to extract features from the datetime columns
        """
        self.logging.log(self.file_object, 'Entered the extractingfromDatetime method of the Preprocessor class')
        try:
            data['Scheduled Delivery Date'] = pd.to_datetime(data['Scheduled Delivery Date'])
            data['Delivered to Client Date'] = pd.to_datetime(data['Delivered to Client Date'])
            data['Delivery Recorded Date'] = pd.to_datetime(data['Delivery Recorded Date'])
            data['Scheduled Delivery Date_year'] = data['Scheduled Delivery Date'].dt.year
            data['Scheduled Delivery Date_month'] = data['Scheduled Delivery Date'].dt.month
            data['Scheduled Delivery Date_day'] = data['Scheduled Delivery Date'].dt.day

            data['Delivered to Client Date_year'] = data['Delivered to Client Date'].dt.year
            data['Delivered to Client Date_month'] = data['Delivered to Client Date'].dt.month
            data['Delivered to Client Date_day'] = data['Delivered to Client Date'].dt.day

            data['Delivery Recorded Date_year'] = data['Delivery Recorded Date'].dt.year
            data['Delivery Recorded Date_month'] = data['Delivery Recorded Date'].dt.month
            data['Delivery Recorded Date_day'] = data['Delivery Recorded Date'].dt.day

            data = data.drop(['Scheduled Delivery Date', 'Delivered to Client Date', 'Delivery Recorded Date'], axis=1)
            return data
        except Exception as e:
            raise Exception(f"(readDataset): Something went wrong while extracting features from the datetime columns\n"+str(e))

    def removeCorrfeatures(self,data):
        """
                This function helps to remove correlated features from the dataset
        """
        self.logging.log(self.file_object, 'Entered the removeCorrfeatures method of the Preprocessor class')
        try:
            df1 = data.copy()
            data_correlation = data.corr()
            columns = np.full((data_correlation.shape[0],), True, dtype=bool)
            for i in range(data_correlation.shape[0]):
                for j in range(i + 1, data_correlation.shape[0]):
                    if data_correlation.iloc[i, j] >= 0.9:
                        if columns[j]:
                            columns[j] = False
            selected_columns = data.columns[columns]
            dataset_corr = data[selected_columns]
            data = pd.concat([dataset_corr, df1['Freight Cost (USD)']], axis=1)
            data = data.drop(
                ['ID', 'Unit of Measure (Per Pack)', 'Line Item Value', 'Unit Price', 'Project Code_fq_en',
                 'Molecule/Test Type_fq_en', 'Brand_fq_en', 'Dosage_fq_en', 'Manufacturing Site_fq_en',
                 'Delivered to Client Date_day', 'Delivery Recorded Date_day'], axis=1)
            return data
        except Exception as e:
            raise Exception(f"(readDataset): Something went wrong while removing correlated features from the dataset\n"+str(e))

    def trainModel(self,data):
        """
                This function helps to train the dataset using models
        """
        self.logging.log(self.file_object, 'Entered the trainModel method of the Preprocessor class')
        try:
            y = data['Freight Cost (USD)']
            X = data.drop('Freight Cost (USD)', axis=1)
            ## scaling the dataset
            scaler = StandardScaler()
            scaled_x = scaler.fit_transform(X)
            pickle.dump(scaler, open('Best_model/standardScalar.sav', 'wb'))
            ## splitting the train and test data
            X_train, X_test, y_train, y_test = train_test_split(scaled_x, y, test_size=0.3, random_state=42)
            models = {
                'Linear Regression': LinearRegression(),
                #'Linear Regression(Ridge)': Ridge(),
                #'Linear Regression(lasso)': Lasso(),
                #'Support vector Regression': SVR(),
                'Random Forest Regressor': RandomForestRegressor(),
                #'XGBoost Regressor': XGBRegressor(),
                #'Catboost': CatBoostRegressor(verbose=0),
                #'Light Gradient boosting': LGBMRegressor()
            }

            for model_name, model in models.items():
                model.fit(X_train, y_train)
                print(model_name, 'trained')
            ## Selecting the best parameters using hyperparameter tuning
            rfrs_model = RandomForestRegressor(n_estimators=108, min_samples_split=2, min_samples_leaf=1,
                                               criterion='absolute_error',
                                               random_state=10)
            rfrs_model.fit(X_train, y_train)
            y_pred_rfrs = rfrs_model.predict(X_test)
            rmse = np.sqrt(np.mean((y_test-y_pred_rfrs) ** 2))
            print("RMSE score after hyperparameter tuning:",rmse)
            self.logging.log(self.file_object, 'Selecting the best parameters using hyperparameter tuning is a success')
            ## saving the model
            pickle.dump(rfrs_model, open('Best_model/modelForPrediction.sav', 'wb'))
            self.logging.log(self.file_object, 'Saving the best model')
            return data
        except Exception as e:
            raise Exception(f"(readDataset): Something went wrong while training the dataset using models\n"+str(e))

    def getTestDataset(self):
        """
                This function helps to collect the test data
        """
        self.logging.log(self.file_object, 'Entered the getTestDataset method of the Preprocessor class')
        try:
            Line_item_quantity = int(request.form['Line_item_quantity'])
            Pack_Price = float(request.form['Pack Price'])
            Weight = float(request.form['Weight (Kilograms)'])
            Scheduled_Delivery_year = int(request.form['Scheduled Delivery Date_year'])
            Scheduled_Delivery_month = int(request.form['Scheduled Delivery Date_month'])
            Scheduled_Delivery_day = int(request.form['Scheduled Delivery Date_day'])
            Country = request.form['Country']
            Inco_term = request.form['Vendor INCO Term']
            Shipment_Mode = request.form['Shipment Mode']
            Product_Group = request.form['Product Group']
            Sub_Classification = request.form['Sub Classification']
            Dosage_Form = request.form['Dosage Form']
            First_Line_Designation = request.form['First Line Designation']
            Fulfill_Via = request.form['Fulfill Via']
            Managed_By = request.form['Managed By']
            test =[Line_item_quantity,Pack_Price,Weight,Country,Inco_term,Shipment_Mode,Product_Group,
                   Sub_Classification,Dosage_Form,First_Line_Designation,Fulfill_Via,Managed_By,
                   Scheduled_Delivery_year,Scheduled_Delivery_month,Scheduled_Delivery_day]
            test = np.array([test])
            cols = ['Line Item Quantity', 'Pack Price', 'Weight (Kilograms)',
                    'Country_fq_en', 'Vendor INCO Term_fq_en', 'Shipment Mode_fq_en',
                    'Product Group_fq_en', 'Sub Classification_fq_en', 'Dosage Form_fq_en',
                    'First Line_Designation_mapping', 'Fulfill Via_mapping',
                    'Managed By_mapping', 'Scheduled Delivery Date_year',
                    'Scheduled Delivery Date_month', 'Scheduled Delivery Date_day']
            test = pd.DataFrame(test,columns=cols)
            return test
        except Exception as e:
            raise Exception(f"(readDataset): Something went wrong while collecting the test data\n"+str(e))

    def prediction(self,data):
        """
                This function helps to predict the test data
        """
        self.logging.log(self.file_object, 'Entered the prediction method of the Preprocessor class')
        try:
            freq_encoding_columns = ['Country_fq_en', 'Vendor INCO Term_fq_en', 'Shipment Mode_fq_en',
                                     'Product Group_fq_en', 'Sub Classification_fq_en', 'Dosage Form_fq_en', ]
            simple_mapping_columns = ['Managed By_mapping', 'Fulfill Via_mapping', 'First Line_Designation_mapping']
            for i in freq_encoding_columns:
                freq = data.groupby(i).size() / len(data)
                data[i] = data[i].map(freq)
            data['First Line_Designation_mapping'] = data['First Line_Designation_mapping'].apply(
                lambda x: 1 if x == 'Yes' else (0 if x == 'No' else None))
            data['Fulfill Via_mapping'] = data['Fulfill Via_mapping'].apply(
                lambda x: 1 if x == 'Direct Drop' else (0 if x == 'From RDC' else None))
            data['Managed By_mapping'] = data['Managed By_mapping'].apply(
                lambda x: 1 if x == 'PMO - US' else (0 if x == 'Haiti Field Office' else None))
            with open("Best_model/standardScalar.sav",'rb') as f:
                scaler = pickle.load(f)
            data_scaled = scaler.transform(data)
            with open("Best_model/modelForPrediction.sav",'rb') as f:
                model = pickle.load(f)
            predict_test = model.predict(data_scaled)
            for i in predict_test:
                return i
        except Exception as e:
            raise Exception(f"(readDataset): Something went wrong while predicting the test data \n"+str(e))