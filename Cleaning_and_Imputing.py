import pandas as pd
import numpy as np
from Application_logging.logger_class import App_Logger
from sklearn.linear_model import LinearRegression


class Clean_and_Impute:
    def __init__(self):
        try:
            self.logging = App_Logger()
            self.file_object =open("Logs/train_logs.txt",'a+')
            self.target = 'Freight Cost (USD)'
        except Exception as e:
            raise Exception(f"(__init__): Something went wrong on initiation process\n" + str(e))

    def readDataset(self,path):
        """
                This function helps to read the train dataset
        """
        self.logging.log(self.file_object, 'Entered the readDataset method of the Clean_and_Impute class')
        try:
            data = pd.read_csv(path)
            self.logging.log(self.file_object, 'Reading data is a success')
            return data.head()
        except Exception as e:
            raise Exception(f"(readDataset): Something went wrong while reading the dataset\n"+str(e))

    def cleaning_feature(self,col,data):
        """
         This function helps to clean the feature
        """
        self.logging.log(self.file_object, 'Entered the cleaning_feature method of the Clean_and_Impute class')
        try:
            value = []
            term = []
            num = []
            index = data[col].unique()
            for i in index:
                if i[0] == 'S':
                    value.append(i)
            for j in value:
                b = j.split(':')
                term.append(b[1])
            for i in term:
                c = i.split(')')
                num.append(int(c[0]))
            for index, id_num in zip(value, num):
                cost = data.iloc[np.where(data['ID'] == id_num)][col]
                term = str(cost)
                replace = term.split()[1]
                data[col].mask(data[col] == index, replace, inplace=True)
            return data[col]
        except Exception as e:
            raise Exception(f"(readDataset): Something went wrong while cleaning the feature\n"+str(e))

    def cleaning_freight_col(self,col,data):
        """
         This function helps to clean the Freight Cost (USD) feature
        """
        self.logging.log(self.file_object, 'Entered the cleaning_freight_col method of the Clean_and_Impute class')
        try:
            a = np.where(data[col] == 'Freight Included in Commodity Cost')
            b = np.where(data[col] == 'Invoiced Separately')
            c = np.where(data[col] == 'Invoiced')
            d = np.concatenate((a, b, c), axis=None)
            sample = data.iloc[d]
            data.drop(sample.index, inplace=True)
            data[col] = data[col].astype('float64')
            return data[col]
        except Exception as e:
            raise Exception(f"(readDataset): Something went wrong while cleaning the Freight Cost(USD) feature\n"+str(e))

    def cleaning_weight_col(self,col,data):
        """
         This function helps to clean the Weight (Kilograms) feature
        """
        self.logging.log(self.file_object, 'Entered the cleaning_weight_col method of the Clean_and_Impute class')
        try:
            data[col] = np.where(data[col] == 'Weight Captured Separately', '0.0',data[col])
            data[col] = np.where(data[col] == 'Weight', '0.0', data[col])
            data[col] = data[col].astype('float64')
            return data[col]
        except Exception as e:
            raise Exception(f"(readDataset): Something went wrong while cleaning the Weight (Kilograms) feature\n"+str(e))

    def imputing_Insurance_col(self,col,data):
        """
         This function helps to clean the Line Item Insurance (USD) feature
        """
        self.logging.log(self.file_object, 'Entered the imputing_Insurance_col method of the Clean_and_Impute class')
        try:
            df1 = data.copy()
            df1 = df1[['Line Item Value', 'Freight Cost (USD)', col]]
            df2 = df1.dropna()
            X = df2.loc[:, ('Line Item Value', 'Freight Cost (USD)')]
            y = df2.loc[:, col]
            model = LinearRegression()
            model.fit(X, y)
            df_null = df1[df1[col].isnull()]
            X_null = df_null.loc[:, ('Line Item Value', 'Freight Cost (USD)')]
            predict = model.predict(X_null)
            predict_series = pd.Series(predict, index=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                                                       11, 12, 14, 79, 80, 81, 82, 83, 84, 85, 86,
                                                       87, 88, 89, 91, 92, 170, 171, 172, 173, 174, 175,
                                                       176, 177, 178, 179, 180, 181, 182, 183, 184, 186, 250,
                                                       251, 252, 253, 254, 255, 256, 257, 314, 315, 316, 317,
                                                       318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328,
                                                       329, 330, 331, 400, 401, 402, 403, 404, 405, 406, 407,
                                                       408, 409, 410, 468, 469, 470, 471, 472, 473, 474, 475,
                                                       476, 477, 478, 479, 480, 544, 545, 546, 547, 548, 549,
                                                       550, 617, 618, 619, 620, 621, 622, 623, 624, 625, 626,
                                                       627, 628, 629, 630, 631, 632, 633, 634, 636, 637, 638,
                                                       639, 641, 642, 806, 807, 808, 809, 810, 811, 812, 813,
                                                       814, 815, 816, 817, 818, 819, 821, 822, 823, 824, 825,
                                                       826, 1005, 1006, 1007, 1008, 1009, 1010, 1011, 1012, 1013, 1014,
                                                       1015, 1016, 1017, 1018, 1019, 1020, 1021, 1022, 1023, 1026, 1029,
                                                       1178, 1179, 1180, 1181, 1182, 1183, 1184, 1185, 1186, 1187, 1188,
                                                       1189, 1190, 1191, 1192, 1193, 1197, 1198, 1200, 1201, 1202, 1203,
                                                       1204, 1403, 1404, 1405, 1406, 1407, 1408, 1409, 1410, 1411, 1412,
                                                       1413, 1414, 1415, 1416, 1417, 1418, 1419, 1420, 1424, 1425, 1426,
                                                       1626, 1627, 1628, 1629, 1630, 1631, 1632, 1633, 1634, 1635, 1636,
                                                       1637, 1638, 1641, 1642, 1643, 1644, 1645, 1646, 1647, 1649, 1832,
                                                       1833, 1834, 1835, 1836, 1837, 1838, 1839, 1842, 1843, 1844, 1846,
                                                       1847, 1848, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026, 2027,
                                                       2028, 2029, 2030, 2031, 2032, 2033, 2034, 2035, 2037, 2038,
                                                       2039])
            df1[col] = df1[col].fillna(predict_series)
            data[col] = df1[col]
            return data[col]
        except Exception as e:
            raise Exception(f"(readDataset): Something went wrong while cleaning the Line Item Insurance (USD) feature\n"+str(e))

    def impute_shipmentmode(self,col,data):
        """
         This function helps to impute the Shipment Mode feature
        """
        self.logging.log(self.file_object, 'Entered the impute_shipmentmode method of the Clean_and_Impute class')
        try:
            data[col] = np.where(data[col].isnull(),"Missing",data[col])
            return data[col]
        except Exception as e:
            raise Exception(f"(readDataset): Something went wrong while imputing the Shipment Mode feature\n"+str(e))

    def impute_dosage_col(self,col,data):
        """
         This function helps to impute the Dosage feature
        """
        self.logging.log(self.file_object, 'Entered the impute_dosage_col method of the Clean_and_Impute class')
        try:
            data[col]=data[col].fillna(0)
            return data[col]
        except Exception as e:
            raise Exception(f"(readDataset): Something went wrong while imputing the Dosage feature\n"+str(e))

    def convertToDatetime(self,data):
        """
         This function helps to convert object to datetime type
        """
        self.logging.log(self.file_object, 'Entered the convertToDatetime method of the Clean_and_Impute class')
        try:
            data['Scheduled Delivery Date'] = pd.to_datetime(data['Scheduled Delivery Date'])
            data['Delivered to Client Date'] = pd.to_datetime(data['Delivered to Client Date'])
            data['Delivery Recorded Date'] = pd.to_datetime(data['Delivery Recorded Date'])
            return data
        except Exception as e:
            raise Exception(f"(readDataset): Something went wrong while converting object to datetime type\n"+str(e))
