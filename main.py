import json
import string
import numpy as np
import pandas as pd
from fastapi import FastAPI
# from tensorflow import keras
from pydantic import BaseModel
from scipy.stats import mode
# import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
import pickle
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.naive_bayes import MultinomialNB
import joblib

# class Symptoms(BaseModel):
#     sym1: str("")
#     sym2: str("")
#     sym3: str("")
#     sym4: str("")
        
# Load model
final_svm_model = joblib.load(open('model/svm_model.joblib', 'rb'))

# Load input scaling parameters
# with open('scaling.json') as f:
#     s = json.load(f)

app = FastAPI()

@app.get("/predict")
async def predictDisease(): #(d_symptoms):

    # f1 = d_symptoms.sym1
    # f2 = d_symptoms.sym2
    # f3 = d_symptoms.sym3
    # f4 = d_symptoms.sym4
    symptoms = ("Itching,Skin Rash,Nodal Skin Eruptions") #[f1, f2, f3, f4]


    DATA_PATH = "data/Training.csv"
    data = pd.read_csv(DATA_PATH).dropna(axis = 1)
    X = data.iloc[:,:-1]
    symptom = X.columns.values

    # value using LabelEncoder
    encoder = LabelEncoder()
    data["prognosis"] = encoder.fit_transform(data["prognosis"])

    # Creating a symptom index dictionary to encode the
    # input symptoms into numerical form
    symptom_index = {}
    for index, value in enumerate(symptoms):
        symptom = " ".join([i.capitalize() for i in value.split("_")])
        symptom_index[symptom] = index

    data_dict = {
	    "symptom_index":symptom_index,
	    "predictions_classes":encoder.classes_
    }
    symptoms = symptoms.split(",")
	
	# creating input data for the models
    input_data = [0] * len(data_dict["symptom_index"])
    for sym in symptoms:
        index = data_dict["symptom_index"][sym]
        input_data[index] = 1
		
	# reshaping the input data and converting it
	# into suitable format for model predictions
    input_data = np.array(input_data).reshape(1,-1)
	
	# generating individual outputs
    svm_prediction = data_dict["predictions_classes"][final_svm_model.predict(input_data)[0]]
	
	# making final prediction by taking mode of all predictions
    final_prediction = mode([svm_prediction])[0][0]
    predictions = {
		# "rf_model_prediction": rf_prediction,
		# "naive_bayes_prediction": nb_prediction,
		"svm_model_prediction": svm_prediction,
		"final_prediction":final_prediction
	}
    return predictions

# async def get_predition (f1, f2, f3, f4): #(d_symptoms: Symptoms):

#     # f1 = d_symptoms.sym1
#     # f2 = d_symptoms.sym2
#     # f3 = d_symptoms.sym3
#     # f4 = d_symptoms.sym4

#     # Reading the train.csv by removing the
#     # # last column since it's an empty column
#     DATA_PATH = "/content/gdrive/MyDrive/disease_dataset/Training.csv"
#     data = pd.read_csv(DATA_PATH).dropna(axis = 1)

#     encoder = LabelEncoder()
#     data["prognosis"] = encoder.fit_transform(data["prognosis"])

#     X_scaled = [[f1, f2, f3, f4]]
#     X = data.iloc[:,:-1]
#     y = data.iloc[:, -1]
#     X_train, X_test, y_train, y_test =train_test_split(
#         X, y, test_size = 0.2, random_state = 24)


#     # f0 = (iris.sepal_length - s['means'][0]) / np.sqrt(s['vars'][0])
#     # f1 = (iris.sepal_width - s['means'][1]) / np.sqrt(s['vars'][1])
#     # f2 = (iris.petal_length - s['means'][2]) / np.sqrt(s['vars'][2])
#     # f3 = (iris.petal_width - s['means'][3]) / np.sqrt(s['vars'][3])
#     # X_scaled = [[f0, f1, f2, f3]]

#     # Defining scoring metric for k-fold cross validation
#     # def cv_scoring(estimator, X, y):
#     #     return accuracy_score(y, estimator.predict(X))
    
#     # Initializing Models
#     models = {
#         "SVC":SVC(),
#         }

#     # Training and testing SVM Classifier
#     svm_model = SVC()
#     svm_model.fit(X_train, y_train)
#     preds = svm_model.predict(X_test)

#     # Training the models on whole data
#     final_svm_model = SVC()
#     final_svm_model.fit(X, y)

#     symptoms = X.columns.values

#     # Creating a symptom index dictionary to encode the
#     # input symptoms into numerical form
#     symptom_index = {}
#     for index, value in enumerate(symptoms):
#         symptom = " ".join([i.capitalize() for i in value.split("_")])
#         symptom_index[symptom] = index

#     data_dict = {
#         "symptom_index":symptom_index,
#         "predictions_classes":encoder.classes_
#     }

#     # Defining the Function
#     # Input: string containing symptoms separated by commmas
#     # Output: Generated predictions by models
#     def predictDisease(symptoms):
#         symptoms = symptoms.split(",")
        
#         # creating input data for the models
#         input_data = [0] * len(data_dict["symptom_index"])
#         for symptom in symptoms:
#             index = data_dict["symptom_index"][symptom]
#             input_data[index] = 1
            
#         # reshaping the input data and converting it
#         # into suitable format for model predictions
#         input_data = np.array(input_data).reshape(1,-1)
        
#         # generating output
#         svm_prediction = data_dict["predictions_classes"][final_svm_model.predict(input_data)[0]]
        
#         # making final prediction by taking mode of all predictions
#         final_prediction = mode([svm_prediction])[0][0]
#         predictions = {
#             "svm_model_prediction": svm_prediction,
#             "final_prediction":final_prediction
#         }
#         return predictions

#     # Testing the function
#     # print(predictDisease("Itching,Skin Rash,Nodal Skin Eruptions"))
        
#     y_pred = final_svm_model.predict(X_scaled)
#     # df_pred = pd.DataFrame({
#     #     'Species': ['Virginica', 'Versicolor', 'Setosa'],
#     #     'Confidence': y_pred.flatten()
#     # })
#     # df_pred['Confidence'] = [round(x,4) for x in df_pred['Confidence']]
#     # df_pred.set_index('Species', inplace=True)
#     # return df_pred.to_dict()
#     return predictDisease("Itching,Skin Rash,Nodal Skin Eruptions")#y_pred