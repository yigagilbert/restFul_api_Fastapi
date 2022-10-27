import json
import string
import numpy as np
import pandas as pd
from fastapi import FastAPI
from tensorflow import keras
from pydantic import BaseModel
from scipy.stats import mode
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib

class Symptoms(BaseModel):
    sepal_length: str("")
    sepal_width: str("")
    petal_length: str("")
    petal_width: str("")
        
# Load model
final_svm_model = joblib.load('model/svm_model.joblib')

# Load input scaling parameters
# with open('scaling.json') as f:
#     s = json.load(f)

app = FastAPI()

@app.get("/predict")
def predictDisease(symptoms : Symptoms):
    DATA_PATH = "data/Training.csv"
    data = pd.read_csv(DATA_PATH).dropna(axis = 1)
    X = data.iloc[:,:-1]
    symptoms = X.columns.values

    # value using LabelEncoder
    encoder = LabelEncoder()

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
    for symptom in symptoms:
        index = data_dict["symptom_index"][symptom]
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

# async def get_predition(iris: Iris):
#     f0 = (iris.sepal_length - s['means'][0]) / np.sqrt(s['vars'][0])
#     f1 = (iris.sepal_width - s['means'][1]) / np.sqrt(s['vars'][1])
#     f2 = (iris.petal_length - s['means'][2]) / np.sqrt(s['vars'][2])
#     f3 = (iris.petal_width - s['means'][3]) / np.sqrt(s['vars'][3])
#     X_scaled = [[f0, f1, f2, f3]]
    
#     y_pred = model.predict(X_scaled)
#     df_pred = pd.DataFrame({
#         'Species': ['Virginica', 'Versicolor', 'Setosa'],
#         'Confidence': y_pred.flatten()
#     })
#     df_pred['Confidence'] = [round(x,4) for x in df_pred['Confidence']]
#     df_pred.set_index('Species', inplace=True)
#     return df_pred.to_dict()