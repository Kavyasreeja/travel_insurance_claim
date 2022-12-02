import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import  LabelEncoder
from xgboost import XGBClassifier
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
#from imblearn.over_sampling import ADASYN
from sklearn.metrics import *
from xgboost import XGBClassifier
#Main header file of the project
st.header("Claim Prediction app")
#Read name of the user
st.text_input("Enter your Name: ", key="name")
#Load the final pre-processed dataset on which the models will be trained
#date pre-processing and cleaning is done in phase-1 with original data
data = pd.read_csv("dic.csv")

#load label encoder
encoder = LabelEncoder()
encoder.classes_ = np.load('classes.npy',allow_pickle=True)

# load model
#best_xgboost_model = xgb.XGBRegressor()
#best_xgboost_model.load_model("best_model.json")
# Preview of the datset
if st.checkbox('Show Training Dataframe'):
    data
   
#Read the appliaction details that has been given to the GUI by any random user
input_Agency=st.subheader("Please select relevant features of your Agency")
left_column, right_column = st.columns(2)
with left_column:
    inp_Agency = st.radio(
        'Name of the Agency',
        np.unique(data['Agency']))   
    
input_Agency_Type=st.subheader("Please select relevant features of your Agency Type")
left_column, right_column = st.columns(2)
with left_column:
    inp_Agency_Type = st.radio(
        'Name of the Agency Type',
        np.unique(data['Agency_Type']))   
    
input_Dist_Channel=st.subheader("Please select relevant features of your Dist Channel")
left_column, right_column = st.columns(2)
with left_column:
    inp_Dist_Channel = st.radio(
        'Name of the Dist Channel',
        np.unique(data['Dist_Channel']))   
    
    
input_Prod_Name=st.subheader("Please select relevant features of your Product")
left_column, right_column = st.columns(2)
with left_column:
    inp_Prod_Name = st.radio(
        'Name of the Product',
        np.unique(data['Prod_Name']))



input_Duration = st.slider('Enter Duration', 0, max(data["Duration"]), 200)

input_Destination=st.subheader("Please select relevant features of your Destination")
left_column, right_column = st.columns(2)
with left_column:
    inp_Destination = st.radio(
        'Name of the Destination',
        np.unique(data['Destination']))

input_Net_Sales = st.slider('Enter Net Sales', 0.0, max(data["Net_Sales"]), 100.0)
input_Commission = st.slider('Enter Commission Value', 0.0, max(data["Commission"]), 100.0)
input_Age = st.slider('Enter Age', 0, max(data["Age"]), 100)

#features = [input_Agency,input_Agency_Type,input_Dist_Channel, input_Prod_Name, input_Duration, input_Destination, input_Net_Sales, input_Commission, input_Age]

#int_features = [int(x) for x in features]
#final_features = [np.array(int_features)]
#x = data.drop(columns='Claim')
#y = data['Claim']
#x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=1)



#xgc = XGBClassifier()

#xgc.fit(x_train,y_train)

#xg_pred = xgc.predict(x_test)
# predict wether the applicant will default or not not if the credit card is issued
if st.button('Make Prediction'):

    inputs = np.expans_dims(
        [input_Agency,input_Agency_Type,input_Dist_Channel, input_Prod_Name, input_Duration, input_Destination, input_Net_Sales, input_Commission, input_Age], 0)
    #Training the best model(XGBoost)
    X = data.drop(['Claim'], axis=1)
    y = data['Claim']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 123)
        #adasyn = ADASYN()
    #X_train,y_train = adasyn.fit_resample(X_train,y_train)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    best_xgboost_model = XGBClassifier(max_depth=5,n_estimators=250, min_child_weight=8)
    best_xgboost_model.fit(X_train, y_train)
    #Make prediction and print output
    prediction = best_xgboost_model.predict(inputs)
  
    if prediction==Yes:
        st.write("Your insurance will be claimed")
    else: 
        st.write("Your insurance will not be claimed")
