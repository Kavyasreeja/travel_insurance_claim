import pickle
import numpy as np

# load the model from disk
loaded_model = pickle.load(open('streamlit_insurance_predictcharges.pkl', 'rb'))

import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import  LabelEncoder
from xgboost import XGBClassifier
import numpy as np
st.header("Claim Prediction app")
st.text_input("Enter your Name: ", key="name")
data = pd.read_csv("travel insurance.csv")
#load label encoder
encoder = LabelEncoder()
encoder.classes_ = np.load('classes.npy',allow_pickle=True)

# load model
#best_xgboost_model = xgb.XGBRegressor()
#best_xgboost_model.load_model("best_model.json")

if st.checkbox('Show Training Dataframe'):
    data
    
st.subheader("Please select relevant features of your Agency")
left_column, right_column = st.columns(2)
with left_column:
    inp_Agency = st.radio(
        'Name of the Agency',
        np.unique(data['Agency']))   
    
st.subheader("Please select relevant features of your Agency Type")
left_column, right_column = st.columns(2)
with left_column:
    inp_Agency_Type = st.radio(
        'Name of the Agency Type',
        np.unique(data['Agency_Type']))   
    
st.subheader("Please select relevant features of your Dist Channel")
left_column, right_column = st.columns(2)
with left_column:
    inp_Dist_Channel = st.radio(
        'Name of the Dist Channel',
        np.unique(data['Dist_Channel']))   
    
    
st.subheader("Please select relevant features of your Product")
left_column, right_column = st.columns(2)
with left_column:
    inp_Prod_Name = st.radio(
        'Name of the Product',
        np.unique(data['Prod_Name']))



input_Duration = st.slider('Enter Duration', 0, max(data["Duration"]), 200)

st.subheader("Please select relevant features of your Destination")
left_column, right_column = st.columns(2)
with left_column:
    inp_Destination = st.radio(
        'Name of the Destination',
        np.unique(data['Destination']))

input_Net_Sales = st.slider('Enter Net Sales', 0.0, max(data["Net_Sales"]), 100.0)
input_Commission = st.slider('Enter Commission Value', 0.0, max(data["Commission"]), 100.0)
input_Age = st.slider('Enter Age', 0, max(data["Age"]), 100)

x = data.drop(columns='Claim')
y = data['Claim']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=1)



xgc = XGBClassifier()

xgc.fit(x_train,y_train)

#xg_pred = xgc.predict(x_test)

if st.button('Make Prediction'):

    inputs = np.expand_dims(
        [input_Agency,input_Agency_Type,input_Dist_Channel, input_Prod_Name, input_Duration, input_Destination, input_Net_Sales, input_Commission, input_Age], 0)
    prediction = xgc.predict(inputs)
    if prediction==Yes:
        st.write("Your insurance will be claimed")
    else: 
        st.write("Your insurance will not be claimed")
