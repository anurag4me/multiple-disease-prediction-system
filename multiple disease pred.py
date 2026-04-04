# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 23:00:42 2026

@author: anura
"""

import pickle
import streamlit as st
from streamlit_option_menu import option_menu

# loading the saved models
diabetes_model = pickle.load(open('D:/My Projects/Multiple Disease Prediction Sytem/saved_models/diabetes_model.sav', 'rb'))

heart_disease_model = pickle.load(open('D:/My Projects/Multiple Disease Prediction Sytem/saved_models/heart_disease_model.sav', 'rb'))

parkinsons_model = pickle.load(open('D:/My Projects/Multiple Disease Prediction Sytem/saved_models/parkinsons_model.sav', 'rb'))


# sidebar for navigation

with st.sidebar:
    selected = option_menu('Multiple Disease Prediction System', 
                           ['Diabetes Prediction', 
                            'Heart Disease Prediction', 
                            'Parkinsons Prediction'], 

                            icons = ['activity', 'heart', 'person'],

                            default_index=0)

# Diabetes Prediction Page
if (selected == 'Diabetes Prediction'):
    # page title
    st.title('Diabetes Prediction using ML')

    # getting the input data from the user

    #columns for input fields
    col1, col2, col3 = st.columns(3)

    with col1:
        Pregnancies = st.text_input('Number of Pregnancies')
        SkinThickness = st.text_input('Skin Thickness Value')
        DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function Value')

    with col2:
        Glucose = st.text_input('Glucose level')	
        Insulin = st.text_input('Insulin Level')
        Age = st.text_input('Age of the Person')

    with col3:
        BloodPressure = st.text_input('Blood Pressure Value')
        BMI = st.text_input('BMI Value')

    
    # code for prediction
    diab_diagnosis= ''
    
    # creating a button for prediction
    if st.button('Diabetes Test Result'):
        diab_diagnosis = diabetes_model.predict([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])

        if (diab_diagnosis[0]==1):
            diab_diagnosis = 'The person is Diabetic'
        else:
            diab_diagnosis = 'The person is Not Diabetic'
    
    st.success(diab_diagnosis)

# Diabetes Prediction Page
if (selected == 'Heart Disease Prediction'):
    # page title
    st.title('Heart Disease Prediction using ML')

# Diabetes Prediction Page
if (selected == 'Parkinsons Prediction'):
    # page title
    st.title('Parkinsons Prediction using ML')