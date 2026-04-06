import os
import pickle
import streamlit as st
from streamlit_option_menu import option_menu


# Set page configuration
st.set_page_config(page_title="Health Assistant",
                   layout="wide",
                   page_icon="🧑‍⚕️")

# getting the working directory of the main.py
working_dir = os.path.dirname(os.path.abspath(__file__))

# ── Load classic ML models ──────────────────────────────────────────────────
diabetes_model = pickle.load(open(f'{working_dir}/saved_models/diabetes_model.sav', 'rb'))
heart_disease_model = pickle.load(open(f'{working_dir}/saved_models/heart_disease_model.sav', 'rb'))
parkinsons_model = pickle.load(open(f'{working_dir}/saved_models/parkinsons_model.sav', 'rb'))

# ── Sidebar navigation ───────────────────────────────────────────────────────
with st.sidebar:
    selected = option_menu(
        'Multiple Disease Prediction System',
        ['Diabetes Prediction',
         'Heart Disease Prediction',
         "Parkinson's Prediction"],
        menu_icon='hospital-fill',
        icons=['activity', 'heart', 'person'],
        default_index=0
    )

# ── Diabetes Prediction ──────────────────────────────────────────────────────
if selected == 'Diabetes Prediction':
    st.title('Diabetes Prediction using ML')

    col1, col2, col3 = st.columns(3)

    with col1:
        Pregnancies = st.text_input('Number of Pregnancies')
        Glucose     = st.text_input('Glucose level')
        BloodPressure = st.text_input('Blood Pressure Value')

    with col2:
        SkinThickness = st.text_input('Skin Thickness Value')
        Insulin = st.text_input('Insulin Level')
        BMI     = st.text_input('BMI Value')

    with col3:
        DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function Value')
        Age = st.text_input('Age of the Person')

    diab_diagnosis = ''
    if st.button('Diabetes Test Result'):
        diab_diagnosis = diabetes_model.predict(
            [[Pregnancies, Glucose, BloodPressure, SkinThickness,
              Insulin, BMI, DiabetesPedigreeFunction, Age]]
        )
        diab_diagnosis = 'The person is Diabetic' if diab_diagnosis[0] == 1 \
                         else 'The person is Not Diabetic'

    st.success(diab_diagnosis)

# ── Heart Disease Prediction ─────────────────────────────────────────────────
if selected == 'Heart Disease Prediction':
    st.title('Heart Disease Prediction using ML')

    col1, col2, col3 = st.columns(3)

    with col1:
        age      = st.text_input('Age')
        sex      = st.text_input('Sex')
        cp       = st.text_input('Chest Pain types')
        trestbps = st.text_input('Resting Blood Pressure')
        chol     = st.text_input('Serum Cholestoral in mg/dl')

    with col2:
        fbs     = st.text_input('Fasting Blood Sugar > 120 mg/dl')
        restecg = st.text_input('Resting Electrocardiographic results')
        thalach = st.text_input('Maximum Heart Rate achieved')
        exang   = st.text_input('Exercise Induced Angina')

    with col3:
        oldpeak = st.text_input('ST depression induced by exercise')
        slope   = st.text_input('Slope of the peak exercise ST segment')
        ca      = st.text_input('Major vessels colored by flourosopy')
        thal    = st.text_input('thal: 0 = normal; 1 = fixed defect; 2 = reversable defect')

    heart_diagnosis = ''
    if st.button('Heart Disease Test Result'):
        user_input = [age, sex, cp, trestbps, chol, fbs,
                      restecg, thalach, exang, oldpeak, slope, ca, thal]
        user_input       = [float(x) for x in user_input]
        heart_prediction = heart_disease_model.predict([user_input])
        heart_diagnosis  = 'The person is having heart disease' if heart_prediction[0] == 1 \
                           else 'The person does not have any heart disease'

    st.success(heart_diagnosis)

# ── Parkinson's Prediction ───────────────────────────────────────────────────
if selected == "Parkinson's Prediction":
    st.title('Parkinsons Prediction using ML')

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        fo             = st.text_input('MDVP:Fo(Hz)')
        fhi            = st.text_input('MDVP:Fhi(Hz)')
        flo            = st.text_input('MDVP:Flo(Hz)')
        Jitter_percent = st.text_input('MDVP:Jitter(%)')
        Jitter_Abs     = st.text_input('MDVP:Jitter(Abs)')

    with col2:
        RAP        = st.text_input('MDVP:RAP')
        PPQ        = st.text_input('MDVP:PPQ')
        DDP        = st.text_input('Jitter:DDP')
        Shimmer    = st.text_input('MDVP:Shimmer')
        Shimmer_dB = st.text_input('MDVP:Shimmer(dB)')

    with col3:
        APQ3 = st.text_input('Shimmer:APQ3')
        APQ5 = st.text_input('Shimmer:APQ5')
        APQ  = st.text_input('MDVP:APQ')
        DDA  = st.text_input('Shimmer:DDA')

    with col4:
        NHR  = st.text_input('NHR')
        HNR  = st.text_input('HNR')
        RPDE = st.text_input('RPDE')
        DFA  = st.text_input('DFA')

    with col5:
        spread1 = st.text_input('spread1')
        spread2 = st.text_input('spread2')
        D2  = st.text_input('D2')
        PPE = st.text_input('PPE')

    parkinsons_diagnosis = ''
    if st.button("Parkinson's Test Result"):
        user_input = [fo, fhi, flo, Jitter_percent, Jitter_Abs,
                      RAP, PPQ, DDP, Shimmer, Shimmer_dB, APQ3, APQ5,
                      APQ, DDA, NHR, HNR, RPDE, DFA, spread1, spread2, D2, PPE]
        user_input             = [float(x) for x in user_input]
        parkinsons_prediction  = parkinsons_model.predict([user_input])
        parkinsons_diagnosis   = "The person has Parkinson's disease" if parkinsons_prediction[0] == 1 \
                                 else "The person does not have Parkinson's disease"

    st.success(parkinsons_diagnosis)