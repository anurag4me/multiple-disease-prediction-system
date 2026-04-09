# MedAI: Multiple Disease Prediction System

This repository hosts MedAI, a comprehensive web application built with Streamlit that leverages machine learning and deep learning to predict a variety of diseases. The system provides a user-friendly interface for risk assessment, an analytics dashboard for data visualization, and an AI-powered health chatbot for preliminary symptom analysis.

## Features

The MedAI application is modular, offering several distinct functionalities:

*   **Health Analytics Dashboard:** Presents visual insights from the various medical datasets used for training the models, highlighting patterns, risk factors, and disease prevalence.
*   **Brain Tumor Detection:** Classifies brain MRI scans into four categories: Glioma, Meningioma, Pituitary Tumor, or No Tumor.
*   **Breast Cancer Prediction:** Differentiates breast tumors as Malignant or Benign based on nucleus measurements from biopsy images.
*   **Diabetes Prediction:** Assesses the risk of diabetes based on key health metrics like glucose levels, BMI, and age.
*   **Heart Disease Prediction:** Predicts the presence of cardiovascular disease using clinical parameters.
*   **Kidney Disease Prediction:** Predicts the likelihood of Chronic Kidney Disease (CKD) from blood and urine test results.
*   **Liver Disease Prediction:** Evaluates the risk of liver disease based on liver function test (LFT) values.
*   **Parkinson's Prediction:** Detects the presence of Parkinson's disease using specific voice measurements as biomarkers.
*   **AI Health Chatbot:** An interactive chatbot that provides general health information based on user-described symptoms.

## Models & Technologies

A combination of machine learning and deep learning models is employed, each trained for a specific prediction task.

*   **Web Framework:** Streamlit
*   **Machine Learning:** Scikit-learn, Pandas, NumPy
*   **Deep Learning:** TensorFlow, Keras
*   **Data Visualization:** Plotly
*   **Image Processing:** Pillow

## Repository Structure

```
.
├── main.py                               # Main Streamlit application
├── requirements.txt                      # Python dependencies
├── saved_models/                         # Directory for all pre-trained models
│   ├── brain_tumor_model.keras
│   ├── breast_cancer_model.sav
│   ├── diabetes_model.sav
│   ├── heart_disease_model.sav
│   ├── kidney_disease_model.sav
│   ├── liver_disease_model.sav
│   └── parkinsons_model.sav
├── Brain Tumor Detection/
│   ├── Brain Tumor Detection.ipynb
│   └── MRI Images/
├── Breast Cancer Prediction/
│   └── breast_cancer_detection.ipynb
├── Diabetes Prediction/
│   ├── Diabetes Prediction.ipynb
│   └── diabetes.csv
├── Heart Disease Prediction/
│   ├── Heart Disease Prediction.ipynb
│   └── heart.csv
├── Kidney Disease Prediction/
│   ├── kidney_disease.csv
│   └── kidney_disease_prediction.ipynb
├── Liver Disease Prediction/
│   ├── indian_liver_patient.csv
│   └── liver_disease_prediction.ipynb
└── Parkinsons Disease Detection/
    ├── Parkinson's Disease Detection.ipynb
    └── parkinsons.csv
```

## How to Run

To run this project locally, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/anurag4me/multiple-disease-prediction-system.git
    cd multiple-disease-prediction-system
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Streamlit application:**
    ```bash
    streamlit run main.py
    ```
The application will automatically open in your default web browser.

## Usage

1.  Navigate to the web application in your browser.
2.  Use the sidebar to navigate between the **Dashboard**, different **Disease Prediction** modules, and the **Health Chatbot**.
3.  **For tabular data predictions** (Diabetes, Heart Disease, etc.):
    *   Enter the required medical parameters in the input fields provided.
    *   Click the "Predict" button to view the model's assessment.
4.  **For Brain Tumor Detection:**
    *   Upload an MRI scan image (.jpg, .jpeg, or .png), Additionally you can use Sample Images for testing from /Brain Tumor Detection MRI Images/Sample Testing Images/ Folder.
    *   Click the "Detect Tumour" button to process the image.
    *   The application will display the prediction (tumor type or no tumor), along with a confidence score and class probabilities.
5.  **For the AI Health Chatbot:**
    *   Type your health questions or describe your symptoms in the chat input box.
    *   The AI will provide general health information and suggest potential areas of concern.