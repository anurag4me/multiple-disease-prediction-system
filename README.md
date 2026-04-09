# Multiple Disease Prediction System

This repository contains a Streamlit web application that leverages machine learning and deep learning to predict various diseases. The system provides a user-friendly interface for predicting Brain Tumors, Diabetes, Heart Disease, and Parkinson's Disease.

## Features

The application is organized into four distinct prediction modules:

*   **Brain Tumor Detection:** Classifies brain MRI scans into four categories: Glioma, Meningioma, Pituitary Tumor, or No Tumor.
*   **Diabetes Prediction:** Predicts whether a person has diabetes based on various health metrics.
*   **Heart Disease Prediction:** Predicts the presence of heart disease based on medical attributes.
*   **Parkinson's Prediction:** Predicts the presence of Parkinson's disease based on specific voice measurements.

## Models Used

Different models are trained for each specific prediction task.

1.  **Brain Tumor Detection (Deep Learning)**
    *   **Model:** A Convolutional Neural Network (CNN) built using transfer learning with the VGG16 architecture.
    *   **Framework:** TensorFlow/Keras.
    *   **Details:** The model is trained on a dataset of brain MRI images. It processes an uploaded MRI scan and predicts the tumor type with an associated confidence score. The final model is saved as `brain_tumor_model.keras`.

2.  **Diabetes, Heart Disease, and Parkinson's Prediction (Machine Learning)**
    *   **Models:**
        *   **Diabetes:** Support Vector Machine (SVM)
        *   **Heart Disease:** Logistic Regression
        *   **Parkinson's:** Support Vector Machine (SVM)
    *   **Framework:** Scikit-learn.
    *   **Details:** These models are trained on tabular datasets containing relevant medical features for each disease. The trained models are saved as `.sav` files using `pickle`.

## Technologies Used

*   **Web Framework:** Streamlit
*   **Deep Learning:** TensorFlow, Keras
*   **Machine Learning:** Scikit-learn
*   **Data Manipulation:** Pandas, NumPy
*   **Image Processing:** Pillow

## Repository Structure

```
├── main.py                          # Main Streamlit application file
├── requirements.txt                 # Python dependencies
├── saved_models/                    # Directory for pre-trained models
│   ├── brain_tumor_model.keras
│   ├── diabetes_model.sav
│   ├── heart_disease_model.sav
│   └── parkinsons_model.sav
├── Brain Tumor Detection/             # Notebook and data for brain tumor model
│   ├── Brain Tumor Detection.ipynb
│   └── MRI Images/
├── Diabetes Prediction/               # Notebook and data for diabetes model
│   ├── Diabetes Prediction.ipynb
│   └── diabetes.csv
├── Heart Disease Prediction/          # Notebook and data for heart disease model
│   ├── Heart Disease Prediction.ipynb
│   └── heart.csv
└── Parkinsons Disease Detection/      # Notebook and data for Parkinson's model
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

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Streamlit application:**
    ```bash
    streamlit run main.py
    ```

    The application will open in your default web browser.

## Usage

1.  Navigate to the web application in your browser.
2.  Use the sidebar to select the disease you want to predict.
3.  **For Diabetes, Heart Disease, and Parkinson's:**
    *   Enter the required medical parameters into the input fields.
    *   Click the corresponding "Test Result" button to see the prediction.
4.  **For Brain Tumor Detection:**
    *   Upload an MRI scan image (`.jpg`, `.jpeg`, or `.png`), Additionally you can use Sample Images for testing from `/Brain Tumor Detection/MRI Images/Sample Testing Images/` Folder.
    *   Click the "Run Brain Tumor Detection" button.
    *   The application will display the uploaded image, the prediction result, a confidence score, and a breakdown of class probabilities.
