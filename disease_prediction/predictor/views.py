from django.shortcuts import render
import numpy as np # type: ignore
import pandas as pd # type: ignore
import pickle
from sklearn.datasets import load_breast_cancer # type: ignore
from django.http import JsonResponse
import requests
from django.shortcuts import render
from django.conf import settings
import google.generativeai as genai # type: ignore
from django.shortcuts import render
from django.conf import settings


# Load models and scalers when Django starts
MODELS = {
    "diabetes": pickle.load(open("models/diabetes.pkl", "rb")),
    "breast_cancer": pickle.load(open("models/breast_cancer.pkl", "rb")),
}

SCALERS = {
    "diabetes": pickle.load(open("models/scaler.pkl", "rb")),
}

# Feature names for different models
FEATURES = {
    "diabetes": ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
                 "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"],
    "breast_cancer": [f"feature_{i}" for i in range(30)],
}


# ------------- Home Page ------------- #
def home(request):
    """Render the home page."""
    return render(request, "index.html")


# ------------- Diabetes Prediction ------------- #
def predict_diabetes(request):
    """Handle form submission and make predictions for diabetes."""
    if request.method == "POST":
        try:
            # Extract input values
            input_data = [float(request.POST.get(field, 0)) for field in FEATURES["diabetes"]]

            # Convert to DataFrame & scale
            input_df = pd.DataFrame([input_data], columns=FEATURES["diabetes"])
            input_scaled = SCALERS["diabetes"].transform(input_df)

            # Make prediction
            prediction = MODELS["diabetes"].predict(input_scaled)[0]
            result = "Positive" if prediction == 1 else "Negative"

            return render(request, "predict_diabetes.html", {
                "form_data": request.POST,
                "prediction": result,
            })

        except Exception as e:
            return render(request, "predict_diabetes.html", {"error": f"Error: {e}"})

    return render(request, "diabetes_prediction.html")


# ------------- Breast Cancer Prediction ------------- #
def breast_cancer_predict(request):
    # Load feature names
    data = load_breast_cancer()
    feature_names = data.feature_names  # List of feature names
    """Handle form submission and make predictions for breast cancer."""
    if request.method == "POST":
        try:
            # Extract input values
            input_data = [float(request.POST.get(f"feature_{i}", 0)) for i in range(30)]
            input_array = np.array(input_data).reshape(1, -1)

            # Make prediction
            prediction = MODELS["breast_cancer"].predict(input_array)[0]
            result = "Malignant (Cancerous)" if prediction == 0 else "Benign (Non-Cancerous)"

            return render(request, "breast_cancer_result.html", {
                "features": input_data,
                "prediction": result,
            })

        except Exception as e:
            return render(request, "breast_cancer_result.html", {"error": f"Error: {e}"})

    return render(request, "breast_cancer_form.html", {"feature_names": feature_names})

# ------------- Heart Disease Prediction ------------- #
def heart_disease_predict(request):
    if request.method == "POST":
        try:
            # Load the scaler and model
            with open("models/heart_disease_scaler.pkl", "rb") as f:
                scaler = pickle.load(f)

            with open("models/heart_disease.pkl", "rb") as f:
                model = pickle.load(f)

            print("Scaler expects:", scaler.n_features_in_)  # Debugging

            # Extract and convert input values
            age = int(request.POST["age"])
            sex = int(request.POST["sex"])
            cp = int(request.POST["cp"])
            trestbps = int(request.POST["trestbps"])
            chol = int(request.POST["chol"])
            fbs = int(request.POST["fbs"])
            restecg = int(request.POST["restecg"])
            thalach = int(request.POST["thalach"])
            exang = int(request.POST["exang"])
            oldpeak = float(request.POST["oldpeak"])
            slope = int(request.POST["slope"])
            ca = int(request.POST["ca"])
            thal = int(request.POST["thal"])

            # Create feature array
            features = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]

            # Add missing features (if needed)
            missing_features = [0] * (20 - len(features))
            features += missing_features

            # Scale input
            input_scaled = scaler.transform([features])

            # Make prediction
            prediction = model.predict(input_scaled)[0]
            result = "Positive (Heart Disease)" if prediction == 1 else "Negative (No Heart Disease)"

            return render(request, "heart_disease_result.html", {"prediction": result})

        except Exception as e:
            return render(request, "heart_disease_result.html", {"error": f"Error: {e}"})

    return render(request, "heart_disease_form.html")

# ------------- Kidney Disease Prediction ------------- #

def kidney_disease_view(request):
    return render(request, 'kidney_disease.html')  # Make sure this template exists


def kidney_disease_predict(request):
    # Load the model
    with open("models/kidney_disease_model.pkl", "rb") as f:
        model = pickle.load(f)

    # Mapping for categorical variables
    red_blood_cells_map = {"normal": 1, "abnormal": 0}
    pus_cell_map = {"normal": 1, "abnormal": 0}
    pus_cell_clumps_map = {"present": 1, "not present": 0}
    bacteria_map = {"present": 1, "not present": 0}
    appetite_map = {"good": 1, "poor": 0}
    peda_edema_map = {"yes": 1, "no": 0}
    aanemia_map = {"yes": 1, "no": 0}
    hypertension_map = {"yes": 1, "no": 0}
    diabetes_mellitus_map = {"yes": 1, "no": 0}
    coronary_artery_disease_map = {"yes": 1, "no": 0}
    if request.method == "POST":
        try:
            # Extract and process input data
            age = float(request.POST["age"])
            blood_pressure = float(request.POST["blood_pressure"])
            specific_gravity = float(request.POST["specific_gravity"])
            albumin = float(request.POST["albumin"])
            sugar = float(request.POST["sugar"])
            red_blood_cells = red_blood_cells_map[request.POST["red_blood_cells"]]
            pus_cell = pus_cell_map[request.POST["pus_cell"]]
            pus_cell_clumps = pus_cell_clumps_map[request.POST["pus_cell_clumps"]]
            bacteria = bacteria_map[request.POST["bacteria"]]
            blood_glucose_random = float(request.POST["blood_glucose_random"])
            packed_cell_volume = float(request.POST["packed_cell_volume"])
            white_blood_cell_count = float(request.POST["white_blood_cell_count"])
            red_blood_cell_count = float(request.POST["red_blood_cell_count"])
            hypertension = hypertension_map[request.POST["hypertension"]]
            diabetes_mellitus = diabetes_mellitus_map[request.POST["diabetes_mellitus"]]
            coronary_artery_disease = coronary_artery_disease_map[request.POST["coronary_artery_disease"]]
            appetite = appetite_map[request.POST["appetite"]]
            peda_edema = peda_edema_map[request.POST["peda_edema"]]
            aanemia = aanemia_map[request.POST["aanemia"]]
            packed_cell_volume = request.POST.get('packed_cell_volume', None)
            blood_urea = float(request.POST["blood_urea"])
            blood_urea = float(request.POST["blood_urea"])
            serum_creatinine = float(request.POST["serum_creatinine"])
            sodium = float(request.POST["sodium"])
            potassium = float(request.POST["potassium"])
            haemoglobin = float(request.POST["haemoglobin"])


            # Make sure all 24 features are included
            features = np.array([[age, blood_pressure, specific_gravity, albumin, sugar, 
                      red_blood_cells, pus_cell, pus_cell_clumps, bacteria, 
                      blood_glucose_random, blood_urea, serum_creatinine, sodium, 
                      potassium, haemoglobin, packed_cell_volume, white_blood_cell_count, 
                      red_blood_cell_count, hypertension, diabetes_mellitus, 
                      coronary_artery_disease, appetite, peda_edema, aanemia]])


            # Debugging: Print the number of features expected vs. provided
            print("Model expects:", model.n_features_in_)  
            print("Input features count:", len(features[0]))  


            # Make prediction
            prediction = model.predict(features)[0]
            result = "Kidney Disease Detected" if prediction == 1 else "No Kidney Disease"

            return render(request, "kidney_disease_result.html", {"prediction": result})

        except Exception as e:
            return render(request, "kidney_disease_result.html", {"error": f"Error: {e}"})

    return render(request, "kidney_disease_form.html")




genai.configure(api_key=settings.GEMINI_API_KEY)

def gemini_chat(request):
    response_text = ""
    if request.method == "POST":
        prompt = request.POST.get("prompt")

        model = genai.GenerativeModel("gemini-1.5-flash")
        chat_session = model.start_chat(history=[])
        response = chat_session.send_message(prompt)

        response_text = response.text if response else "No response received."

    return render(request, "gemini_prompt.html", {"response": response_text})


