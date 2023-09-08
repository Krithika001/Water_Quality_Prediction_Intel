# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 14:48:53 2023

@author: 19pd19
"""

from flask import Flask, render_template, request
# import xgboost

app = Flask(__name__, template_folder="g:/Sem 9/Water_Quality_Prediction/src/template")

import numpy as np
import pickle

file_path = 'g:/Sem 9/Water_Quality_Prediction/models/Logistic_Regression.pkl'

with open(file_path, 'rb') as file:
    loaded_model = pickle.load(file)

print("Model Loaded")

# Dictionary of water quality feature effects
feature_effects = {
    'pH': {
        'positive': "Optimal pH levels (usually around 7) indicate neutral water, which is generally safe for consumption.",
        'negative': "Extreme pH levels (too acidic or too alkaline) can indicate water contamination and may be harmful to health."
    },
    'Iron': {
        'positive': "A low level of iron is generally safe and doesn't affect water quality.",
        'negative': "High levels of iron can cause water to taste and smell bad and may stain clothes and fixtures."
    },
    'Nitrate': {
        'positive': "Low levels of nitrates are generally safe for consumption.",
        'negative': "High levels of nitrates can indicate contamination from agricultural runoff and pose health risks, especially for infants."
    },
    'Chloride': {
        'positive': "Chloride ions at safe levels are used for water disinfection.",
        'negative': "Excessive chloride levels can make water taste salty and may corrode pipes and equipment."
    },
    'Lead': {
        'positive': "Low levels of lead are safe, and it is important to have lead-free drinking water.",
        'negative': "High levels of lead can cause severe health issues, especially in children, and must be avoided."
    },
    'Zinc': {
        'positive': "Zinc at safe levels is an essential nutrient for humans.",
        'negative': "Excessive zinc levels can cause gastrointestinal issues and affect water taste."
    },
    'Color': {
        'positive': "Natural coloration in water is generally harmless.",
        'negative': "Unnatural coloration may indicate contamination and should be investigated."
    },
    'Turbidity': {
        'positive': "Low turbidity indicates clear water, which is generally preferred for aesthetics.",
        'negative': "High turbidity can make water appear cloudy and may contain suspended particles."
    },
    'Fluoride': {
        'positive': "Fluoride at safe levels can help prevent tooth decay.",
        'negative': "Excessive fluoride can lead to dental fluorosis and other health issues."
    },
    'Copper': {
        'positive': "Copper at safe levels is essential for health.",
        'negative': "High copper levels can cause gastrointestinal issues and affect water taste."
    },
    'Odor': {
        'positive': "No noticeable odor is generally preferred for drinking water.",
        'negative': "Foul odors may indicate contamination or water quality issues."
    },
    'Sulfate': {
        'positive': "Sulfate at safe levels is generally harmless.",
        'negative': "High sulfate levels can have a laxative effect and affect water taste."
    },
    'Conductivity': {
        'positive': "Conductivity can help monitor water quality, and stable levels are desired.",
        'negative': "Extreme conductivity fluctuations can indicate contamination."
    },
    'Chlorine': {
        'positive': "Chlorine is commonly used for water disinfection at safe levels.",
        'negative': "Excessive chlorine levels can lead to a strong taste and odor."
    },
    'Manganese': {
        'positive': "Low levels of manganese are generally safe and may even be beneficial.",
        'negative': "High manganese levels can affect water taste and cause staining."
    },
    'Total Dissolved Solids': {
        'positive': "A moderate level of TDS is common in natural water and generally safe.",
        'negative': "Excessive TDS may affect water taste and quality."
    },
    'Source': {
        'positive': "A clean and reliable water source is essential for safe drinking water.",
        'negative': "Contaminated water sources can pose serious health risks."
    },
    'Water Temperature': {
        'positive': "Moderate water temperatures are generally comfortable for various uses.",
        'negative': "Extreme temperatures can affect water quality and aquatic ecosystems."
    },
    'Air Temperature': {
        'positive': "Air temperature may influence water temperature but does not directly affect water quality.",
        'negative': "Extreme air temperatures, such as extreme cold, can lead to the freezing of water sources, which may disrupt water supply systems."
    }
}


# Function to explain the effects of a feature
def explain_feature_effect(feature_name):
    if feature_name in feature_effects:
        positive_effect = feature_effects[feature_name]['positive']
        negative_effect = feature_effects[feature_name]['negative']
        return f"Effects of {feature_name} in water:\n\nPositive Effect: {positive_effect}\n\nNegative Effect: {negative_effect}"
    else:
        return "Feature not found in the dictionary."

# Function to input features for prediction
def input_features_for_prediction(user_inputs):
    max_na_count = 6  # Maximum allowed "na" inputs
    na_count = 0

    for feature in feature_names:
        while True:
            user_input = user_inputs.get(feature, '')

            if user_input == 'NaN' or user_input.lower() == 'na':
                na_count += 1
                if na_count > max_na_count:
                    return "Too many 'na' inputs. Exiting."
                user_inputs[feature] = -1
                break
            else:
                try:
                    value = float(user_input)
                    user_inputs[feature] = value
                    break
                except ValueError:
                    return "Invalid input. Please enter a valid number or 'na'."

    # Fill missing values using the imputer
    custom_input = np.array(list(user_inputs.values())).reshape(1, -1)
    # Make predictions
    custom_pred = loaded_model.predict(custom_input)
    if custom_pred[0] == 0:
        return "Hurray!! The water is usable"
    else:
        return "Oops!! The water is not usable"

# Home route
@app.route('/')
def home():
    feature_names = list(feature_effects.keys())
    return render_template('index.html', feature_names = feature_names)

# Effects route
@app.route('/effects', methods = ['POST'])
def effects():
    selected_feature = request.form.get('feature')
    effect_text = explain_feature_effect(selected_feature)
    return render_template('index.html', feature_names = feature_names,effect_text=effect_text)

# Prediction route
@app.route('/prediction', methods = ['POST'])
def prediction():
    user_inputs = {}
    for feature in feature_names:
        user_inputs[feature] = request.form.get(feature)

    result_text = input_features_for_prediction(user_inputs)
    return render_template('index.html', feature_names = feature_names, result_text=result_text)

if __name__ == '__main__':
    feature_names = list(feature_effects.keys())
    app.run(debug=True)

