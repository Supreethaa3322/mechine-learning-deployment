import numpy as np
from flask import Flask, request, jsonify

# Instantiate the Flask application (to ensure fresh routes in notebook environment)
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    if not request.json:
        return jsonify({'error': 'No input data provided'}), 400

    # The model expects 8 features in a specific order:
    # Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age
    required_features = [
        'Pregnancies',
        'Glucose',
        'BloodPressure',
        'SkinThickness',
        'Insulin',
        'BMI',
        'DiabetesPedigreeFunction',
        'Age'
    ]

    input_data = request.json
    feature_values = []
    for feature in required_features:
        value = input_data.get(feature)
        if value is None:
            return jsonify({'error': f'Missing feature: {feature}'}), 400
        feature_values.append(value)

    try:
        # Convert input to a numpy array and reshape for the model
        features_array = np.array(feature_values).reshape(1, -1)

        # Scale the input features using the loaded scaler
        scaled_features = loaded_scaler.transform(features_array)

        # Make prediction
        prediction = loaded_model.predict(scaled_features)
        prediction_proba = loaded_model.predict_proba(scaled_features)

        # Return prediction as JSON
        return jsonify({
            'prediction': int(prediction[0]),
            'probability_no_diabetes': prediction_proba[0][0],
            'probability_diabetes': prediction_proba[0][1]
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

print("'/predict' endpoint defined. It now uses the loaded scaler and model.")

import numpy as np
from flask import Flask, request, jsonify

# Instantiate the Flask application (to ensure fresh routes in notebook environment)
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    if not request.json:
        return jsonify({'error': 'No input data provided'}), 400

    # The model expects 8 features in a specific order:
    # Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age
    required_features = [
        'Pregnancies',
        'Glucose',
        'BloodPressure',
        'SkinThickness',
        'Insulin',
        'BMI',
        'DiabetesPedigreeFunction',
        'Age'
    ]

    input_data = request.json
    feature_values = []
    for feature in required_features:
        value = input_data.get(feature)
        if value is None:
            return jsonify({'error': f'Missing feature: {feature}'}), 400
        feature_values.append(value)

    try:
        # Convert input to a numpy array and reshape for the model
        features_array = np.array(feature_values).reshape(1, -1)

        # Scale the input features using the loaded scaler
        scaled_features = loaded_scaler.transform(features_array)

        # Make prediction
        prediction = loaded_model.predict(scaled_features)
        prediction_proba = loaded_model.predict_proba(scaled_features)

        # Return prediction as JSON
        return jsonify({
            'prediction': int(prediction[0]),
            'probability_no_diabetes': prediction_proba[0][0],
            'probability_diabetes': prediction_proba[0][1]
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

print("'/predict' endpoint defined. It now uses the loaded scaler and model.")


if __name__ == "__main__":
    print("Starting prediction API with preprocessing and model inference...")
    app.run(debug=True)
