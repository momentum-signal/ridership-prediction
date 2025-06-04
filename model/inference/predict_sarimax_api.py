# API for making predictions using a trained SARIMAX model
# model/inference/predict_sarimax_api.py

from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import pandas as pd
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the trained SARIMAX model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "../models/sarimax_model_weekly.pkl")
with open(MODEL_PATH, "rb") as model_file:
    model = pickle.load(model_file)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Parse input JSON
        data = request.json
        if not data:
            return jsonify({"error": "Request body is empty or not valid JSON."}), 400        # Validate required keys
        required_keys = ["day_of_week", "is_weekend", "is_holiday", "month"]
        missing_keys = [key for key in required_keys if key not in data]
        if missing_keys:
            return jsonify({"error": f"Missing required keys: {', '.join(missing_keys)}"}), 400
        
        # Extract features as a list in the correct order
        exog_values = [
            data["day_of_week"],
            data["is_weekend"], 
            data["is_holiday"],
            data["month"]
        ]
        
        # Convert to DataFrame with proper column names
        features = pd.DataFrame([exog_values], columns=['day_of_week', 'is_weekend', 'is_holiday', 'month'])
        
        # Make prediction
        try:
            prediction = model.forecast(steps=1, exog=features)
            print(f"Prediction successful: {prediction}")
            
            # Extract the prediction value properly
            prediction_value = prediction.iloc[0] if hasattr(prediction, 'iloc') else prediction[0]
            
            # Return prediction (convert to float)
            return jsonify({"prediction": float(prediction_value)})
        except Exception as forecast_error:
            print(f"Forecast error: {forecast_error}")
            return jsonify({"error": f"Forecast failed: {str(forecast_error)}"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/stations", methods=["GET"])
def get_stations():
    try:
        # Define the path to the dataset
        dataset_path = os.path.join(os.path.dirname(__file__), "../data/cleaned_data.csv")

        # Load the dataset
        data = pd.read_csv(dataset_path)

        # Extract unique station names from both origin and destination columns
        origin_stations = data["origin"].dropna().unique()
        destination_stations = data["destination"].dropna().unique()
        
        # Combine and get unique stations
        unique_stations = sorted(list(set(list(origin_stations) + list(destination_stations))))

        # Return the station names as a JSON response
        return jsonify({"stations": unique_stations})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
