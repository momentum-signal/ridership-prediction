import joblib
import pandas as pd
import numpy as np


def add_features_for_inference(df):
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['hour_sin'] = np.sin(2 * np.pi * df['datetime'].dt.hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['datetime'].dt.hour / 24)
    df['lag_1_week'] = 0  # no historical data at inference
    return df


class LightGBMPredictor:
    def __init__(self, model_bundle_path):
        bundle = joblib.load(model_bundle_path)
        self.model = bundle['model']
        self.origin_encoder = bundle['origin_encoder']
        self.destination_encoder = bundle['destination_encoder']

    def predict(self, new_data):
        new_data = add_features_for_inference(new_data)
        new_data['origin_encoded'] = self.origin_encoder.transform(new_data['origin'])
        new_data['destination_encoded'] = self.destination_encoder.transform(new_data['destination'])

        features = ['hour_sin', 'hour_cos', 'day_of_week', 'is_weekend', 'is_holiday', 'lag_1_week', 'origin_encoded', 'destination_encoded']
        return self.model.predict(new_data[features])


# Usage example
if __name__ == "__main__":
    predictor = LightGBMPredictor("../models/saved_models/lightgbm_model.pkl")

    sample_input = pd.DataFrame([{
        'datetime': '2025-04-01 19:00',
        'origin': 'KL Sentral',
        'destination': 'Tanjong Malim',
        'day_of_week': 2,
        'is_weekend': 0,
        'is_holiday': 1
    }])
    # 2025 - 01 - 01, 11: 00, Batu Caves, Kajang, 5, 3, 0, 1
    # 2025 - 04 - 01, 15: 00, Seremban, KL Sentral, 100, 2, 0, 1
    # 2025 - 04 - 01, 19: 00, KL Sentral, Tanjong Malim, 21, 2, 0, 1
    prediction = predictor.predict(sample_input)
    sample_input['predicted_ridership'] = prediction
    print(sample_input[['datetime', 'origin', 'destination', 'predicted_ridership']])
