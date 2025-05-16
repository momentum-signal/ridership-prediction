import torch
import pandas as pd
import numpy as np
from datetime import datetime
from model.models.train_nbeats import NBeats


class NBeatsPredictor:
    def __init__(self, model_path: str, training_data_path: str = "../data/cleaned_data.csv"):
        # Device setup
        self.device = torch.device("cpu")

        # Load model with version mismatch handling
        self.model = NBeats.load_from_checkpoint(model_path, strict=False).to(self.device)
        self.model.eval()

        # Load and preprocess training data
        self.df = pd.read_csv(training_data_path)
        self.df['datetime'] = pd.to_datetime(self.df['date'] + ' ' + self.df['time'])

        self._prepare_encoders()
        self._calculate_baselines()

    def _prepare_encoders(self):
        """Initialize all required encoders"""
        # Station encoding
        all_stations = pd.concat([self.df['origin'], self.df['destination']]).unique()
        self.station_encoder = {s: i for i, s in enumerate(sorted(all_stations))}

        # Time features analysis
        self.df['hour'] = self.df['datetime'].dt.hour
        self.hourly_avg = self.df.groupby('hour')['ridership'].mean()
        self.global_avg = self.df['ridership'].mean()

    def _calculate_baselines(self):
        """Calculate baseline statistics"""
        # Route-specific averages
        self.route_stats = self.df.groupby(['origin', 'destination'])['ridership'].agg(['mean', 'std'])

        # Time-based multipliers
        self.time_multipliers = {
            'morning_peak': (7, 9, 1.5),
            'evening_peak': (17, 19, 1.8),
            'weekend': (None, None, 1.2)
        }

    def _get_time_multiplier(self, dt: datetime) -> float:
        """Get time-based adjustment factor"""
        hour = dt.hour
        is_weekend = dt.weekday() >= 5

        for period, (start, end, mult) in self.time_multipliers.items():
            if period == 'weekend' and is_weekend:
                return mult
            elif start <= hour <= end:
                return mult
        return 1.0

    def predict_with_confidence(self, dt: datetime, origin: str, destination: str) -> tuple:
        """Make prediction with error percentage estimate"""
        try:
            # Verify stations exist
            if origin not in self.station_encoder or destination not in self.station_encoder:
                raise ValueError(f"Unknown station: {origin if origin not in self.station_encoder else destination}")

            # Get route statistics
            route_mean = self.route_stats.loc[(origin, destination), 'mean']
            route_std = self.route_stats.loc[(origin, destination), 'std']

            # Prepare features
            hour = dt.hour
            features = [
                           np.sin(2 * np.pi * hour / 24),
                           np.cos(2 * np.pi * hour / 24),
                           dt.weekday(),
                           1 if dt.weekday() >= 5 else 0,
                           self.station_encoder[origin],
                           self.station_encoder[destination]
                       ][:self.model.hparams.input_size]

            # Get model prediction
            input_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)
            with torch.no_grad():
                raw_pred = self.model(input_tensor).item()

            # Apply calibration
            time_mult = self._get_time_multiplier(dt)
            calibrated = (raw_pred + route_mean * time_mult) / 2

            # Apply reasonable bounds
            lower_bound = max(0, route_mean - 2 * route_std)
            upper_bound = route_mean + 2 * route_std
            final_pred = np.clip(calibrated, lower_bound, upper_bound)

            # Calculate error percentage (based on historical variability)
            error_pct = min(100, (route_std / route_mean * 100)) if route_mean > 0 else 0

            return int(final_pred), round(error_pct)

        except Exception as e:
            print(f"Prediction warning: {str(e)}")
            route_mean = self.route_stats.loc[(origin, destination), 'mean'] if (origin,
                                                                                 destination) in self.route_stats.index else self.global_avg
            return int(route_mean), 50  # Default 50% error for fallback


if __name__ == "__main__":
    # Initialize predictor
    predictor = NBeatsPredictor(
        model_path="../saved_models/nbeats-best.ckpt",
        training_data_path="../data/cleaned_data.csv"
    )

    # Test cases with sample expected values for error calculation
    test_cases = [
        # (datetime(2025, 1, 2, 19, 0), "KL Sentral", "Serdang", 100),
        # (datetime(2025, 1, 3, 10, 0), "KL Sentral", "Seremban", 32),
        # (datetime(2025, 1, 4, 8, 30), "Bangi", "KL Sentral", 75),
        (datetime(2025, 2, 8, 6, 0), "Pulau Sebang (Tampin)", "KL Sentral", 25),
        (datetime(2025, 1, 2, 19, 0), "KL Sentral", "Serdang",  100),
        (datetime(2025, 1, 3, 10, 0), "KL Sentral", "Seremban",  32),
        (datetime(2025, 4, 5, 15, 0), "Batu Caves", "KL Sentral",  100),
        (datetime(2025, 4, 5, 15, 0), "Batu Caves", "Kampung Batu",  28),
    ]

    print("Prediction Results with Confidence:")
    print("=" * 50)
    for dt, orig, dest, expected in test_cases:
        pred, error_pct = predictor.predict_with_confidence(dt, orig, dest)
        actual_error = abs(pred - expected) / expected * 100

        print(f"{orig}→{dest} at {dt.strftime('%a %H:%M')}:")
        print(f"  Predicted: {pred} ±{error_pct}% (Expected: {expected})")
        print(f"  Actual Error: {actual_error:.1f}%")
        print("-" * 50)



"""
("Pulau Sebang (Tampin)", "KL Sentral", datetime(2025, 2, 8, 6, 0), 25),
("KL Sentral",  "Serdang",   datetime(2025, 1, 2, 19, 0), 100),
("KL Sentral",  "Seremban",  datetime(2025, 1, 3, 10, 0), 32),
("Batu Caves", "KL Sentral", datetime(2025, 4, 5, 15, 00), 100),
("Batu Caves", "Kampung Batu", datetime(2025, 4, 5, 15, 00), 28),
"""