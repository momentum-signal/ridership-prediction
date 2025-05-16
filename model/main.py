from models.train_lightgbm import train as train_lgb
from inference.predict_lightgbm import LightGBMPredictor


def run_pipeline():
    # Step 1: Train all models
    train_lgb()
    # Add other training calls...

    # Step 2: Example prediction
    predictor = LightGBMPredictor("../models/saved_models/lightgbm_model.pkl")
    print(predictor.predict(sample_data))


if __name__ == "__main__":
    run_pipeline()