from lightgbm import LGBMRegressor
import joblib
from model.utils.data_loader import load_data
from model.utils.feature_engineer import add_features
from sklearn.preprocessing import LabelEncoder


def train():
    df = load_data("../data/cleaned_data.csv")
    df = add_features(df)

    # Encode origin and destination
    origin_encoder = LabelEncoder()
    destination_encoder = LabelEncoder()

    df['origin_encoded'] = origin_encoder.fit_transform(df['origin'])
    df['destination_encoded'] = destination_encoder.fit_transform(df['destination'])

    features = ['hour_sin', 'hour_cos', 'day_of_week', 'is_weekend', 'is_holiday', 'lag_1_week', 'origin_encoded',
                'destination_encoded']
    target = 'ridership'

    model = LGBMRegressor(
        n_estimators=500,
        learning_rate=0.0005,
        max_depth=10,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    model.fit(df[features], df[target])

    # Save model and encoders
    joblib.dump({
        'model': model,
        'origin_encoder': origin_encoder,
        'destination_encoder': destination_encoder
    }, "../models/lightgbm_model.pkl")


if __name__ == "__main__":
    train()