# This file is to check the SARIMAX model trained on weekly seasonality data.
# model/inference/check_model.py



import pickle
import pandas as pd
import os

# Load the model
model_path = os.path.join(os.path.dirname(__file__), "../models/sarimax_model_weekly.pkl")
with open(model_path, 'rb') as f:
    model = pickle.load(f)

print("Model summary:")
print(model.summary())
print("\nModel exog names:", getattr(model.model, 'exog_names', 'No exog names'))
exog = getattr(model.model, 'exog', None)
print("Model exog shape:", exog.shape if exog is not None else 'No exog shape')
