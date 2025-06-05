import torch
from torchmetrics import MeanAbsoluteError, MeanSquaredError, R2Score

from models.train_nbeats import NBeats
from utils.data_loader import load_data
from utils.feature_engineer import add_features
from utils.nbeats.nbeats_data_loader import prepare_data

# Load data
df = load_data("./data/cleaned_data.csv")
df = add_features(df)
_, val_loader, _ = prepare_data(df)

# Load trained model
ckpt_path = "./saved_models/nbeats_model.ckpt"
model = NBeats.load_from_checkpoint(ckpt_path)
model.eval()

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Metrics
mae = MeanAbsoluteError().to(device)
rmse = MeanSquaredError(squared=False).to(device)
r2 = R2Score().to(device)

all_preds, all_targets = [], []

with torch.no_grad():
    for x, y in val_loader:
        x, y = x.to(device), y.to(device)
        y_hat = model(x)
        all_preds.append(y_hat)
        all_targets.append(y)

# Combine predictions
y_pred = torch.cat(all_preds)
y_true = torch.cat(all_targets)

# Compute metrics
print(f"MAE  : {mae(y_pred, y_true).item():.4f}")
print(f"RMSE : {rmse(y_pred, y_true).item():.4f}")
print(f"R²    : {r2(y_pred, y_true).item():.4f}")
