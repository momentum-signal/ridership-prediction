#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
print(torch.cuda.get_device_name(0))


# In[2]:


import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint


# === Load and preprocess data ===
df = pd.read_csv("cleaned_data.csv")

if "datetime" not in df.columns:
    df["datetime"] = pd.to_datetime(df["date"] + " " + df["time"], format='%Y-%m-%d %H:%M')

df["route"] = df["origin"] + "_" + df["destination"]

le_origin = LabelEncoder()
df["origin_id"] = le_origin.fit_transform(df["origin"])

le_dest = LabelEncoder()
df["dest_id"] = le_dest.fit_transform(df["destination"])

scaler = StandardScaler()
df["ridership_scaled"] = scaler.fit_transform(df[["ridership"]])

# === Dataset class ===
class LSTMDataset(Dataset):
    def __init__(self, df, seq_len=7):
        self.seq_len = seq_len
        self.samples = []
        for route, group in df.groupby("route"):
            group = group.sort_values("datetime")
            for i in range(len(group) - seq_len):
                seq = group.iloc[i:i+seq_len]
                target = group.iloc[i+seq_len]["ridership_scaled"]
                self.samples.append({
                    "features": seq[["ridership_scaled", "day_of_week", "is_weekend", "is_holiday"]].values.astype(np.float32),
                    "origin_id": seq.iloc[-1]["origin_id"],
                    "dest_id": seq.iloc[-1]["dest_id"],
                    "target": np.float32(target)
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return (
            torch.tensor(sample["features"]),
            torch.tensor(sample["origin_id"]),
            torch.tensor(sample["dest_id"]),
            torch.tensor(sample["target"])
        )

# === Lightning Module ===
class LitLSTM(pl.LightningModule):
    def __init__(self, input_size, hidden_size, origin_vocab, dest_vocab, emb_dim):
        super().__init__()
        self.origin_emb = nn.Embedding(origin_vocab, emb_dim)
        self.dest_emb = nn.Embedding(dest_vocab, emb_dim)
        self.lstm = nn.LSTM(input_size + emb_dim * 2, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.loss_fn = nn.MSELoss()

    def forward(self, x_seq, origin_id, dest_id):
        origin_e = self.origin_emb(origin_id)
        dest_e = self.dest_emb(dest_id)
        emb_cat = torch.cat([origin_e, dest_e], dim=-1)
        emb_expanded = emb_cat.unsqueeze(1).repeat(1, x_seq.size(1), 1)
        x_combined = torch.cat([x_seq, emb_expanded], dim=-1)
        out, _ = self.lstm(x_combined)
        final_hidden = out[:, -1, :]
        return self.fc(final_hidden).squeeze()

    def training_step(self, batch, batch_idx):
        x_seq, origin_id, dest_id, y = batch
        y_hat = self(x_seq, origin_id, dest_id)
        loss = self.loss_fn(y_hat, y)
        self.log('train_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x_seq, origin_id, dest_id, y = batch
        y_hat = self(x_seq, origin_id, dest_id)
        loss = self.loss_fn(y_hat, y)
        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)







# In[34]:


# === Data preparation ===
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, shuffle=False)

train_dataset = LSTMDataset(train_df)
val_dataset = LSTMDataset(val_df)





# In[38]:


# Loader
'''
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)
'''
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=64,
    shuffle=True,
    num_workers=15,  # bump this up from 0 or 1
    persistent_workers=True,  # keep workers alive between epochs, smoother runs
    pin_memory=True  # if using CUDA, this helps speed a bit
)
val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=64,
    shuffle=False,           # usually no shuffle for validation
    num_workers=15,          # match your train loader or a bit less
    pin_memory=True,
    persistent_workers=True  # keep workers alive between epochs, smoother runs
)


# In[39]:


# Save the model with the lowest validation loss
checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',
    mode='min',
    save_top_k=1,
    filename='best-checkpoint-{epoch:02d}-{val_loss:.4f}',
    verbose=True
)

# === Initialize model ===
model = LitLSTM(
    input_size=4,  # ridership_scaled + 3 time features
    hidden_size=64,
    origin_vocab=df["origin_id"].nunique(),
    dest_vocab=df["dest_id"].nunique(),
    emb_dim=4
)



# === Trainer ===
'''
trainer = pl.Trainer(
    max_epochs=10,
    accelerator="cpu",  # or "mps" if you want to try GPU on M1
    devices=1,
    log_every_n_steps=10,
    enable_progress_bar=False,
    callbacks=[checkpoint_callback]

)
'''


# In[40]:


from pytorch_lightning.callbacks import EarlyStopping
'''
# Save
import joblib

joblib.dump(le_origin, 'le_origin.joblib')
joblib.dump(le_dest, 'le_dest.joblib')
joblib.dump(scaler, 'scaler.joblib')
'''
# Load
le_origin = joblib.load('le_origin.joblib')
le_dest = joblib.load('le_dest.joblib')
scaler = joblib.load('scaler.joblib')

early_stopping_callback = EarlyStopping(
    monitor='val_loss',
    patience=7,  # how many epochs to wait after last improvement
    verbose=True,
    mode='min'
)

trainer = pl.Trainer(
    max_epochs=50,
    accelerator="gpu",  # use "gpu" for CUDA-enabled GPU
    devices=1,          # number of GPUs to use
    log_every_n_steps=10,
    enable_progress_bar=False,  # might as well see the shiny bar now
    callbacks=[checkpoint_callback, early_stopping_callback],

)


# In[41]:


# === Train ===
model.train()
trainer.fit(model, train_loader, val_loader, ckpt_path='lightning_logs/version_3/checkpoints/best-checkpoint-epoch=09-val_loss=0.2550.ckpt')


# In[42]:


# Prediction time
from pytorch_lightning import LightningModule

best_model_path = "lightning_logs/version_7/checkpoints/best-checkpoint-epoch=17-val_loss=0.2432.ckpt"
model = LitLSTM.load_from_checkpoint(best_model_path,
    input_size=4,
    hidden_size=64,
    origin_vocab=df["origin_id"].nunique(),
    dest_vocab=df["dest_id"].nunique(),
    emb_dim=4
)
model.eval()
model.freeze()


# In[43]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

preds = []
targets = []

with torch.no_grad():
    for x_seq, origin_id, dest_id, y in val_loader:
        # Move data to the same device as model
        x_seq = x_seq.to(device)
        origin_id = origin_id.to(device)
        dest_id = dest_id.to(device)
        y = y.to(device)

        y_hat = model(x_seq, origin_id, dest_id)

        # Move predictions and targets back to CPU before converting to NumPy
        preds.append(y_hat.cpu().numpy())
        targets.append(y.cpu().numpy())

# Flatten
preds = np.concatenate(preds)
targets = np.concatenate(targets)


# In[44]:


preds_unscaled = scaler.inverse_transform(preds.reshape(-1, 1)).flatten()
targets_unscaled = scaler.inverse_transform(targets.reshape(-1, 1)).flatten()


# In[45]:


import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.plot(targets_unscaled, label='Actual')
plt.plot(preds_unscaled, label='Predicted')
plt.legend()
plt.title('Actual vs Predicted Ridership')
plt.xlabel('Sample')
plt.ylabel('Ridership')
plt.tight_layout()
plt.show()


# In[46]:


from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

preds_original = scaler.inverse_transform(preds.reshape(-1, 1)).flatten()
targets_original = scaler.inverse_transform(targets.reshape(-1, 1)).flatten()

# Already have `preds` and `targets` from earlier
mse = mean_squared_error(targets_original, preds_original)
rmse = np.sqrt(mse)
mae = mean_absolute_error(targets_original, preds_original)
r2 = r2_score(targets_original, preds_original)

print(f"📊 Evaluation Metrics:")
print(f"  MSE  = {mse:.4f}")
print(f"  RMSE = {rmse:.4f}")
print(f"  MAE  = {mae:.4f}")
print(f"  R²   = {r2:.4f}")


# In[49]:


import matplotlib.pyplot as plt

# Optionally inverse transform if you scaled
# preds = scaler.inverse_transform(preds.reshape(-1, 1)).flatten()
# targets = scaler.inverse_transform(targets.reshape(-1, 1)).flatten()

plt.figure(figsize=(14, 6))
plt.plot(targets_original[-1000:], label='Actual', linewidth=2)
plt.plot(preds_original[-1000:], label='Predicted', linewidth=2)
plt.title('🚌 Ridership Prediction vs Actual')
plt.xlabel('Time Steps')
plt.ylabel('Scaled Ridership')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# In[ ]:




