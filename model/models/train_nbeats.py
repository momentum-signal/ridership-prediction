import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch import optim
from typing import Tuple
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.preprocessing import MinMaxScaler

# Import from your data loader module
from model.utils.nbeats.nbeats_data_loader import prepare_data
from model.utils.data_loader import load_data
from model.utils.feature_engineer import add_features


class NBeatsBlock(nn.Module):
    def __init__(self,
                 input_size: int,
                 theta_size: int,
                 basis_function: nn.Module,
                 n_layers: int,
                 layer_width: int):
        super().__init__()
        self.input_size = input_size
        self.theta_size = theta_size
        self.basis_function = basis_function

        layers = []
        for i in range(n_layers):
            layers.append(nn.Linear(input_size if i == 0 else layer_width, layer_width))
            layers.append(nn.ReLU())
        self.fc_stack = nn.Sequential(*layers)
        self.theta_f = nn.Linear(layer_width, theta_size, bias=False)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.fc_stack(x)
        theta = self.theta_f(x)
        return self.basis_function(theta)


class GenericBasis(nn.Module):
    def __init__(self, backcast_size: int, forecast_size: int):
        super().__init__()
        self.backcast_size = backcast_size
        self.forecast_size = forecast_size

    def forward(self, theta: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return theta[:, :self.backcast_size], theta[:, -self.forecast_size:]


class NBeats(pl.LightningModule):
    def __init__(self,
                 input_size: int,
                 output_size: int = 1,
                 n_stacks: int = 30,
                 n_layers: int = 4,
                 layer_width: int = 512,
                 learning_rate: float = 1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate

        self.stacks = nn.ModuleList()
        for _ in range(n_stacks):
            block = NBeatsBlock(
                input_size=input_size,
                theta_size=input_size + output_size,
                basis_function=GenericBasis(input_size, output_size),
                n_layers=n_layers,
                layer_width=layer_width
            )
            self.stacks.append(block)

        self.final_layer = nn.Linear(n_stacks * output_size, output_size)
        self.loss_fn = nn.MSELoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        forecasts = []
        for stack in self.stacks:
            backcast, forecast = stack(x)
            forecasts.append(forecast)
            x = x - backcast
        forecast = torch.cat(forecasts, dim=1)
        return self.final_layer(forecast)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.learning_rate)


def train_model(data_path: str = "model/data/cleaned_data.csv",
                n_stacks: int = 30,
                n_layers: int = 4,
                layer_width: int = 512,
                learning_rate: float = 1e-3,
                max_epochs: int = 25,
                patience: int = 25) -> Tuple[NBeats, MinMaxScaler]:
    # Load and prepare data
    df = load_data(data_path)
    df = add_features(df)
    train_loader, val_loader, scaler = prepare_data(df)
    input_size = 5

    # Initialize model
    model = NBeats(
        input_size=input_size,
        n_stacks=n_stacks,
        n_layers=n_layers,
        layer_width=layer_width,
        learning_rate=learning_rate
    )

    # Configure trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=patience, mode="min"),
            ModelCheckpoint(
                monitor="val_loss",
                dirpath="model/saved_models/",
                filename="nbeats-best",
                save_top_k=1
            )
        ],
        logger=TensorBoardLogger("lightning_logs", name="nbeats"),
        accelerator="auto",
        devices="auto"
    )

    # Train model
    trainer.fit(model, train_loader, val_loader)
    return model, scaler


if __name__ == "__main__":
    model, scaler = train_model()
    print("Model training completed successfully!")