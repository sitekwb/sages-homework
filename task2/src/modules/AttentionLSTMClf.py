import pytorch_lightning as pl
import torch
import torch.nn as nn


class AttentionLSTMClf(pl.LightningModule):
    def __init__(self, n_features: int, hidden_size: int = 10, num_layers: int = 2, dropout: float = 0.01,
                 sequence_length: int = 512, out_features: int = 4, learning_rate: float = 0.02,
                 batch_size: int = 32):
        super().__init__()
        self.lstm = nn.LSTM(input_size=n_features,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            dropout=dropout,
                            batch_first=True,
                            bidirectional=True)
        self.tanh = nn.Tanh()
        self.linear = nn.Linear(in_features=hidden_size*2*sequence_length*2,
                                out_features=out_features)

        self.hidden_size = hidden_size
        self.loss_function = nn.NLLLoss()
        self.softmax = nn.Softmax()
        self.lr = learning_rate
        self.batch_size = batch_size

    def forward(self, x):
        lstm_output, lstm_hidden = self.lstm.forward(x.float())
        lstm_resized = lstm_output.reshape(self.batch_size, -1)
        lstm_activated = self.tanh(lstm_resized)
        linear_output = self.linear.forward(lstm_activated)
        pred = self.softmax(linear_output)
        return pred

    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self.forward(x)
        loss = self.loss_function(pred, y)
        self.log("train_loss", loss, prog_bar=True)
        return {'loss': loss, 'pred': pred}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self.forward(x)
        loss = self.loss_function(pred, y)
        self.log("val_loss", loss, prog_bar=True)
        return {'loss': loss, 'pred': pred}

    def test_step(self, batch, batch_idx):
        x, y = batch
        pred = self.forward(x)
        loss = self.loss_function(pred, y)
        self.log("test_loss", loss, prog_bar=True)
        return {'loss': loss, 'pred': pred}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)