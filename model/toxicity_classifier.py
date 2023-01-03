import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from transformers import AutoTokenizer, AutoModel
from torchmetrics import MeanSquaredError
from typing import Any


class ToxicityClassifier(pl.LightningModule):
    def __init__(
        self, bert_model: str, lr: float = 2e-3, dropout: float = 0.3, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.bert_model = bert_model
        self.lr = lr
        self.text_model = AutoModel.from_pretrained(bert_model)
        self.fn = nn.Linear(self.text_model.config.hidden_size, 1)
        self.relu = nn.ReLU()
        self.criterion = nn.BCEWithLogitsLoss()
        self.dropout = nn.Dropout(dropout)
        self.save_hyperparameters()
        self.metric = MeanSquaredError()

    def forward(self, attention_mask, input_ids, *args, **kwargs) -> Any:
        h = self.text_model(
            attention_mask=attention_mask, input_ids=input_ids, return_dict=True
        ).pooler_output
        h = self.relu(h)
        h = self.dropout(h)
        logits = self.fn(h)
        return logits.view(-1)

    def training_step(self, data, *args, **kwargs):
        out = self(attention_mask=data["attention_mask"], input_ids=data["input_ids"])
        loss = self.criterion(out, data["target"].float())
        mse = self.metric(out, data["target"].float())

        self.log_dict(
            {"train_loss": loss, "train_mse": mse},
            prog_bar=True,
            on_step=True,
            on_epoch=True,
        )

        return loss

    def validation_step(self, data, *args, **kwargs):
        out = self(attention_mask=data["attention_mask"], input_ids=data["input_ids"])
        loss = self.criterion(out, data["target"].float())
        mse = self.metric(out, data["target"].float())

        self.log_dict(
            {"val_loss": loss, "val_mse": mse},
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )

        return loss

    def test_step(self, data, *args, **kwargs):
        out = self(attention_mask=data["attention_mask"], input_ids=data["input_ids"])
        loss = self.criterion(out, data["target"].float())
        mse = self.metric(out, data["target"].float())

        self.log_dict(
            {"test_loss": loss, "test_mse": mse},
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )

        return loss

    def configure_optimizers(self):
        return optim.AdamW(self.parameters(), lr=self.lr)
