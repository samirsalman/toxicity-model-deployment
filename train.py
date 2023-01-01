from pathlib import Path
from typing import List
from data import DataWrapper, Preprocessor, Splitter
from model import ToxicityClassifier
import pandas as pd
from transformers import AutoTokenizer
import pytorch_lightning as pl
from typer import Option
import typer
from pytorch_lightning.callbacks import ModelCheckpoint, ModelPruning, RichProgressBar
from pytorch_lightning.loggers.wandb import WandbLogger


def train(
    data_path: str = Option("data/datasets/dataaset.csv", "-d", "--dataset"),
    bert_model: str = Option("bert-base-cased", "-b", "--bert"),
    label_columns: List[str] = Option(
        [
            "toxic",
            "severe_toxic",
            "obscene",
            "threat",
            "insult",
            "identity_hate",
        ],
        "-l",
        "--target",
    ),
    text_column: str = Option("comment_text", "-t", "--text"),
    data_outpath: str = Option("data/datasets/split", "-o", "--data-output"),
    max_len: int = Option(128, "-l", "--max-len"),
    dropout: float = Option(0.3, "-D", "--dropout"),
    max_epochs: int = Option(2, "-e", "--epochs"),
    batch_size: int = Option(32, "-b", "--batch-size"),
    seed: int = Option(97, "-s", "--seed"),
    model_outpath: str = Option("artifacts/model", "-m", "--model-output"),
):
    pl.seed_everything(seed=seed)

    data = pd.read_csv(data_path)

    preprocessor = Preprocessor()
    data = preprocessor.pipeline(
        data=data,
        label_columns=label_columns,
        text_col=text_column,
    )

    dataset_path = Path(data_outpath) / "dataset.csv"
    data.to_csv(str(dataset_path), index=False)

    splitter = Splitter(
        data_file=str(dataset_path),
        out_folder=data_outpath,
        split_size={"train": 0.7, "val": 0.2, "test": 0.1},
        validation=True,
    )

    train, val, test = splitter.split_data(seed=97)

    data_wrapper = DataWrapper(
        train=train,
        test=test,
        val=val,
        text_column=text_column,
        target_column="target",
        tokenizer=AutoTokenizer.from_pretrained(bert_model),
        max_len=max_len,
    )

    torch_wrapper = data_wrapper.generate_wrapper(batch_size=batch_size)

    classifier = ToxicityClassifier(
        bert_model=bert_model,
        dropout=dropout,
    )

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        callbacks=[
            ModelCheckpoint(
                dirpath=model_outpath,
                every_n_train_steps=5000,
                save_on_train_epoch_end=True,
                save_last=True,
                mode="min",
                monitor="val_loss",
            ),
            RichProgressBar(),
            ModelPruning("l1_unstructured", amount=0.5),
        ],
        precision=16,
        logger=WandbLogger(name="toxicity-classifier"),
    )

    trainer.fit(
        model=classifier,
        train_dataloaders=torch_wrapper.train,
        val_dataloaders=torch_wrapper.val,
    )


if __name__ == "__main__":
    typer.run(train)
