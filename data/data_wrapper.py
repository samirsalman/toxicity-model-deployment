from dataclasses import dataclass
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from typing import List


@dataclass
class TorchData:
    train: DataLoader
    test: DataLoader
    val: DataLoader = None


class ToxicityDataset(Dataset):
    def __init__(
        self,
        data: pd.DataFrame,
        tokenizer: AutoTokenizer,
        max_len: int,
        text_col: str,
        target_col: str,
        is_test: bool = False,
    ) -> None:
        super().__init__()
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.text_col = text_col
        self.target_col = target_col
        self.is_test = is_test

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text = self.data[self.text_col].tolist()[index]
        target = (
            self.data[self.target_col].tolist()[index] if not self.is_test else None
        )
        tokenization = self.tokenizer.encode_plus(
            text,  # Sentence to encode.
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            max_length=self.max_len,  # Pad & truncate all sentences.
            return_attention_mask=True,  # Construct attn. masks.
            return_tensors="pt",  # Return pytorch tensors.
            truncation=True,
            padding="max_length",
        )
        return dict(
            text=text,
            target=torch.tensor(target, dtype=torch.float32) if self.is_test else -1,
            attention_mask=tokenization["attention_mask"].flatten(),
            input_ids=tokenization["input_ids"].flatten(),
        )


class DataWrapper:
    def __init__(
        self,
        train: pd.DataFrame,
        test: pd.DataFrame,
        text_column: str,
        target_column: str,
        tokenizer: AutoTokenizer,
        max_len: int = 128,
        val: pd.DataFrame = None,
    ) -> None:
        self.train = train
        self.test = test
        self.text_column = text_column
        self.target_column = target_column
        self.val = val
        self.tokenizer = tokenizer
        self.max_len = max_len

    def create_datasets(
        self,
    ):
        return [
            ToxicityDataset(
                data=data,
                tokenizer=self.tokenizer,
                max_len=self.max_len,
                text_col=self.text_column,
                target_col=self.target_column,
            )
            for data in [self.train, self.val, self.test]
            if data is not None
        ]

    def create_dataloaders(
        self, datasets: List[Dataset], batch_size: int, num_workers: int = 8
    ):

        return TorchData(
            train=DataLoader(
                datasets[0],
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
            ),
            val=None
            if len(datasets) < 3
            else DataLoader(
                datasets[1],
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
            ),
            test=DataLoader(
                datasets[1] if len(datasets) < 3 else datasets[2],
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
            ),
        )

    def generate_wrapper(self, batch_size: int) -> TorchData:
        return self.create_dataloaders(self.create_datasets(), batch_size=batch_size)
