from collections import namedtuple
from pathlib import Path
from sklearn.model_selection import train_test_split
import pandas as pd


class Splitter:
    def __init__(
        self,
        data_file: str,
        out_folder: str,
        validation: bool = True,
        split_size: dict = dict(train=0.7, val=0.2, test=0.1),
    ) -> None:
        assert (
            validation and "val" in split_size
        ), "If you want to have validation set you have to provide at least a split_size of shape 3"

        assert (
            len(split_size.keys()) > 1 or len(split_size.keys()) < 3
        ), "The split_size must be 2 or 3"

        self.data = pd.read_csv(data_file, encoding="utf-8")
        self.out_folder = Path(out_folder)
        self.out_folder.mkdir(parents=True, exist_ok=True)
        self.split_size = split_size
        self.validation = validation

    def split_data(self, seed: int = 12):
        data_size = len(self.data)

        train_size = int(data_size * self.split_size.get("train"))
        test_size = int(data_size * self.split_size.get("test"))
        val_size = int(data_size * self.split_size.get("val")) if self.validation else 0

        round_diff = data_size - (train_size + test_size + val_size)
        if round_diff > 0:
            train_size += round_diff

        train = self.data

        if self.validation:
            train, val = train_test_split(
                self.data, test_size=val_size, random_state=seed
            )

        train, test = train_test_split(train, test_size=test_size, random_state=seed)

        train.to_csv(self.out_folder / "train.csv", index=False)
        if self.validation:
            val.to_csv(self.out_folder / "val.csv", index=False)
        test.to_csv(self.out_folder / "test.csv", index=False)

        return train, val if self.validation else None, test
