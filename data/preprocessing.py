import pandas as pd


class Preprocessor:
    def __init__(
        self,
        clean: bool = True,
        lowercase: bool = False,
        label_sum: bool = True,
    ) -> None:
        self.clean = clean
        self.lowercase = lowercase
        self.label_sum = label_sum

    def pipeline(
        self,
        data: pd.DataFrame,
        label_columns: list[str],
        text_col: str,
    ):
        if self.lowercase:
            data[text_col] = data[text_col].str.lower()
        if self.clean:
            data[text_col] = data[text_col].dropna()
            count = data[text_col].str.split().str.len()
            data[text_col] = data[count > 1][text_col]

        if self.label_sum:
            data["target"] = data[label_columns].sum(axis=1)

        return data
