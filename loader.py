import os
from typing import Union

import pandas as pd


def load_data(path: str = "product_data.csv", fmt: str = "json") -> Union[pd.DataFrame, dict]:
    data_json = pd.read_csv(path)
    if fmt == "json":
        data = data_json.to_dict(orient="records")
        data_json = {}
        for line in data:
            data_json[line["name"]] = line
    return data_json

