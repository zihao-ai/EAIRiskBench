import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import json


def df_has_row(df, row):
    # 检查df中是否已经存在row里面的key value,row为dict类型，然后返回是否存在以及第一次出现的行数
    if (df[row.keys()] == row.values()).all():
        return True, df.index[df[row.keys()] == row.values()].tolist()[0]
    else:
        return False, None


def read_csv(file_path):
    df = pd.read_csv(file_path, index_col=False, skipinitialspace=True, escapechar="\\", quotechar='"')
    return df


def clean_json(json_str):
    if "json" in json_str:
        json_str = json_str.replace("json", "")
    if json_str.startswith("```\n"):
        json_str = json_str.replace("```\n", "")
    if json_str.endswith("\n```"):
        json_str = json_str.replace("\n```", "")
    json_str = json_str.replace("\n", "")
    return json_str


def read_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)


def save_json(data, file_path):
    with open(file_path, "w") as f:
        json.dump(data, f)
