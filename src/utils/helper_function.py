import pandas as pd

def rename(df, object_name):
    df = df.rename(columns=object_name)
    return df
def filter(dataset, name_col: str, name_value: list):
    data = dataset[dataset[name_col].isin(name_value)]
    return data