import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")
phobert = AutoModel.from_pretrained("vinai/phobert-base-v2")

def rename(df, object_name):
    df = df.rename(columns=object_name)
    return df
def filter(dataset, name_col: str, name_value: list):
    data = dataset[dataset[name_col].isin(name_value)]
    return data

def get_feature_vector(lst):
    lst_tokenizer = [torch.tensor([tokenizer.encode(key)]) for key in lst]
    with torch.no_grad():
        feature_vectors = {}
        for index, input_ids in enumerate(lst_tokenizer):
            feature_vectors[lst[index]] = phobert(input_ids).pooler_output
    return feature_vectors