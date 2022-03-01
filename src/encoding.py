import pandas as pd


def te_map(data: pd.core.frame.DataFrame, target_col: str, feature_col: str) -> dict:
    mappings = dict()
    table = data.groupby(feature_col).agg({target_col: "mean"})
    for item_num in range(len(table)):
        mappings.update({table.iloc[item_num, :].name: table.iloc[item_num, :][0]})
    return mappings


def target_encoding(
    data: pd.core.frame.DataFrame, target_col: str
) -> pd.core.frame.DataFrame:
    for obj_col in data.loc[:, data.dtypes == "object"]:
        data[obj_col].replace(te_map(data, target_col, obj_col), inplace=True)
    return data
