import pandas as pd
from sklearn.preprocessing import OneHotEncoder


def one_hot_encode_cols(df: pd.DataFrame, cols_to_encode: list):
    """
    Helper method that creates one-hot-encoding for the given columns cols_to_encode in given df.
    Args:
        df:
        cols_to_encode: column names of columns which shall be one hot encoded

    Returns: copy of df where cols_to_encode are dropped and replaced by new one hot encoded columns

    """
    out_df = df.drop(columns=cols_to_encode, inplace=False)
    for col in cols_to_encode:
        if df[col].isin(["true", "false"]).all():
            out_df[col] = (df[col] == "true") * 1
        elif df[col].isin([True, False]).all():
            out_df[col] = df[col] * 1
        else:
            one_hot_enc = OneHotEncoder(sparse=False)
            transformed = one_hot_enc.fit_transform(df[[col]])
            for idx, cat in enumerate(one_hot_enc.categories_[0]):
                out_df[col + "_" + cat] = transformed[:, idx]
    return out_df
