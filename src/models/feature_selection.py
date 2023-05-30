import pandas as pd
from typing import Union

from src.utils import utils, db_handler
from src.postprocessing.create_target_dataset import remove_nan_values

standard_key_columns = ["frame", "match_id", "half"]
path = "src/models/uncorrelated_features/"


def main(name_target_set_or_shift_seconds: Union[float, str] = 2.):
    """
    stores correlation analysis to excel.
    Calculates all correlations and marks correlated features for thresholds 50, 60 and 70 percent.
    Args:
        name_target_set_or_shift_seconds:  name of the table in the database

    Returns: nothing, stores data to excel

    """
    if isinstance(name_target_set_or_shift_seconds, float) or isinstance(name_target_set_or_shift_seconds, int):
        name_target_dataset = utils.get_table_name_target_dataset(name_target_set_or_shift_seconds)
    else:
        name_target_dataset = name_target_set_or_shift_seconds
    target_dataset = db_handler.get_table(name_target_dataset)
    target_dataset = remove_nan_values(target_dataset)

    for corr_threshold in (0.5, 0.6, 0.7):
        new_cols = get_uncorrelated_features(target_dataset, threshold=corr_threshold)
        filename = get_filename_uncorr_features_excel(corr_threshold, name_target_dataset)
        col_series = pd.Series(new_cols, name=f"Uncorrelated_features_{str(int(corr_threshold * 100))}")
        col_series.to_excel(filename)


def get_filename_uncorr_features_excel(corr_threshold: float, name_target_dataset: str):
    """
    Get the excel filename in which the main method stores the correlation analysis
    Args:
        corr_threshold: Threshold for the correlation
        name_target_dataset: name of the table in the database for which the correlation analysis was made

    Returns: filename

    """
    filename = path + name_target_dataset + f"_uncorr_features_{str(int(corr_threshold * 100))}.xlsx"
    return filename


def get_uncorr_features_from_file(corr_threshold: float, name_target_dataset: str):
    """
    load the correlation analysis from the main method for a specific threshold
    Args:
        corr_threshold: Threshold for the correlation
        name_target_dataset: name of the table in the database for which the correlation analysis was made

    Returns: filename
    """
    filename = get_filename_uncorr_features_excel(corr_threshold, name_target_dataset)
    df_feat = pd.read_excel(filename, sheet_name=0)
    cols = df_feat[f"Uncorrelated_features_{str(int(corr_threshold * 100))}"].values.tolist()
    return cols


def get_uncorrelated_features(df: pd.DataFrame, threshold: float = 0.8, key_columns: list or None = None,
                              method="spearman") -> list:
    """
    Returns list of features that are uncorrelated (less than given threshold) by spearman.
    Features that are highly correlated with target are chosen first.
    Args:
        df: df with features and (optional) key columns. key columns are ignored anyway. df should must column target.
        threshold: threshold (absolute) which is used for spearman correlation
        key_columns: columns which to ignore (i.e. key columns)
        method:
            pearson : standard correlation coefficient
            kendall : Kendall Tau correlation coefficient
            spearman : Spearman rank correlation

    Returns: list of uncorellated features + key_columns + ["target"]

    """
    if key_columns is None:
        key_columns = standard_key_columns
    value_columns = [col for col in df.columns if col not in key_columns]
    corr_matrix = df[value_columns].corr(method).abs()
    corr_matrix.sort_values("target", ascending=False, inplace=True)
    final_features = list()
    possible_features = [col for col in corr_matrix.index if col != "target"]
    for feat in possible_features:
        corr_matrix.loc[feat, feat] = 0  # put zero values to diagonal to ensure right maximum calculation
        if corr_matrix.loc[final_features + [feat], final_features + [feat]].max().max() <= threshold:
            final_features.append(feat)
        else:
            pass
    return key_columns + final_features + ["target"]


if __name__ == "__main__":
    main(name_target_set_or_shift_seconds="target_dataset_shift_2sec_hypothesis_testing")
