import src.utils.tools
import pandas as pd
from src import config
from src.metrics import defensive_pressure, calculate_velocities, organization, formation_lines, \
    numerical_superiority, distance_to_the_ball
from src.utils import db_handler, utils
import time
from joblib import Parallel, delayed
from tqdm import tqdm


def main():
    """ main script which executes all metric calculations steps that are selected in the config.ini.
    All metric values are stored into the database in a key value format.
    This method is executed when running the script as suggested in the readme.
    """
    calculated_match_metric_pairs = get_calculated_match_metric_pairs()
    matches_metrics = get_match_metric_pairs_that_have_to_be_calculated(calculated_match_metric_pairs)

    # use_parallelization: Use with caution, time advantage not proven yet
    use_parallelization = config.get_use_parallelization(process="postprocessing")
    if use_parallelization and len(matches_metrics) > 1:
        n_jobs = config.get_available_processes_for_multiprocessing(process="postprocessing")
        print(f"Parallelize calculation of metrics into {n_jobs} different processes.")
        print(f"(Starts are tracked, process needs ca. 10 - 15 min after last start).")
        Parallel(n_jobs=n_jobs)(
            delayed(execute_metric)(match_metric, calculated_match_metric_pairs) for match_metric in
            tqdm(matches_metrics))
    else:
        print(f"Calculation of metrics without parallelization.")
        for match_metric in tqdm(matches_metrics):
            execute_metric(match_metric, calculated_match_metric_pairs)


def execute_metric(match_metric: tuple, calculated_match_metric_pairs: pd.DataFrame):
    """
    Calculates metric and stores values into database (in key value format).
    Args:
        match_metric: (match, metric) determines which metric to calculate for which match
        calculated_match_metric_pairs:  as given from method get_calculated_match_metric_pairs

    Returns: nothing, metric values are stored in database

    """
    match = match_metric[0]
    metric = match_metric[1]
    start_time = time.time()
    player_or_team_metric = config.metrics[metric]
    if player_or_team_metric == "player":
        calculated_table = utils.table_name_calculated_player_values
    else:
        calculated_table = utils.table_name_calculated_team_values

    # check if match has to be skipped or deleted from values
    player_or_team_metric = config.metrics[metric]
    if not config.get_skip_existing_matches("postprocessing") and (
            is_metric_calculated(metric, match, calculated_match_metric_pairs)):
        db_handler.delete_stm(calculated_table, "match_id", match)
    print(f"Executing metric {metric} for match {match}")
    if metric == "defensive_pressure":
        df = defensive_pressure.defensive_pressure(match_id=match)
    elif metric == "distance_to_the_ball":
        df = distance_to_the_ball.distance_to_the_ball(match_id=match)
    elif metric == "velocity":
        df = calculate_velocities.velocities(match)
        if not df.empty:
            df = src.utils.tools.reformat_velocities_dataframe_for_db(df)[
                ["match_id", "half", "frame", "player_id", "velocity"]]
    elif metric == "organization":
        df = organization.organization(match)
    elif metric == "formation_lines":
        df = formation_lines.distances_between_formation_lines(match)
    elif metric == "numerical_superiority":
        df = numerical_superiority.numerical_superiority(match)
    else:
        raise NotImplementedError("#24 This metric is not implemented yet!")

    if df.empty:
        print(f"Warning: No values written into db for metric {metric} for match {match}!")
    else:
        if config.get_reduce_metrics_to_every_fifth_frame():
            df = df[df["frame"] % 5 == 0]
        utils.metrics_to_db(df, player_or_team_metric)
        print(f"---{metric} executed in {round((time.time() - start_time) / 60., 2)} minutes ---")


def get_match_metric_pairs_that_have_to_be_calculated(calculated_match_metric_pairs: pd.DataFrame) -> list:
    """ based on the configurations, this method determines which metrics have to be calculated for which matches"""
    matches_to_process = config.get_matches_to_process()
    if isinstance(matches_to_process, str) and matches_to_process == "all":
        matches_to_process = utils.get_all_matches(utils.table_name_events, exclude_blacklist=True)
    all_metrics = [metric for metric in config.metrics.keys() if config.get_execute_metric(metric)]
    match_metric_pairs = list()
    for match in matches_to_process:
        for metric in all_metrics:
            if config.get_skip_existing_matches(process="postprocessing") and (
                    is_metric_calculated(metric, match, calculated_match_metric_pairs)
            ):
                continue
            match_metric_pairs.append((match, metric))
    return match_metric_pairs


def get_calculated_match_metric_pairs() -> pd.DataFrame:
    """
    To reduce database requests the dataframe includes information about all calculated metrics (on match level).
    """
    df_list = list()
    for calculation_table in [utils.table_name_calculated_player_values, utils.table_name_calculated_team_values]:
        if not db_handler.has_table(calculation_table):
            temp_df = pd.DataFrame(data=[], columns=["match_id", "metric"])
        else:
            statement = f"select distinct match_id, metric from {calculation_table}"
            temp_df = db_handler.postgres_to_df(statement)
        df_list.append(temp_df)
    calculated_metrics = pd.concat(df_list)
    return calculated_metrics


def is_metric_calculated(metric: str, match_id: str, calculated_match_metric_pairs: pd.DataFrame) -> bool:
    """
    checks if the metric values for given match are already stored in the database
    Args:
        metric:
        match_id:
        calculated_match_metric_pairs:

    Returns: boolean value if metric for given match is already stored in database
    """
    if metric == "organization":
        metric = "surface_area"
    elif metric == "numerical_superiority":
        metric = "numerical_superiority_own_half"
    elif metric == "formation_lines":
        metric = "dis_def_mid_defending_team"
    return calculated_match_metric_pairs[
               (calculated_match_metric_pairs["match_id"] == match_id) &
               (calculated_match_metric_pairs["metric"] == metric)
               ].shape[0] > 0


if __name__ == "__main__":
    main()
