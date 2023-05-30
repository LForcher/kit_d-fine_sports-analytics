import pandas as pd
import numpy as np
import time
from joblib import Parallel, delayed

from src.utils import utils, db_handler
from src import config
from tqdm import tqdm
from src.metrics import team_information, pitch_divisions
from src.metrics import metric_utils

from src.postprocessing.postprocessing_utils import get_config_all_possible_metrics

three_key_cols = ["match_id", "half", "frame"]
stoppage_of_play = "StoppageOfPlay"
ball_going_out_of_play = "BallGoingOutOfPlay"
shot_at_goal = "ShotAtGoal"
ball_claiming = "BallClaiming"
unsuccessful_pass = "UnsuccessfulPass"
tackling_game = "TacklingGame"
no_possession_change = "NoBallPossessionChange"
raw_event_to_ball_gain_mapping = {
    "FinalWhistle": stoppage_of_play,
    "Foul": stoppage_of_play,
    "FreeKick": stoppage_of_play,
    "Offside": stoppage_of_play,
    "ThrowIn": ball_going_out_of_play,
    "GoalKick": ball_going_out_of_play,
    "CornerKick": ball_going_out_of_play,
    "ShotAtGoal": shot_at_goal,
    "BallClaiming": ball_claiming,
    "Play": unsuccessful_pass,
    "TacklingGame": tackling_game
}


def main():
    """
    Main method which creates the raw and final target dataset.
    This method is executed by running this script as suggested in the README.
    Returns: nothing, stores data into database.
    """

    # determine which matches have to be processed
    skip_existing_matches = config.get_skip_existing_matches(process="target_dataset")
    matches_to_process = config.get_matches_to_process()
    if isinstance(matches_to_process, str) and matches_to_process == "all":
        all_matches = utils.get_all_matches(utils.table_name_calculated_player_values)
    else:
        if not isinstance(matches_to_process, list):
            raise ValueError(f"# value for matches to process is not correct. given value: {matches_to_process}")
        all_matches = matches_to_process

    if skip_existing_matches and db_handler.has_table(utils.table_name_target_dataset_raw) and \
            not db_handler.is_table_empty(utils.table_name_target_dataset_raw):
        existing_matches = utils.get_all_matches(utils.table_name_target_dataset_raw)
        all_matches = [match for match in all_matches if match not in existing_matches]
    else:
        db_handler.drop_table(utils.table_name_target_dataset_raw)

    # use_parallelization: Use with caution, time advantage not proven yet
    use_parallelization = config.get_use_parallelization(process="target_dataset")
    if use_parallelization and len(all_matches) > 1:
        n_jobs = config.get_available_processes_for_multiprocessing(process="target_dataset")
        print(f"Parallelize creating raw target dataset using {n_jobs} different processes.")
        start_time = time.time()
        Parallel(n_jobs=n_jobs)(delayed(create_target_dataset)(match_id) for match_id in tqdm(all_matches))
        print(f"Processed in {np.round((time.time() - start_time) / 60., 2)} minutes")
    else:
        print(f"Creating raw target dataset without parallelization.")
        start_time = time.time()
        for match_id in tqdm(all_matches):
            create_target_dataset(match_id)
        print(f"Processed in {np.round((time.time() - start_time) / 60., 2)} minutes")

    prediction_horizon = 2
    target_dataset_with_final_target_variable(prediction_horizon)

    # add contextual information (see paper)
    base_table = utils.get_table_name_target_dataset(prediction_horizon)
    create_hypothesis_testing_target_set(base_table + "_hypothesis_testing", base_table, remove_nan=True)

    # create special set reduced to ball positions on pitch
    # i.e. to test if ball gains on outer lanes differ from inner lane, same for offensive/defensive thirds
    reduce_target_set_based_on_values(base_table + "_hypothesis_testing", "_middle_lane",
                                      filter_cons={"lane_middle": [1]})
    reduce_target_set_based_on_values(base_table + "_hypothesis_testing", "_outer_lanes",
                                      filter_cons={"lane_middle": [0]})
    reduce_target_set_based_on_values(base_table + "_hypothesis_testing", "_third_attacking",
                                      filter_cons={"third_attacking": [1]})
    reduce_target_set_based_on_values(base_table + "_hypothesis_testing", "_third_middle",
                                      filter_cons={"third_middle": [1]})
    reduce_target_set_based_on_values(base_table + "_hypothesis_testing", "_third_defending",
                                      filter_cons={"third_defending": [1]})


def reduce_target_set_based_on_values(name_target_set: str, suffix_new_target_set: str, filter_cons: dict):
    """
    Helper to create special set reduced to ball positions on pitch
    i.e. to test if ball gains on outer lanes differ from inner lane, same for offensive/defensive thirds
    Args:
        name_target_set: table name of the base target dataset
        suffix_new_target_set: which suffix to add to the new created table in the database
        filter_cons: key: column to filter, value: list of values

    Returns: nothing, stores data into database table name_target_set + suffix_new_target_set
    """
    df = db_handler.get_table_with_condition_dict(name_target_set, filter_cons)
    db_handler.write_df_to_postgres(name_target_set + suffix_new_target_set, df, if_exists="replace")


def remove_nan_values(df: pd.DataFrame) -> pd.DataFrame:
    """ Some classifiers can not handle nan values.
    As we have only very few nan value anyway (much less than 1 percent,
     happens e.g. after substitutions when player tracking starts few seconds later)
     all those values are dropped."""
    cols_only_nan_values = [col for col in df.columns if df[col].isnull().all()]
    print(f"Dropping cols with only nan values: ")
    print(cols_only_nan_values)
    df = df[[col for col in df.columns if col not in cols_only_nan_values]].copy()
    original_len = df.shape[0]
    df.dropna(inplace=True)
    print(f"Dropping na rows ({round((1 - (df.shape[0] / original_len)) * 100, 2)}% of total rows)")
    return df.copy()


def target_dataset_with_final_target_variable(prediction_horizon: int = 2):
    """
    Until this point the target variable described if there is a ball gain in the next frame.
    In the paper we predict, if there is a ball gain in the next two seconds.
    This method finalizes the target variable such that it classifies if there is a ball gain in the next
    prediction_horizon (in the paper = 2) seconds.

    Returns: nothing, stores data (e.g. target dataset with finalized target variable) into database

    """
    table_name = utils.get_table_name_target_dataset(prediction_horizon)
    table_name_raw_dataset = utils.table_name_target_dataset_raw
    matches = utils.get_all_matches(table_name_raw_dataset)
    db_handler.drop_table(table_name)
    for match in matches:
        # process match by match to reduce data load
        df = db_handler.get_table_with_condition(table_name_raw_dataset, "match_id", match)
        shift_frames = int(config.get_frames_per_second() * prediction_horizon)

        # note that we generally only want to make predictions for game situations
        # where at least for five seconds the same team is in possession of the ball.
        # I.e. we can calculate "is there a ball change in the next two seconds" from the previously
        # calculated "is there a ball change in the next second".
        df["target"] = target_in_x_next_frames(df["frame"], df["target"], shift_frames)
        df.sort_values(["match_id", "half", "frame"], inplace=True)
        target_data_list = list()
        for half in (1, 2):
            df_half = df[df["half"] == half]
            relevant_frames = [df_half["frame"].iloc[0]]
            for frame in df_half["frame"]:
                if frame - relevant_frames[-1] >= shift_frames:
                    relevant_frames.append(frame)
            target_data_list.append(df_half[df_half["frame"].isin(relevant_frames)].copy())
        target_data = pd.concat(target_data_list)
        db_handler.write_df_to_postgres(table_name, target_data, if_exists="append")


def create_hypothesis_testing_target_set(name: str, table_name_base_target_dataset: str,
                                         remove_nan: bool = True):
    """
    Add contextual information to the target dataset.
    Contextual information is information on the teams playing the matches (e.g. position in the final table) and
    information on where the regarded situation happens on the pitch (e.g. defending the middle lane).
    All the information is one hot encoded.
    Args:
        name: table name where to store the data
        table_name_base_target_dataset: name of the target dataset to use as basis
        remove_nan: whether to remove all nan values from the target dataset or not

    Returns: nothing, stores target dataset into database table table_name_base_target_dataset

    """
    print(f"Creating special set: {name}")
    base_target_dataset = db_handler.get_table(table_name_base_target_dataset)

    if remove_nan:
        target_dataset = remove_nan_values(base_target_dataset)
    else:
        target_dataset = base_target_dataset.copy()

    original_cols = target_dataset.columns
    final_set = list()
    for match_id, df_match in tqdm(target_dataset.groupby("match_id")):
        team_info = team_information.main(match_id)
        tmp_df = df_match.merge(team_info, how="left", on=["match_id", "half", "frame"], validate="one_to_one")
        pitch_df = pitch_divisions.main(match_id)
        tmp_df_2 = tmp_df.merge(pitch_df, how="left", on=["match_id", "half", "frame"], validate="one_to_one")
        res_df = metric_utils.one_hot_encode_cols(tmp_df_2, [col for col in tmp_df_2 if col not in original_cols])
        final_set.append(res_df)
    df_final = pd.concat(final_set)
    new_cols = [col for col in df_final if col not in original_cols]
    df_final[new_cols] = df_final[new_cols].fillna(0, inplace=False)
    db_handler.write_df_to_postgres(name, df_final, if_exists="replace")


def target_in_x_next_frames(frames: pd.Series, targets: pd.Series, x_next: int) -> np.array:
    """ check if in the next x frames the target variable is always 0 or at least one time 1"""
    new_target = targets.copy()
    for x in range(x_next):
        temp_frames = frames.shift(-x)
        temp_target = targets.shift(-x)
        target_adder = np.where(np.abs(temp_frames - frames) < x_next, temp_target, 0)
        new_target += target_adder
    new_target = (new_target != 0) * 1
    return new_target


def create_target_dataset(match_id: str):
    """
    Create raw target dataset. Therefore filter the frames to relevant frames based on assumptions,
    aggregate the base/raw metrics over time (as given by final metric names) and add the target column.
    Here the target column indicates whether the ball is gained in the following frame.
    Args:
        match_id: dfl match id

    Returns: a raw target dataset with features and a target column, where the target column represents a ball gain
        in the following frame. This raw target dataset is later entriched by more features (team and pitch information)
        and the target column is aggregated over more time.

    """
    positions = utils.get_position_data([match_id])

    # reduce frames to relevant frames: match for more than 5 seconds not interrupted
    positions["relevant_frames_no_ball_out_of_play"] = get_relevant_frames_no_ball_out_of_play(
        positions["ballstatus"], reduced_to_every_fifth_frame=False)
    # next line is necessary to correctly process the rows where one half starts and next begins
    full_window_same_half = positions["half"] == positions["half"].shift(
        (max(config.get_min_frames_ball_in_play(reduced_to_every_fifth_frame=False),
             config.get_min_frames_no_ball_change(reduced_to_every_fifth_frame=False))) - 1)

    positions["ball_change"] = positions["possession"] != positions["possession"].shift()
    positions["relevant_frames_no_ball_change"] = get_relevant_frames_no_ball_change(positions["ball_change"])
    if config.get_reduce_metrics_to_every_fifth_frame():
        # the data is downsampled to every fifth frame, because we eventually need even less frames
        # to handle the downsampling correctly, the ball changes have to be considered here
        positions["ball_change"] = target_in_x_next_frames(positions["frame"], positions["ball_change"], 5)
        fifth_frame_relevance = positions["frame"] % 5 == 0
        frame_difference_per_row = 5
    else:
        fifth_frame_relevance = True
        frame_difference_per_row = 1
    df_relevant_frames = positions[(positions["relevant_frames_no_ball_out_of_play"] == 1)
                                   & (positions["relevant_frames_no_ball_change"] == 1)
                                   & full_window_same_half
                                   & fifth_frame_relevance
                                   ].copy()

    target_data_list = list()

    for half in (1, 2):
        df_half = df_relevant_frames[df_relevant_frames["half"] == half]
        target_column = get_target_column(False, half, match_id, df_half["frame"], df_half["ball_change"])

        target_data = get_final_metrics(match_id, half, df_half["frame"])

        if target_data.shape[0] != len(target_column):
            raise Exception()
        target_data["target"] = target_column
        target_data["match_id"] = match_id
        target_data["half"] = half
        target_data_list.append(target_data.copy())
    df_final = pd.concat(target_data_list)

    db_handler.write_df_to_postgres(utils.table_name_target_dataset_raw, df_final, if_exists="append")


def get_target_column(granular_categories: bool, half: int, match_id: str, sorted_frames: pd.Series,
                      sorted_ball_changes: pd.Series):
    """
    For the first paper we only use the option granular_categories = False.
    As a future extension there will be the possibility to categorize ball_changes on basis on event data to investigate
    the differences in ball changes based on the type of ball change. Thus "granular_categories = True" is still a beta
    version which is currently in development.
    Args:
        granular_categories: if True, ball_change_categories are given, otherwise ball_change_column
        half: only data(i.e. frames) from one half can be given
        match_id: dfl match id
        sorted_frames: frame numbers (ascending)
        sorted_ball_changes: series of boolean values (ball possession change) corresponding to sorted frames

    Returns: target column, if granular_categories = False, then consisting of 0s and 1s.

    """
    if granular_categories:
        events = utils.get_events_with_corrected_frame_number(match_id).astype({"frame": "int64"})
        events["ball_change_category"] = map_events_to_ball_gain_category(events["event_type"])
        events.sort_values(["half", "frame"], inplace=True)
        events.sort_values(["frame_start", "ball_change_category"], ascending=True, inplace=True)
        # drop event duplicates (e.g. Tackling & OtherBallAction often describe the same event)
        events = events[events["event_type"] != "OtherBallAction"]
        events.drop_duplicates(subset="frame_start", keep="first", inplace=True)
        df_temp = pd.merge_asof(sorted_frames, events[events["half"] == half], on="frame",
                                direction="nearest")
        df_temp["ball_change_category"] = np.where(
            sorted_ball_changes == 0,
            no_possession_change,
            df_temp["ball_change_category"]
        )
        target_column = df_temp["ball_change_category"].tolist()
    else:
        target_column = sorted_ball_changes.tolist()
    return target_column


def get_relevant_frames_no_ball_out_of_play(ballstatus: pd.Series,
                                            reduced_to_every_fifth_frame: bool = False) -> pd.Series:
    """ Per default (i.e. in the paper) only situations where the ball was at least five seconds in the game are
    regarded. These frames are determined here. """
    frame_window = config.get_min_frames_ball_in_play(reduced_to_every_fifth_frame)
    ballstatus_sum_last_seconds = ballstatus.rolling(
        min_periods=1, window=frame_window).sum() - ballstatus
    relevant_frames = ballstatus_sum_last_seconds == (frame_window - 1)
    return relevant_frames


def get_relevant_frames_no_ball_change(ball_changes: pd.Series):
    """ Per default (i.e. in the paper) only situations where the ball was in possession of the same team for at least
    five seconds are regarded. These frames are determined here. """
    frame_window = config.get_min_frames_no_ball_change()
    ball_changes_last_seconds = ball_changes.rolling(
        min_periods=1, window=frame_window).sum() - ball_changes
    relevant_frames = ball_changes_last_seconds == 0
    return relevant_frames


def convert_weights_string_from_excel(str_weights: str or None):
    """ Weights for metrics - e.g. weight the velocity of the closest defender more than the rest.
    This is not used for the first paper, but left as a possible future extension."""
    if str_weights == "None" or str_weights is None:
        return None
    else:
        weights = str_weights.replace(" ", "").replace(",", ".").split("-")
        weights = [float(i) for i in weights]
        return weights


def get_final_metrics(match_id: str, half: int, relevant_frames: list) -> pd.DataFrame:
    """
    In the execute_metrics all metrics were calculated for all players and all teams -> here called raw metrics.
    Now all raw metrics are filtered to relevant players/teams and aggregated according to the metric names
    (over time as well as over space).
    Args:
        match_id: dfl match id
        half: half of the match
        relevant_frames: frames which are relevant for the target dataset

    Returns: dataframe with aggregated metric values

    """
    config_metrics = get_config_all_possible_metrics()
    base_metrics = config_metrics["base_metric"].unique().tolist()

    df_player_metrics = utils.get_metrics(level="player", match_ids=[match_id], halfs=[half],
                                          frames="all", metrics=base_metrics)
    df_team_metrics = utils.get_metrics(level="team", match_ids=[match_id], halfs=[half],
                                        frames="all", metrics=base_metrics)
    df_closest_players = utils.get_positions_attackers_defenders_data_sorted_by_distance_to_ball([match_id])
    df_closest_players = df_closest_players[df_closest_players["half"] == half]
    if config.get_reduce_metrics_to_every_fifth_frame():
        df_closest_players = df_closest_players[df_closest_players["frame"] % 5 == 0]
    df_player_metrics.sort_values(["half", "frame"], inplace=True)
    df_team_metrics.sort_values(["half", "frame"], inplace=True)
    frames = df_player_metrics["frame"].drop_duplicates().tolist()
    team_frames = df_team_metrics["frame"].drop_duplicates().tolist()
    if config.get_reduce_metrics_to_every_fifth_frame():
        if frames != list(range(0, len(frames) * 5, 5)) or frames != team_frames:
            raise ValueError("The frames were expected to be numbered (here every 5th frame) ...")
    else:
        if frames != list(range(len(frames))) or frames != team_frames:
            raise ValueError("The frames were expected to be numbered ...")
    metrics_list = list()
    regular_vals_list = list()

    # in the first loop the data is aggregated spatially -
    # that's why only time independent keys of the metric configurations matter
    time_indep_keys = ["base_metric", "attackers", "defenders", "attacker_weights", "defender_weights",
                       "defending_or_attacking"]
    for _, metric_config in config_metrics.groupby(time_indep_keys, dropna=False):
        agg_metric_vals = [0] * len(frames)
        control_vals = None
        if metric_config["team_or_player"].iloc[0] == "player":
            # metrics that are on player level (e.g. velocity)
            for ad in ["attacker", "defender"]:
                # whether to calculate values for defenders or attackers
                if metric_config[f"{ad}s"].iloc[0] > 0:
                    use_defenders = ad == "defender"
                    weights = convert_weights_string_from_excel(metric_config[f"{ad}_weights"].iloc[0])
                else:
                    # no values to aggregate: e.g. if ad==defender but only attackers are relevant for final metric
                    continue
                tmp_vals, control_vals = averaging_closest_players(df_closest_players.copy(), use_defenders,
                                                                   int(metric_config[f"{ad}s"].iloc[0]),
                                                                   weights,
                                                                   metric_config["base_metric"].iloc[0],
                                                                   df_player_metrics,
                                                                   )
                if tmp_vals.shape[0] == len(frames):
                    agg_metric_vals += tmp_vals
                else:
                    breakpoint()
                    raise ValueError("Control point #42: False implementation")

                # aggregate control vals to get one team per frame
                for idx in range(1, len(control_vals)):
                    if (control_vals[0] != control_vals[idx]).any():
                        raise Exception("Control point #43: Different teams for one metric in same frame!")
                control_vals = control_vals[0]

        elif metric_config["team_or_player"].iloc[0] == "team":
            # metrics on team level (e.g. numerical superiority)
            tmp_vals, control_vals = reduce_team_metric_to_team(df_closest_players.copy(), df_team_metrics,
                                                                metric_config["base_metric"].iloc[0],
                                                                metric_config["defending_or_attacking"].iloc[0],
                                                                )
            if tmp_vals.shape[0] == len(frames):
                agg_metric_vals += tmp_vals
            else:
                raise ValueError("false implementation")
        else:
            raise NotImplementedError("# no other option available")
        if control_vals is None:
            raise ValueError("# no values given!")
        # add time information -> metrics that include shifting, averaging and differencing values over time
        for _2, row in metric_config.iterrows():
            final_metric_vals, regular_vals = shift_and_average_values_over_time(
                pd.Series(agg_metric_vals, name=row["final_metric"]),
                row["period_calculator"],
                row["time_shift(seconds)"],
                row["len_period"],
                control_vals=control_vals,
                reduced_to_every_fifth_frame=True)

            metrics_list.append(final_metric_vals)
            regular_vals_list.append(regular_vals.copy())
    metric_df = pd.DataFrame(metrics_list).T

    # shift metrics by one, because we are looking into ball_changes -> then also attacker and defender roles change!
    if config.get_reduce_metrics_to_every_fifth_frame():
        frames_skipped = 5
    else:
        frames_skipped = 1
    metric_df["frame"] = frames
    metric_df["frame"] += frames_skipped

    # check if all values are regular (controls all aggregations)
    df_regular_vals = pd.DataFrame(regular_vals_list).T
    df_regular_vals["regular_value"] = df_regular_vals.all(axis=1)
    df_regular_vals["frame"] = frames
    df_regular_vals["frame"] += frames_skipped
    relevant_values_regularity = df_regular_vals[df_regular_vals["frame"].isin(relevant_frames)]
    if not relevant_values_regularity["regular_value"].all():
        raise Exception(
            "ERROR: Values were aggregated false! Defender or attacker values include values from both teams!")

    # fill last rows with nan
    frames_missing_info = range(frames_skipped)
    metric_df = pd.concat((metric_df, pd.DataFrame({"frame": frames_missing_info}, index=frames_missing_info)), axis=0)
    out_df = metric_df[metric_df["frame"].isin(relevant_frames)].copy()
    return out_df


def reduce_team_metric_to_team(df_closest_players: pd.DataFrame, df_metrics, base_metric,
                               attacking_or_defending) -> (np.array, np.array):
    """
    All metrics were calculated for both teams. This method determines which is defending/attacking team in
    the respective frames and returns the respective metric values according to the metric names for those frames.
    Args:
        df_closest_players: dataframe as given by table table_name_positions_of_attackers_and_defenders_dist_to_ball
                            this is here used because it also includes information on which team is attacking/defending
        df_metrics: dataframe with all values of the base metrics (also called raw metrics)
        base_metric: which metric values to take for averaging
        attacking_or_defending: whether to calculate values for the attacking or defending team

    Returns: metric values for the defending/attacking team as chosen,
                list of control values to ensure not mixing up values of different teams

    """
    if "def" in attacking_or_defending:
        key = "defending_team_id"
    else:
        key = "attacking_team_id"
    team_vals = df_metrics.merge(df_closest_players, left_on=["match_id", "half", "frame", "team_id"],
                                 right_on=["match_id", "half", "frame", key], how="inner",
                                 validate="one_to_one")[df_metrics.columns]
    out_vals = np.array(team_vals[base_metric].values)
    control_vals_team = team_vals["team_id"]
    return out_vals, np.array(control_vals_team.values)


def shift_and_average_values_over_time(metric_vals: pd.Series, period_calculator: str, time_shift_seconds: float,
                                       len_period: float, control_vals: np.array,
                                       reduced_to_every_fifth_frame: bool = False) -> (pd.Series, pd.Series):
    """
    This method calculates adds the time dimension to raw metrics. This means that values are averaged/shifted or the
    difference of values is calculated over a period of time.

    Args:
        metric_vals: raw metric values to add the time info to
        period_calculator: form of aggregation, values are "average" and "difference".
            If any other value is given, values are "shifted".
        time_shift_seconds: length of time interval that values shall be shifted
        len_period: in case of period_calculator = "average" or "difference" - time interval over that the average
            or respectively difference of values is taken
        control_vals: team id, this method checks if only values from one team are aggregated over time
        reduced_to_every_fifth_frame: all this code is written for a sampling frequence of 25 Hertz.
            If values were downsampled to 5 Hertz, set this to true.

    Returns: aggregated values (with included timely information),
     control values to mark those datapoints where only data of the same team is aggregated (only those datapoints
        should be considered for further calculations)


    """
    if period_calculator not in ("average", "difference"):
        len_period = 0
    time_shift_in_frames = int(time_shift_seconds * config.get_frames_per_second(reduced_to_every_fifth_frame))
    period_in_frames = int(len_period * config.get_frames_per_second(reduced_to_every_fifth_frame))
    # time_shift and period_in_frames must not be longer than min_seconds_ball_in_play and seconds_no_ball_change
    # shift of 1 is added because defenders are the attackers in the frame where the ball is gained
    len_control = time_shift_in_frames + period_in_frames + 1
    # lose one because max shift is one frame less (later we skip one frame, because defenders become attackers..)
    max_shift_value = min(config.get_min_frames_no_ball_change(reduced_to_every_fifth_frame),
                          config.get_min_frames_ball_in_play(reduced_to_every_fifth_frame)) - 1
    if len_control > max_shift_value:
        if len_control - 1 == max_shift_value:
            # some metrics aggregate over 5 seconds
            # -> that is not possible, if the ball is in possession for exactly 5 seconds
            # in that case the metric is aggregated over 5 seconds minus one frame
            if period_in_frames > max_shift_value:
                period_in_frames -= 1
            else:
                time_shift_in_frames -= 1
            len_control -= 1
        else:
            raise ValueError("#49 Values to take are further ago than ball change or ball out of possesion!")

    final_vals = metric_vals.shift(time_shift_in_frames)
    if period_calculator == "average" and len_period > 0:
        final_vals = final_vals.rolling(period_in_frames).mean()
    elif period_calculator == "difference" and len_period > 0:
        final_vals = final_vals - final_vals.shift(period_in_frames)

    control_series = pd.Series(control_vals, name="control")
    if len(control_series.unique()) != 2:
        raise Exception("Control series must have the two team_ids as unique values!")
    control_series_equals_first_team = (control_series == control_series.iloc[0]) * 1
    # the team has to be either always the first or always the second team for the entire rolling window!
    rw = control_series_equals_first_team.rolling(len_control)
    control_series_regular_value = (rw.sum() == len_control) | (rw.sum() == 0)
    return final_vals, control_series_regular_value


def averaging_closest_players(df_closest_players: pd.DataFrame, use_defenders: bool, number_players,
                              weights: None or list,
                              base_metric: str, df_metrics: pd.DataFrame, ) -> (np.array, list):
    """
    Averages the values for any metrics for the closest players.
    E.g. for velocity_3_defenders the velocity of the three defenders, that are closest to the ball, is averaged.
    The time dimension is not touched by this method.
    Args:
        df_closest_players: dataframe as given by table table_name_positions_of_attackers_and_defenders_dist_to_ball
        use_defenders: whether to average defenders or attackers
        number_players: how many players to consider
        weights: whether to weight the closest player to the ball more or equally
        base_metric: which metric values to take for averaging
        df_metrics: dataframe with all values of the base metrics

    Returns: array with averaged values, list of control values to ensure not mixing up values of different teams

    """

    metric_vals = list()
    control_vals_team = list()
    for idx in range(number_players):
        if use_defenders:
            key = f"defender_{idx}_id"
            key_team = f"defending_team_id"
        else:
            key = f"attacker_{idx}_id"
            key_team = "attacking_team_id"
        player_val = df_metrics.merge(df_closest_players, left_on=["match_id", "half", "frame", "player_id"],
                                      right_on=["match_id", "half", "frame", key], how="right",
                                      validate="one_to_one")[df_metrics.columns.tolist() + [key_team]]
        player_val.sort_values(["match_id", "half", "frame"], inplace=True)
        player_val.reset_index(inplace=True, drop=True)
        metric_vals.append(player_val[base_metric])
        control_vals_team.append(player_val[key_team].values)
    metric_vals = np.array(metric_vals)
    avg_vals = np.average(metric_vals, weights=weights, axis=0)
    return avg_vals, control_vals_team


def map_events_to_ball_gain_category(events: pd.Series) -> pd.Series:
    """ not used for first paper"""
    full_mapping = {}
    unique_event_values = events.unique()
    for key, value in raw_event_to_ball_gain_mapping.items():
        for event in unique_event_values:
            if key.lower() in event.lower():
                full_mapping[event] = value
    rv = events.map(full_mapping)
    return rv


if __name__ == "__main__":
    main()
