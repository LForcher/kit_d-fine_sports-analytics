import pandas as pd
import numpy as np
import src.utils.db_handler as db_handler
import src.utils.tools as tools
from src import config

pd.options.mode.chained_assignment = "raise"

table_name_events = "events"
table_name_positions = "positions"
table_name_pitch = "pitches"
table_name_code_mappings = "code_mappings"
column_match_id = "match_id"
table_name_calculated_player_values = "calculated_player_values"
table_name_calculated_team_values = "calculated_team_values"
table_name_categorical_team_values = "categorical_team_values"
table_pos_to_event_mapping = "pos_to_event_mapping"
table_name_positions_of_attackers_and_defenders = "positions_attackers_defenders"
table_name_positions_of_attackers_and_defenders_dist_to_ball = "positions_attackers_defenders_sorted_dist_to_ball"
table_name_positions_players_on_pitch = "positions_players_on_pitch"
table_name_target_dataset_raw = "target_dataset_raw"

raw_data_path = "data/raw_data/"

COLUMN_DTYPES = {'match_id': 'string',
                 'half': 'int8', 'frame': 'int32',
                 'home_player_0_x': 'float32', 'home_player_0_y': 'float32',
                 'home_player_1_x': 'float32', 'home_player_1_y': 'float32',
                 'home_player_2_x': 'float32', 'home_player_2_y': 'float32',
                 'home_player_3_x': 'float32', 'home_player_3_y': 'float32',
                 'home_player_4_x': 'float32', 'home_player_4_y': 'float32',
                 'home_player_5_x': 'float32', 'home_player_5_y': 'float32',
                 'home_player_6_x': 'float32', 'home_player_6_y': 'float32',
                 'home_player_7_x': 'float32', 'home_player_7_y': 'float32',
                 'home_player_8_x': 'float32', 'home_player_8_y': 'float32',
                 'home_player_9_x': 'float32', 'home_player_9_y': 'float32',
                 'home_player_10_x': 'float32', 'home_player_10_y': 'float32',
                 'home_player_11_x': 'float32', 'home_player_11_y': 'float32',
                 'home_player_12_x': 'float32', 'home_player_12_y': 'float32',
                 'home_player_13_x': 'float32', 'home_player_13_y': 'float32',
                 'home_player_14_x': 'float32', 'home_player_14_y': 'float32',
                 'home_player_15_x': 'float32', 'home_player_15_y': 'float32',
                 'home_player_16_x': 'float32', 'home_player_16_y': 'float32',
                 'home_player_17_x': 'float32', 'home_player_17_y': 'float32',
                 'home_player_18_x': 'float32', 'home_player_18_y': 'float32',
                 'home_player_19_x': 'float32', 'home_player_19_y': 'float32',
                 'away_player_0_x': 'float32', 'away_player_0_y': 'float32',
                 'away_player_1_x': 'float32', 'away_player_1_y': 'float32',
                 'away_player_2_x': 'float32', 'away_player_2_y': 'float32',
                 'away_player_3_x': 'float32', 'away_player_3_y': 'float32',
                 'away_player_4_x': 'float32', 'away_player_4_y': 'float32',
                 'away_player_5_x': 'float32', 'away_player_5_y': 'float32',
                 'away_player_6_x': 'float32', 'away_player_6_y': 'float32',
                 'away_player_7_x': 'float32', 'away_player_7_y': 'float32',
                 'away_player_8_x': 'float32', 'away_player_8_y': 'float32',
                 'away_player_9_x': 'float32', 'away_player_9_y': 'float32',
                 'away_player_10_x': 'float32', 'away_player_10_y': 'float32',
                 'away_player_11_x': 'float32', 'away_player_11_y': 'float32',
                 'away_player_12_x': 'float32', 'away_player_12_y': 'float32',
                 'away_player_13_x': 'float32', 'away_player_13_y': 'float32',
                 'away_player_14_x': 'float32', 'away_player_14_y': 'float32',
                 'away_player_15_x': 'float32', 'away_player_15_y': 'float32',
                 'away_player_16_x': 'float32', 'away_player_16_y': 'float32',
                 'away_player_17_x': 'float32', 'away_player_17_y': 'float32',
                 'away_player_18_x': 'float32', 'away_player_18_y': 'float32',
                 'away_player_19_x': 'float32', 'away_player_19_y': 'float32',
                 'ball_x': 'float32', 'ball_y': 'float32',
                 'possession': 'int8', 'ballstatus': 'int8'}


def get_table_name_target_dataset(shift_seconds: float = 0):
    """ get name target dataset where target variable describes if there is a ball gain in the next shift_seconds."""
    if shift_seconds == 0:
        table_name = "target_dataset"
    else:
        if shift_seconds % 1 == 0:
            table_name = f"target_dataset_shift_{int(shift_seconds)}sec"
        elif shift_seconds % 1 == 0.5:
            table_name = f"target_dataset_shift_{int(shift_seconds)}_5sec"
        else:
            raise NotImplementedError("Please implement name!")
    return table_name


def get_events_with_corrected_frame_number(match_id: str) -> pd.DataFrame:
    """
    Returns event data with mapped frame number from matching positions.
    Args:
        match_id: dfl match id

    Returns: event data with frame_start and frame_end from matching

    """
    # finalwhistle is event for both teams
    events = get_event_data([match_id]).drop(columns="team", inplace=False).drop_duplicates()
    events.drop_duplicates(inplace=True)  # error in script: InSubstitutions marked as OutSubstitutions
    event_frames = db_handler.get_table_with_condition(table_pos_to_event_mapping, "match_id", match_id)
    event_frames = event_frames[["event_id", "frame_start", "frame_end", "event_type"]].drop_duplicates()
    # merge also on event_type because substitutions have two rows with same event_id,
    events = events.merge(event_frames, left_on=["dfl_event_id", "eID"], right_on=["event_id", "event_type"],
                          validate="one_to_one")
    events.drop(columns=["gameclock", "minute", "second", "event_id", "eID"], inplace=True)
    events.loc[(events.frame_start < 0), 'frame_start'] = 0
    events.loc[(events.frame_end < 0), 'frame_end'] = 0
    events.frame_end = events.frame_end.fillna(-1)
    events.frame_start = events.frame_start.fillna(-1)
    events['frame'] = events.frame_start
    return events.astype({'frame_start': 'int32', 'frame_end': 'int32', "frame": "int32"})


def get_merged_position_events(match_id: str, map_player_names=False) -> pd.DataFrame:
    """ Merges positions and events based on the frame numbers that were corrected in matching."""
    events = get_events_with_corrected_frame_number(match_id)
    if map_player_names:
        positions = tools.map_player_ids(get_position_data([match_id]))
    else:
        positions = get_position_data([match_id])
    events_and_positions = tools.merge_positions_and_events(positions, events, expand_events=True)
    return events_and_positions


def get_all_matches(table_name: str, exclude_blacklist=True) -> list:
    """
    Returns list of all available matches in given table
    Args:
        table_name: name of table
        exclude_blacklist: exclude blacklist matches from config ini

    Returns: list of match_ids

    """

    df = db_handler.get_distinct_col_values(table_name, column_match_id)
    if exclude_blacklist:
        df = df[~df["match_id"].isin(config.get_blacklist_matches())]

    return df[column_match_id].values.tolist()


def get_team_playing_first_half_left_to_right(positions: pd.DataFrame) -> str:
    """
    Args:
        positions: from db table
    Returns: "home" or "away"
    """
    if len(positions["match_id"].unique()) > 1:
        raise ValueError("Match_ID not unqiue!")
    first_half = positions[positions['half'] == 1]
    home_team_first_x = first_half[[col for col in first_half.columns if "home" in col and "x" in col]]
    away_team_first_x = first_half[[col for col in first_half.columns if "away" in col and "x" in col]]
    if (home_team_first_x.mean().sum() / 11) < (away_team_first_x.mean().sum() / 11):
        return "home"
    else:
        return "away"


def get_team_id(match_id: str, home_or_away: str) -> str:
    """
    Get the dfl-team-id
    Args:
        match_id: dfl-match-id
        home_or_away: "home" or "away"

    Returns: dfl team id

    """
    conditions = {"match_id": [match_id], "team": [home_or_away]}
    df = db_handler.get_table_with_condition_dict(table_name_code_mappings, conditions)[["team_id"]].drop_duplicates()
    if df.shape[0] != 1:
        raise ValueError("#2345 Something is wrong, team in mapping not unique.")
    return df["team_id"].iloc[0]


def get_position_left_goal_x():
    return -52.5


def get_goal_y():
    return 0


def get_position_right_goal_x():
    return 52.5


def get_attacked_goal_position(team_playing_first_half_left_to_right: str, half: int or list,
                               possession: int or list) -> (int or list, int or list):
    """

    Args:
        team_playing_first_half_left_to_right: "home" or "away"
        half: 1 or 2
        possession: 1 (home) or 2 (away)

    Returns: tuple with x and y value of goal position

    """
    left_goal_x = get_position_left_goal_x()
    right_goal_x = get_position_right_goal_x()
    goal_y = get_goal_y()

    if team_playing_first_half_left_to_right == "home":
        # home team plays first half from left to right
        return np.where(possession == half, right_goal_x, left_goal_x), goal_y

    elif team_playing_first_half_left_to_right == "away":
        # home team plays first half from right to left
        return np.where(possession == half, left_goal_x, right_goal_x), goal_y
    else:
        raise ValueError("#123 Check this, wrong value for team_playing_first_half_left_to_right")


def get_attacking_team(possession: int) -> str:
    """
    Args:
        possession: 1 (home) or 2 (away)

    Returns: "home" or "away"

    """
    if possession == 1:
        attacker = "home"
    else:
        attacker = "away"
    return attacker


def get_defending_team(possession: int) -> str:
    """
    Args:
        possession: 1 (home) or 2 (away)
    Returns: "home" or "away"
    """
    attacker = get_attacking_team(possession)
    if attacker == "home":
        defender = "away"
    else:
        defender = "home"
    return defender


def get_player_id_and_team_from_col_name_in_positions_data(col_name: str, match_id: str) -> (str, str):
    """
    Returns the player_id given a col_name in the positions_data
    Args:
        col_name: col_name of table positions
        match_id: DFL match id

    Returns: player_id and team ("home" or "away")

    """
    col_name_parts = col_name.split("_")
    if len(col_name_parts) < 3:
        # not a col name with a player id
        return col_name, "unknown"
    team = col_name_parts[0]
    xid = col_name_parts[2]
    conditions = {"match_id": [match_id], "xid": [xid], "team": [team]}
    df = db_handler.get_table_with_condition_dict(table_name_code_mappings, conditions)
    if df.shape[0] == 1:
        pid = df["pid"].item()
    else:
        if df.empty:
            raise ValueError(f"U2 match {match_id} or xid {col_name} not found in match_information!")
        raise ValueError(f"#U1 code mapping not unique for match {match_id}.")
    return pid, team


def get_player_id_suffix_and_team_from_col_name_in_positions_data(col_name: str, match_id: str) -> (str, str):
    """
    Returns the player_id given a col_name in the positions_data
    Args:
        col_name: col_name of table positions
        match_id: DFL match id

    Returns: player_id and team ("home" or "away")

    """
    col_name_parts = col_name.split("_")
    suffix = col_name_parts[-1]
    if len(col_name_parts) < 3:
        # not a col name with a player id
        return col_name, "unknown"
    team = col_name_parts[0]
    xid = col_name_parts[2]
    conditions = {"match_id": [match_id], "xid": [xid], "team": [team]}
    df = db_handler.get_table_with_condition_dict(table_name_code_mappings, conditions)
    if df.shape[0] == 1:
        pid = df["pid"].item()
    else:
        if df.empty:
            raise ValueError(f"U2 match {match_id} or xid {col_name} not found in match_information!")
        raise ValueError(f"#U1 code mapping not unique for match {match_id}.")
    return pid + f'_{suffix}', team


def get_xid_and_team_from_pid(pid: str, match_id: str) -> (int, str):
    """
    Returns the xID (see floodlight, xID is the column number that a player has in the positions data)
    Args:
        pid: DFL player ID
        match_id: DFL match ID

    Returns: xID (number between 1 and 20) and team (home or away)

    """
    conditions = {"match_id": [match_id], "pid": [pid]}
    df = db_handler.get_table_with_condition_dict(table_name_code_mappings, conditions)
    if df.shape[0] != 1:
        raise ValueError(f"""ValueError #U1 code mapping not unique.
        Looked for pID {pid} in match {match_id}, but got:
        {df}""")
    xid = int(df["xid"].iloc[0])
    team = df["team"].iloc[0]
    return xid, team


def get_pos_col_name(pid: str, match_id: str, x_or_y: str) -> str:
    """
    Get the col_name of given player in the positions data.
    Args:
        x_or_y: "x" or "y"
        pid: DFL player ID
        match_id: DFL match id

    Returns: col name as given in positions data (e.g. home_player_1_x)

    """
    xid, team = get_xid_and_team_from_pid(pid, match_id)
    return f"""{team}_player_{xid}_{x_or_y}"""


def get_event_data(list_of_match_ids: list or None = None) -> pd.DataFrame:
    """
    If list_of_match_ids is None, all event data is returned. Otherwise only given matches are requested.
    Args:
        list_of_match_ids: e.g. ['003C34', '003C35']
    Returns: event data (format as given in database table)
    """
    if list_of_match_ids is None:
        df = db_handler.get_table(table_name_events)
    else:
        df = db_handler.get_table_with_condition_list(table_name_events, column_match_id, list_of_match_ids)
    return df


def get_position_data(list_of_match_ids: list) -> pd.DataFrame:
    """
    Get position data. Position data is a very large table, thus only given matches are requested.
    Args:
        list_of_match_ids: e.g. ['003C34', '003C35']

    Returns: position data of selected matches (format as given in database table)
    """
    df = db_handler.get_table_with_condition_list(table_name_positions, column_match_id, list_of_match_ids)
    df = df.astype(COLUMN_DTYPES).dropna(axis=1, how="all").copy()
    df.sort_values(["match_id", "half", "frame"], inplace=True)
    df.reset_index(inplace=True, drop=True)
    return df


def get_positions_attackers_defenders_data(list_of_match_ids: list) -> pd.DataFrame:
    """
    Get position data. Position data is a very large table, thus only given matches are requested.
    Args:
        list_of_match_ids: e.g. ['003C34', '003C35']

    Returns: position data of selected matches (format as given in database table)
    """
    df = db_handler.get_table_with_condition_list(table_name_positions_of_attackers_and_defenders, column_match_id,
                                                  list_of_match_ids)
    df.sort_values(["match_id", "half", "frame"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def get_positions_attackers_defenders_data_sorted_by_distance_to_ball(list_of_match_ids: list) -> pd.DataFrame:
    """
    Get position data. Position data is a very large table, thus only given matches are requested.
    Args:
        list_of_match_ids: e.g. ['003C34', '003C35']

    Returns: position data of selected matches (format as given in database table)
    """
    df = db_handler.get_table_with_condition_list(table_name_positions_of_attackers_and_defenders_dist_to_ball,
                                                  column_match_id, list_of_match_ids)
    df.sort_values(["match_id", "half", "frame"], inplace=True)
    df.reset_index(inplace=True, drop=True)
    return df


def get_positions_players_on_pitch(list_of_match_ids: list) -> pd.DataFrame:
    """
    Get position data. Position data is a very large table, thus only given matches are requested.
    Args:
        list_of_match_ids: e.g. ['003C34', '003C35']

    Returns: position data of selected matches (format as given in database table)
    """
    df = db_handler.get_table_with_condition_list(table_name_positions_players_on_pitch, column_match_id,
                                                  list_of_match_ids)
    return df


def metrics_to_db(df: pd.DataFrame, level: str, metric_columns: None or list = None):
    """
    Transforms dataframe with metrics for players/teams into key-value-format and writes it into database.
    Args:
        df: mandatory columns:
                match_id: str
                half: int
                frame: int
                player_id or team_id: str, depending on level. If player, then should equal pId.
                other_player_id: str, OPTIONAL, if level = player. If player, then should equal pId.
            additional_columns:
                all additional columns are interpreted as metrics.
        level: "player" or "team" or "team_categorical":
            depending on whether to write player or team metrics into database
        metric_columns: Optional VALIDATION parameter: validate that the dataframe contains only the columns
                that shall be written into the database
    Returns:
        no return value, values are written into db.

    """
    if level == "player":
        key_columns = ["match_id", "half", "frame", "player_id", "other_player_id"]
        if "other_player_id" not in df:
            df["other_player_id"] = ""
        table_name = table_name_calculated_player_values
    elif level == "team":
        key_columns = ["match_id", "half", "frame", "team_id"]
        table_name = table_name_calculated_team_values
    elif level == "team_categorical":
        key_columns = ["match_id", "half", "frame"]
        table_name = table_name_categorical_team_values
    else:
        raise ValueError("#1 Please check value for parameter level in method player_metric_to_db")

    if set(key_columns) > set(df.columns):
        raise ValueError("""#2 Missing columns in df for method player_metric_to_db! 
        Please ensure that df has columns ['match_id', 'half', 'frame', 'player_id' """)

    # transform relational data into key-value-format
    additional_columns = [col for col in df.columns if col not in key_columns]
    if metric_columns is not None:
        if sorted(additional_columns) != sorted(metric_columns):
            raise ValueError("Please check if too many columns were given!")
    df_list = []
    for col in additional_columns:
        df_temp = df[key_columns + [col]].copy()
        df_temp.rename(columns={col: "metric_value"}, inplace=True)
        df_temp["metric"] = col
        df_list.append(df_temp)

    db_handler.write_df_to_postgres(table_name, pd.concat(df_list), if_exists="append")


def get_metrics(level: str, match_ids: list or str, halfs: list or str = "all",
                frames: list or str = "all", ids: str = "all", metrics: list or str = "all") -> pd.DataFrame:
    """
    Requests all metrics based on given filters. If all frames, halfs etc. shall be requested, use "all" as value.
    Args:
        match_ids: e.g. ['003C34', '003C35'], "all" is not allowed here!
        halfs: Either [1], [2], [1,2] or "all"
        frames: List of frame numbers or "all"
        level: "player" or "team" or "team_categorical"
        ids: list of ids or "all", have to correspond to level (i.e. either player or team ids)
        metrics: list of metrics

    Returns: Dataframe with metrics, transformed into relational format. Columns:
                match_id: str
                half: int
                frame: int
                player_id or team_id: str, depending on level. If player, then should equal pId
            additional_columns:
                all additional columns are interpreted as metrics.
    """
    filters = {"match_id": match_ids}
    if isinstance(halfs, list):
        filters["half"] = halfs
    elif halfs != "all":
        raise ValueError("#3 False value for halfs in get_metrics!")
    if isinstance(frames, list):
        filters["frame"] = frames
    elif frames != "all":
        raise ValueError("#4 False value for frames in get_metrics!")
    if isinstance(ids, list):
        filters[level + "_id"] = ids
    elif ids != "all":
        raise ValueError("#5 False value for ids in get_metrics!")
    if isinstance(metrics, list):
        filters["metric"] = metrics
    elif metrics != "all":
        raise ValueError("#6 False value for metrics in get_metrics!")
    elif not isinstance(match_ids, list):
        raise ValueError("#7 False value for matches in get_metrics!")

    if level == "player":
        table_name = table_name_calculated_player_values
    elif level == "team":
        table_name = table_name_calculated_team_values
    elif level == "team_categorical":
        table_name = table_name_categorical_team_values
    else:
        raise ValueError(f"#8 level wrong: {level}")

    df = db_handler.get_table_with_condition_dict(table_name, filters)
    if df.empty:
        return df

    # transform values from key-value-format into relational table
    key_columns = ["match_id", "half", "frame"]
    if level in ("player", "team"):
        key_columns += [level + "_id"]
    # TODO include other_player_id - currently this is ignored, because it causes problems and it is not necessary yet
    # problematic: different "other_player_id"s for same player (e.g. NULL and filled)
    # if level == "player":
    #    key_columns.append("other_player_id")
    df.sort_values(key_columns, inplace=True)
    df.reset_index(drop=True, inplace=True)
    df_all = None
    for metric, df_metric in df.groupby("metric"):
        df_temp = df_metric.rename(columns={"metric_value": metric})
        df_temp = df_temp[key_columns + [str(metric)]]
        if df_all is not None:
            try:
                df_all = pd.merge(df_all, df_temp, how="outer", on=key_columns, validate="one_to_one")
            except pd.errors.MergeError:
                raise pd.errors.MergeError(f"Merge keys not unique! Check match_ids {match_ids} in table {table_name},"
                                           f"metrics: {metric}")
        else:
            df_all = df_temp
    for col in ["half", "frame"]:
        df_all[col] = df_all[col].astype(int)
    df_all.sort_values(key_columns, inplace=True)
    df_all.reset_index(drop=True, inplace=True)
    return df_all


def euclid_metric(x1: float or pd.Series, y1: float or pd.Series, x2: float or pd.Series,
                  y2: float or pd.Series) -> float or pd.Series:
    """
    Calculate the 2 dimensional (squared) euclidean distance between (x1,y1) and (x2,y2)

    :param x1: value or series/list of values of x coordinates (player1/ball)
    :param y1: value or series/list of values of y coordinates (player1/ball)
    :param x2: value or series/list of values of x coordinates (player2/ball)
    :param y2: value or series/list of values of y coordinates (player2/ball)

    :return: int or list/pd.Series of int (if list/pd.Series is given)
    """
    try:
        return np.round(np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2), 2)
    except Exception:
        print("MPAE warning #9, distance between x and y could not be calculated.")
        breakpoint()
        raise Exception("check this")


def get_frame_number(gameclock: float) -> int:
    return int(np.round(gameclock * config.get_frames_per_second()))


def filter_positions_to_frames(positions: pd.DataFrame, half: int, gameclock: float, look_back: int,
                               look_ahead: int) -> (pd.DataFrame, int):
    """
    Filter the position data (as given from db table) to frames
    Args:
        positions: positions data from db table
        half: half to filter
        gameclock: gameclock (given by floodlight) to filter around
        look_back: frames (in seconds) to include before given time
        look_ahead: frames (in seconds) to include after given time

    Returns: filtered positions data, frame number which was filtered around

    """
    positions = positions[positions["half"] == half]
    current_frame = min(get_frame_number(gameclock), positions["frame"].max())
    frames = filter_frames(positions, current_frame, look_back, look_ahead)
    return frames


def filter_frames(positions: pd.DataFrame, center_frame: int, look_back: int, look_ahead: int) -> (pd.DataFrame, int):
    """
    Filter the position data (as given from db table) to frames
    Args:
        positions: (filtered) positions data from db table
        center_frame: frame to filter around
        look_back: frames (in seconds) to include before given time
        look_ahead: frames (in seconds) to include after given time

    Returns: filtered positions data, frame number which was filtered around

    """
    frames_per_second = config.get_frames_per_second()
    first_frame = center_frame - look_back * frames_per_second
    last_frame = center_frame + look_ahead * frames_per_second
    frames = positions[positions["frame"] >= first_frame]
    frames = frames[frames["frame"] <= last_frame]
    return frames.copy()
