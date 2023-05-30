import numpy as np
import sqlalchemy.exc
from floodlight.io import dfl
from src.utils import db_handler
import json
import pandas as pd

from src.utils.utils import table_name_events, table_name_positions, table_name_pitch, table_name_code_mappings


def store_event_data_in_db(file_name: str, match_id: str, skip_existing_matches: bool):
    """
    Stores the event data from floodlight with some additional info in a db table.
    :param file_name: tuple of dataframes (t1_ht1, t1_ht2, t2_ht1, t2_ht2)
    :param match_id:
    :param skip_existing_matches: if False, all data for given match is dropped from db before inserting new data
    Returns: nothing, data is stored into database
    """
    if db_handler.has_table(table_name_events):
        if db_handler.value_exists_in_column(table_name_events, "match_id", match_id):
            if skip_existing_matches:
                return
            else:
                db_handler.delete_stm(table_name_events, "match_id", match_id)
    print(f"Processing event data for match {match_id}.")
    event_data = dfl.read_event_data_xml(file_name)
    df_all_list = []
    for event, team, half in zip(event_data, ["home", "home", "away", "away"],
                                 [1, 2, 1, 2]):
        df = event.events.copy()
        event_cols = df.columns.tolist()
        df["match_id"] = match_id
        df["half"] = half
        df["team"] = team
        df["qualifier"] = df["qualifier"].apply(lambda x: json.dumps(x))
        # put description columns to front
        df = df[[col for col in df.columns if col not in event_cols] + event_cols]
        df_all_list.append(df)
    df_all = pd.concat(df_all_list, axis=0)
    db_handler.write_df_to_postgres(table_name_events, df_all, if_exists="append")


def store_positions_data_in_db(tracking_file_name: str, file_match_info_name: str, match_id: str,
                               skip_existing_matches: bool):
    """
    Aggregates all the position data from floodlight into one db table.
    positions_data from floodlight is a tuple of dataframes (home_ht1, home_ht2, away_ht1, away_ht2, ball_ht1, ball_ht2,
     possession_ht1, possession_ht2, ballstatus_ht1, ballstatus_ht2, pitch)
    :param file_match_info_name: 
    :param tracking_file_name: 
    :param match_id:
    :param skip_existing_matches: if False, all data for given match is dropped from db before inserting new data
    Returns: nothing, data is stored into database
    """

    if db_handler.has_table(table_name_positions):
        if db_handler.value_exists_in_column(table_name_positions, "match_id", match_id):
            if skip_existing_matches:
                return
            else:
                db_handler.delete_stm(table_name_positions, "match_id", match_id)
    print(f"Processing positions data for match {match_id}.")
    positions_data = dfl.read_position_data_xml(tracking_file_name, file_match_info_name)

    home_ht, away_ht, ball_ht, possession_ht, ballstatus_ht = dict(), dict(), dict(), dict(), dict()
    (home_ht[1], home_ht[2], away_ht[1], away_ht[2], ball_ht[1], ball_ht[2], possession_ht[1], possession_ht[2],
     ballstatus_ht[1], ballstatus_ht[2], pitch) = positions_data
    for half in [1, 2]:
        cols = [f"home_player_{player}_{xy}" for player in range(home_ht[1].x.shape[1]) for xy in ["x", "y"]]
        cols += [f"away_player_{player}_{xy}" for player in range(away_ht[1].x.shape[1]) for xy in ["x", "y"]]
        cols += ["ball_x", "ball_y", "possession", "ballstatus"]
        data = np.concatenate((home_ht[half].xy, away_ht[half].xy, ball_ht[half].xy,
                               possession_ht[half].code.reshape((-1, 1)), ballstatus_ht[half].code.reshape((-1, 1))),
                              axis=1)
        df = pd.DataFrame(data, columns=cols)

        positions_cols = df.columns.tolist()
        df["match_id"] = match_id
        df["half"] = half
        df["frame"] = range(df.shape[0])
        # put description columns to front
        df = df[[col for col in df.columns if col not in positions_cols] + positions_cols]
        if home_ht[half].framerate != 25:
            raise Exception("Framerate different than usual!")
        try:
            print(f"Storing positions into db({match_id}, half {half}).")
            db_handler.write_df_to_postgres(table_name_positions, df, if_exists="append")
        except sqlalchemy.exc.ProgrammingError as _e:
            # this happens for example if in the match, that is first read, less players existed
            # because then a new column needs to be created
            existing_df = db_handler.get_table(table_name_positions)
            df_all = pd.concat((existing_df, df), axis=0)
            for col in positions_cols:
                df_all[col] = df_all[col].astype(float)
            print("Do not stop process! Table positions has to be rebuilt due to new additional player.")
            db_handler.write_df_to_postgres(table_name_positions, df_all, if_exists="replace")


def store_pitch_info(file_name: str, match_id: str, skip_existing_matches: bool):
    """
    Stores the pitch info (pitch size) in db table.
    :param file_name:
    :param match_id:
    :param skip_existing_matches: if False, all data for given match is dropped from db before inserting new data
    Returns: nothing, data is stored into database
    """
    if db_handler.has_table(table_name_pitch):
        if db_handler.value_exists_in_column(table_name_pitch, "match_id", match_id):
            if skip_existing_matches:
                return
            else:
                db_handler.delete_stm(table_name_pitch, "match_id", match_id)
    pitch = dfl.read_pitch_from_mat_info_xml(file_name)

    data = np.concatenate((np.asarray(pitch.xlim).reshape(1, -1), np.asarray(pitch.ylim).reshape(1, -1),
                           np.asarray(pitch.center).reshape(1, -1)), axis=1)
    cols = ["x_start", "x_end", "y_start", "y_end", "x_center", "y_center"]
    df = pd.DataFrame(data, columns=cols)
    df["match_id"] = match_id
    # put description columns to front
    df = df[[col for col in df.columns if col not in cols] + cols]
    db_handler.write_df_to_postgres(table_name_pitch, df, if_exists="append")


def store_mat_info_links(file_name: str, match_id: str, skip_existing_matches: bool):
    """
    Stores the code mappings (i.e. player identifiers) in db table. Documentation of identifiers see floodlight.
    :param file_name:
    :param match_id:
    :param skip_existing_matches: if False, all data for given match is dropped from db before inserting new data
    Returns: nothing, data is stored into database
    """
    if db_handler.has_table(table_name_code_mappings):
        if db_handler.value_exists_in_column(table_name_code_mappings, "match_id", match_id):
            if skip_existing_matches:
                return
            else:
                db_handler.delete_stm(table_name_code_mappings, "match_id", match_id)
    links_from_mat_info = dfl.create_links_from_mat_info(file_name, return_df_with_names=True)

    df_all = pd.DataFrame()
    for team in ["Home", "Away"]:
        data = links_from_mat_info[1][team]
        df = pd.DataFrame(data.items(), columns=["pid", "jid"])
        df["xid"] = df["jid"].apply(lambda x: links_from_mat_info[0][team][x])
        df["team"] = team.lower()
        df_all = pd.concat((df_all, df.copy()))
    df_all["match_id"] = match_id
    # reverse column order
    df_all = df_all[df_all.columns[::-1]]
    df_all["team_id"] = np.where(df_all["team"].str.lower() == "home",
                                 links_from_mat_info[2],
                                 links_from_mat_info[3])
    db_handler.write_df_to_postgres(table_name_code_mappings, df_all, if_exists="append")
    db_handler.write_df_to_postgres("mapping_player_names", links_from_mat_info[-1], if_exists="append")
