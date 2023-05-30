import pandas as pd
import numpy as np
from src import config
from src.utils import utils
pd.options.mode.chained_assignment = "raise"
import ast


def extract_player_id(in_string):
    if "player" in in_string:
        out_string = '_'.join(in_string.split("_")[0:3])
    elif "ball" in in_string:
        out_string = '_'.join(in_string.split("_")[0:1])
    else:
        out_string = in_string.split("_")[0]
    return out_string


def get_player_id_column_names(in_df):
    column_names = list(set(in_df.columns)-set(config.status_column_names))
    return column_names


def get_player_ids(in_df):
    player_id = list(set(map(extract_player_id, get_player_id_column_names(in_df))))
    return player_id


def make_frame_number_unique(in_df):
    out_df = in_df.sort_values(by=['half', 'frame'], ascending=True).copy()
    out_df['frame'] = 1 + out_df.frame + (out_df.half-1) * (out_df.loc[out_df.half == 1, 'frame'].max() + 1)
    return out_df


def fill_time_column(in_df):
    out_df = in_df.copy()
    out_df['Time [s]'] = out_df.frame * (1 / config.get_frames_per_second())
    return out_df.sort_values(by=['half', 'frame'])


def map_player_ids(in_df):
    out_df = in_df.copy()
    player_ids = get_player_id_column_names(in_df)
    match_id = in_df.match_id.unique()
    if len(match_id) > 1:
        raise ValueError("mapping not supported for multiple matches")
    else:
        match_id = match_id[0]
    mapped_player_ids = list(map(lambda x: utils.get_player_id_and_team_from_col_name_in_positions_data(x, match_id)[0], player_ids))
    columns_dict = pd.Series(mapped_player_ids, index=player_ids).to_dict()
    out_df.rename(columns=columns_dict, inplace=True)
    return out_df


def reformat_velocities_dataframe_for_db(in_df):
    cols = ["match_id", "half", "frame"]
    v_gate = (in_df.columns.str.contains("_v")) & ~(in_df.columns.str.contains("_vx")) & ~(in_df.columns.str.contains("_vy"))
    column_names = list(map(extract_player_id, in_df.loc[:, v_gate].columns))
    velocity_df = in_df.loc[:, v_gate]
    velocity_df.columns = column_names
    out_df = map_player_ids(pd.concat([velocity_df, in_df[cols]], axis=1))
    out_df = melt_dataframe(out_df)
    return out_df.rename(columns={"variable": "player_id", 'value': 'velocity'})


def melt_dataframe(out_df):
    out_df = out_df.melt(id_vars=["match_id", "half", "frame"])
    return out_df


def filter_data_for_change(in_df, window=25, key='possession', ball_active_period=-1):
    """ filter_data_for_change( tracking_data, window=25, key='possession', ball_active_period=-1 )

        Filter tracking data on changes in key column, cut window around event if ball is active for at least ball_active_period.

        Parameters
        -----------
            window: number of frames to extract before and after event
            key: look for changes in key column
            ball_active_period: only extract data if ball is active for more than ball_active_period frames. Ignore ballstatus if -1
        Returrns
        -----------
           filtered dataframe
        """

    df_changes = in_df[key].diff().fillna(0)
    in_df[key + '_change'] = df_changes
    idx = in_df.loc[in_df[key + '_change'] != 0, 'frame']

    frame_list = list(idx)
    df_list = []
    for frame in frame_list:
        if frame >= window:
            gate = (in_df.frame >= frame - window) & (in_df.frame <= frame)
            active_ball = in_df.loc[(in_df.frame >= frame - ball_active_period) & (
                        in_df.frame <= frame), 'ballstatus'].sum() == ball_active_period
            if ball_active_period < 0:
                df_list.append(in_df.loc[gate, :])
            elif active_ball:
                df_list.append(in_df.loc[gate, :])

    return pd.concat(df_list, axis=0).drop(key + '_change', axis=1)


def get_velocity_column_names(in_df):
    mask = in_df.columns.str.contains("_v")
    return in_df.columns[mask]


def get_position_column_names(in_df):
    mask = (in_df.columns.str.contains("_x")) | (in_df.columns.str.contains("_y"))
    return in_df.columns[mask]


def get_velocity_df(in_df):
    columns = list(set(list(get_velocity_column_names(in_df))).union({'match_id', 'half', 'frame'}))
    return in_df[columns]


def get_positions_df(in_df):
    columns = list(set(list(get_position_column_names(in_df))).union({'match_id', 'half', 'frame'}))
    return in_df[columns]


def extract_home_player_id(in_string):
    if "home" in in_string:
        out_string = '_'.join(in_string.split("_")[0:3])
        return out_string


def extract_away_player_id(in_string):
    if "away" in in_string:
        out_string = '_'.join(in_string.split("_")[0:3])
        return out_string


def get_home_player_ids(in_df):
    player_id = list(set(map(extract_home_player_id, get_player_id_column_names(in_df))))
    return player_id


def get_away_player_ids(in_df):
    player_id = list(set(map(extract_away_player_id, get_player_id_column_names(in_df))))
    return player_id


def cut_dataframe_into_chunks_where_ball_is_active(in_df, n=125):
    in_df = in_df.sort_values(by=['half', 'frame']).reset_index(drop=True)
    list_df = [in_df[i:i + n] for i in range(0, in_df.shape[0], n)]
    chunk = []
    for i in range(0, len(list_df)):
        if list_df[i].ballstatus.all() == 1:
            chunk.append(list_df[i])
    return pd.concat(chunk, axis=0)


def get_qualifier_entries(in_df, entry='all'):
    qualifier = in_df.copy().qualifier.values.all()
    dictionary = ast.literal_eval(qualifier)
    qualifier_entry = dictionary.get(entry)
    if entry == 'all':
        return dictionary
    elif qualifier_entry != None:
        return qualifier_entry
    else:
        return ''


def extract_data_for_recipient(in_df):
    out_df = in_df.copy()
    out_df.qualifier = out_df.qualifier.fillna("{'Recipient':'None'}")
    pIDs = get_qualifier_entries(out_df, 'Recipient')
    active_player = pIDs
    out_df['recipient'] = active_player
    player_columns = in_df.columns[in_df.columns.str.contains(active_player)]
    columns = list(set(player_columns).union(set(config.target_data_columns)) - {'Time [s]'})
    return out_df[columns]


def get_recipient(in_string):
    dictionary = ast.literal_eval(in_string)
    qualifier_entry = dictionary.get('Recipient')
    if qualifier_entry != None:
        return qualifier_entry
    else:
        return ''


def reduce_data_to_recipient(in_df):
    out_df = in_df.copy()
    out_df.qualifier = out_df.qualifier.fillna("{'Recipient':'None'}")
    recipients = list(map(get_recipient, out_df.qualifier.values))
    out_df['recipient'] = recipients
    return out_df


def get_player_in_possession(in_string):
    dictionary = ast.literal_eval(in_string)
    qualifier_entry = dictionary.get('Player')
    if qualifier_entry != None:
        return qualifier_entry
    else:
        return ''


def reduce_data_to_player_in_possession(in_df):
    out_df = in_df.copy()
    out_df.qualifier = out_df.qualifier.fillna("{'Player':'None'}")
    player = list(map(get_player_in_possession, out_df.qualifier.values))
    out_df['player'] = player
    return out_df


def dist_to_ball(in_df):
    out_df = in_df.copy()
    ball_x = out_df.ball_x
    ball_y = out_df.ball_y
    player_id = list(set(get_player_ids(out_df)) - set(['match_id', 'half', 'frame', 'ball']))

    if len(player_id) > 2:
        raise ValueError("non-singlet player_id")

    player_x = out_df[player_id[0] + "_x"]
    player_y = out_df[player_id[0] + "_y"]

    out_df[player_id[0] + "_dist"] = np.sqrt((ball_x - player_x) ** 2 + (ball_y - player_y) ** 2)
    return out_df


def get_dist_to_ball(in_df):
    player_ids = list(set(get_player_ids(in_df.dropna(axis=1, how='all'))) - set(['ball', 'possession']))
    out_df_list = [in_df[['match_id', 'half', 'frame']]]
    match_ids = in_df.match_id.unique()

    if not (len(match_ids) == 1):
        raise ValueError(f"multiple match_id found {match_ids}")

    for player_id in player_ids:
        positions_single_player = in_df[['frame', 'ball_x', 'ball_y', player_id + "_x", player_id + "_y"]].dropna(
            axis=0)
        positions_and_velocities_single_player = dist_to_ball(positions_single_player)

        out_df_list.append(positions_and_velocities_single_player[[player_id + '_dist']])
    distances = pd.concat(out_df_list, axis=1)
    return pd.merge(in_df, distances, on=['match_id', 'half', 'frame'])


def fill_events(in_df):
    out_df = in_df.loc[(in_df.frame_end > in_df.frame_start), :].copy()
    df_repeated = out_df
    if len(out_df) > 0:
        df_size = (out_df.frame_end - out_df.frame_start).values[0] + 1
        df_repeated = pd.concat([df_repeated] * df_size, ignore_index=True).copy()
        df_repeated.frame = df_repeated.frame + df_repeated.index
    return df_repeated


def merge_positions_and_events(positions_df, events_df, expand_events=False):
    merged_df = pd.merge(positions_df, events_df, on=['match_id', 'half', 'frame'], how='outer', indicator=True)
    gate = (merged_df._merge == "both") | (merged_df._merge == "left_only")
    merged_df.frame_end = merged_df.frame_end.replace(np.inf, np.nan).fillna(-1)
    merged_df.frame_start = merged_df.frame_start.replace(np.inf, np.nan).fillna(-1)
    merged_df.loc[gate, :].drop(columns={'_merge'}).astype({'frame_start': 'int32', 'frame_end': 'int32'})
    merged_df.astype({'frame_start': 'int32', 'frame_end': 'int32'})
    if expand_events:
        events_expanded = events_df.reset_index(drop=True).groupby(['half', 'frame']).apply(fill_events).reset_index(drop=True)
    else:
        events_expanded = events_df.reset_index(drop=True)
    return pd.merge(positions_df, events_expanded, on=['match_id', 'half', 'frame'], how='outer', indicator=False)