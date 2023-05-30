import pandas as pd
import numpy as np
from src.utils import utils
from src.utils import db_handler
from src import config
import json


def permutate_positions_into_attacker_and_defender_cols(match_id: str):
    """
    Restructures the position data such that there are 22 columns containing attacker positions (x and y) and
    22 columns containing defender positions.
    Pre-condition: Table table_name_positions_players_on_pitch has to be filled (see previous preprocessing step)
    """
    df = utils.get_positions_players_on_pitch([match_id])
    if df.empty:
        print(f"Warning: No positions for {match_id}! Skipping match!")
        return
    team_left_to_right = utils.get_team_playing_first_half_left_to_right(df)
    # determine goal position
    goal_x, goal_y = utils.get_attacked_goal_position(
        team_left_to_right, half=df["half"], possession=df["possession"]
    )
    df["attacked_goal_x"] = goal_x
    df["attacked_goal_y"] = goal_y
    df["attacking_team_id"], df["defending_team_id"] = _get_attacking_and_defending_team(match_id, df["possession"])
    for i in range(11):
        condition = df["possession"] == 1
        df[f"attacker_{i}_id"] = np.where(condition, df[f"home_player_{i}_id"], df[f"away_player_{i}_id"])
        df[f"attacker_{i}_val_x"] = np.where(condition, df[f"home_player_{i}_x"], df[f"away_player_{i}_x"])
        df[f"attacker_{i}_val_y"] = np.where(condition, df[f"home_player_{i}_y"], df[f"away_player_{i}_y"])
        df[f"defender_{i}_id"] = np.where(~condition, df[f"home_player_{i}_id"], df[f"away_player_{i}_id"])
        df[f"defender_{i}_val_x"] = np.where(~condition, df[f"home_player_{i}_x"], df[f"away_player_{i}_x"])
        df[f"defender_{i}_val_y"] = np.where(~condition, df[f"home_player_{i}_y"], df[f"away_player_{i}_y"])
        df.drop(columns=[f"home_player_{i}_id", f"home_player_{i}_x", f"home_player_{i}_y", f"away_player_{i}_id",
                         f"away_player_{i}_x", f"away_player_{i}_y"], inplace=True)

    db_handler.write_df_to_postgres(utils.table_name_positions_of_attackers_and_defenders, df, if_exists="append")
    return df


def permutate_attacker_and_defender_cols_by_distance_to_ball(match_id: str):
    """
    Restructures the position data such that there are 22 columns containing attacker positions (x and y) and
    22 columns containing defender positions.
    Pre-condition: Table table_name_positions_of_attackers_and_defenders has to be filled
     (see previous preprocessing step)
    """
    in_df = utils.get_positions_attackers_defenders_data([match_id])
    if in_df.empty:
        print(f"Warning: No positions for {match_id}! Skipping match!")
        return
    out_df = in_df[["match_id", "half", "frame", "ball_x", "ball_y", "possession", "ballstatus", "attacked_goal_x",
                    "attacked_goal_y", "attacking_team_id", "defending_team_id"]].copy()
    distances = {}
    for player in ["attacker", "defender"]:
        for idx in range(11):
            distances[idx] = utils.euclid_metric(in_df["ball_x"], in_df["ball_y"],
                                                 in_df[f"{player}_{idx}_val_x"], in_df[f"{player}_{idx}_val_y"])
        distances = pd.DataFrame(distances)
        ranks = distances.rank(axis=1, method="first", na_option="bottom") - 1
        for out_idx in range(11):
            out_df[f"{player}_{out_idx}_id"] = np.nan
            out_df[f"{player}_{out_idx}_val_x"] = np.nan
            out_df[f"{player}_{out_idx}_val_y"] = np.nan
            for in_idx in range(11):
                condition = ranks[in_idx] == out_idx
                out_df[f"{player}_{out_idx}_id"] = np.where(condition, in_df[f"{player}_{in_idx}_id"],
                                                            out_df[f"{player}_{out_idx}_id"])
                out_df[f"{player}_{out_idx}_val_x"] = np.where(condition, in_df[f"{player}_{in_idx}_val_x"],
                                                               out_df[f"{player}_{out_idx}_val_x"])
                out_df[f"{player}_{out_idx}_val_y"] = np.where(condition, in_df[f"{player}_{in_idx}_val_y"],
                                                               out_df[f"{player}_{out_idx}_val_y"])
    db_handler.write_df_to_postgres(utils.table_name_positions_of_attackers_and_defenders_dist_to_ball, out_df,
                                    if_exists="append")
    return out_df


def _get_attacking_and_defending_team(match_id: str, possession: int or list) -> (
        str or list, str or list):
    """
    Helper function to determine which team is defending in which frame.
    Args:
        match_id: dfl match id
        possession: 1 (home) or 2 (away)

    Returns:

    """
    home_team_id = utils.get_team_id(match_id, "home")
    away_team_id = utils.get_team_id(match_id, "away")

    attacking_team_ids = np.where(possession == 1, home_team_id, away_team_id)
    defending_team_ids = np.where(possession == 2, home_team_id, away_team_id)

    return attacking_team_ids, defending_team_ids


def sort_positions_by_teams(match_id: str):
    """
    Restructures the position data such that there are only 44 columns containing the players on the pitch
    (x and y data). I.e. there are no nan values in the columns for players on the bench (except for situations
    where players have seen red cards).
    Pre-condition: Table table_name_positions has to be filled (see previous preprocessing step)
    """
    positions = utils.get_position_data([match_id])
    if positions.empty:
        print(f"Warning: No positions for {match_id}! Skipping match!")
        return
    positions.dropna(axis=1, inplace=True, how="all")

    # replace identifiers
    identifiers = {col: utils.get_player_id_and_team_from_col_name_in_positions_data(col, match_id)[0]
                   for col in positions.columns if "player" in col}
    identifiers_dfl_to_x = {identifiers[col_x]: col_x for col_x in identifiers.keys() if "_x" in col_x}

    # order cols
    player_cols = [col for col in positions.columns if "player_" in col]
    other_cols = [col for col in positions.columns if col not in player_cols]
    player_x_vals = [col for col in player_cols if "_x" in col]
    player_order_helper = positions[player_x_vals].sum(axis=0)
    player_order_helper.sort_values(ascending=True, inplace=True)
    player_cols_order = list()
    for col in player_order_helper.index:
        player_cols_order += [col, col.replace("_x", "_y")]
    pos_filled = ~positions.isna()
    starters_cols_order = [col for col in player_cols_order if pos_filled[col].iloc[0]]
    player_home_cols_order = [col for col in starters_cols_order if "home" in col]
    player_away_cols_order = [col for col in starters_cols_order if "away" in col]

    # get events for substitutions
    events = utils.get_event_data([match_id])
    # In and OutSubstitution are usually the same, but later duplicates are dropped anyway-> make sure to forget no one
    events = events[(events["eID"] == "InSubstitution") | (events["eID"] == "OutSubstitution")]
    qualifiers = events["qualifier"].values.tolist()
    qualifiers = [json.loads(qu) for qu in qualifiers]
    substitutions_dfl_ids = {qu["PlayerOut"]: qu["PlayerIn"] for qu in qualifiers}
    substitutions_x = {identifiers_dfl_to_x[col]: identifiers_dfl_to_x[substitutions_dfl_ids[col]]
                       for col in substitutions_dfl_ids.keys()}
    substitutions_y = {col.replace("_x", "_y"): substitutions_x[col].replace("_x", "_y")
                       for col in substitutions_x.keys()}
    substitutions = substitutions_x | substitutions_y

    # rearrange positions
    df_list = [positions[other_cols]]
    for idx, col in enumerate(player_home_cols_order + player_away_cols_order):
        if idx >= 22:
            name = f"away_player_{int((idx - 22) / 2)}"
        else:
            name = f"home_player_{int(idx / 2)}"
        name += col[-2:]

        if col in substitutions:
            series = pd.Series(np.where(positions[col].isna(), positions[substitutions[col]], positions[col]),
                               name=name)
            list_of_names = np.where(positions[col].isna(), identifiers[substitutions[col]], identifiers[col])

        else:
            series = positions[col]
            list_of_names = [identifiers[col]] * positions.shape[0]
            series.name = name

        if series.isna().any():
            # if less than 10 seconds of nan values, fill values by ffill/bfill
            if series.isna().sum() < config.get_frames_per_second() * 10:
                series.ffill(inplace=True)
            else:
                print(f"# There are lots of nan values! Happened in {match_id} for {name} - probably a red card.")
        df_list.append(series)
        if "_y" in col:
            series_names = pd.Series(list_of_names, name=name.replace("_y", "") + "_id")
            df_list.append(series_names)
    df = pd.concat(df_list, axis=1)

    db_handler.write_df_to_postgres(utils.table_name_positions_players_on_pitch, df, if_exists="append")
