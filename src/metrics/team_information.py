import pandas as pd
import numpy as np

from src.utils import utils, db_handler


def main(match_id) -> pd.DataFrame:
    """
    Calculates additional information both playing teams on frame level. For further info see return value.
    Args:
        match_id: dfl match id

    Returns: df with team information. Columns:[
        'match_id', 'half', "frame", 'home_team_is_in_possession', 'attacking_team_is_team_with_more_possession',
        'home_team_buli_top_6', 'away_team_buli_top_6', 'home_team_buli_middle_6', 'away_team_buli_middle_6',
        'home_team_buli_bottom_6', 'away_team_buli_bottom_6']. All value columns are boolean.
    """
    positions = utils.get_positions_attackers_defenders_data([match_id])
    home_team, away_team = get_home_and_away_team(match_id)
    team_more_possession, team_less_possession = get_team_with_more_possession(positions["possession"],
                                                                               home_team, away_team)
    df = positions[["match_id", "half", "frame", "possession"]].copy()
    df["home_team"] = home_team
    df["away_team"] = away_team
    df["team_more_possession"] = team_more_possession
    df["team_less_possession"] = team_less_possession
    df["home_team_is_in_possession"] = df["possession"] == 1
    df["attacking_team_is_team_with_more_possession"] = np.where(df["home_team_is_in_possession"],
                                                                 df["home_team"] == df["team_more_possession"],
                                                                 df["away_team"] == df["team_more_possession"])
    out_df = df.replace(get_dfl_id_to_team_name_mapping())
    out_df["home_team_buli_top_6"] = out_df["home_team"].isin(get_team_names_places(1, 6))
    out_df["away_team_buli_top_6"] = out_df["away_team"].isin(get_team_names_places(1, 6))
    out_df["home_team_buli_middle_6"] = out_df["home_team"].isin(get_team_names_places(7, 12))
    out_df["away_team_buli_middle_6"] = out_df["away_team"].isin(get_team_names_places(7, 12))
    out_df["home_team_buli_bottom_6"] = out_df["home_team"].isin(get_team_names_places(13, 18))
    out_df["away_team_buli_bottom_6"] = out_df["away_team"].isin(get_team_names_places(13, 18))
    if (
            out_df["home_team_buli_top_6"] + out_df["home_team_buli_middle_6"] +
            out_df["home_team_buli_bottom_6"] * 1 != 1).any():
        raise ValueError("#one buli team not in buli table!")
    if (
            out_df["away_team_buli_top_6"] + out_df["away_team_buli_middle_6"] +
            out_df["away_team_buli_bottom_6"] * 1 != 1).any():
        raise ValueError("#one buli team not in buli table!")
    if (~out_df["home_team"].isin(get_buli_results_21_22().values()).any() or
            ~out_df["home_team"].isin(get_buli_results_21_22().values()).any()):
        raise ValueError(f"#missing decoding for {out_df['home_team'].iloc[0]} or {out_df['away_team'].iloc[0]}")

    # we reduce the number of columns to prevent that we overwhelm the resulting model
    out_df = out_df[[
        'match_id', 'half', "frame", 'home_team_is_in_possession', 'attacking_team_is_team_with_more_possession',
        'home_team_buli_top_6', 'away_team_buli_top_6', 'home_team_buli_middle_6', 'away_team_buli_middle_6',
        'home_team_buli_bottom_6', 'away_team_buli_bottom_6']]
    return out_df


def get_home_and_away_team(match_id: str) -> (str, str):
    """ get dfl id for home and away team. """
    code_mappings = db_handler.get_table(utils.table_name_code_mappings)
    code_mappings = code_mappings[code_mappings["match_id"] == match_id]
    home_team = code_mappings[code_mappings["team"] == "home"]["team_id"].unique()
    away_team = code_mappings[code_mappings["team"] == "away"]["team_id"].unique()
    if len(home_team) != 1 or len(away_team) != 1:
        raise ValueError("# Code mappings not unique!")
    else:
        home_team = home_team[0]
        away_team = away_team[0]
    return home_team, away_team


def get_team_with_more_possession(possession: pd.Series, home_team: str, away_team: str) -> (str, str):
    """
    Returns team name with more possession in the match
    Args:
        possession: Series of 0 and 1s as given in position data
        home_team: team name
        away_team: team name

    Returns: team_with_more_possession, team_less_possession

    """
    possession = (possession - 1).copy()
    if set(possession.values) - {0, 1} != set():
        raise ValueError("Wrong assumption that possession is in 0,1")
    if possession.sum() <= possession.shape[0] / 2:
        team_with_more_possession = home_team
        team_less_possession = away_team
    else:
        team_with_more_possession = away_team
        team_less_possession = home_team
    return team_with_more_possession, team_less_possession


def get_dfl_id_to_team_name_mapping() -> dict:
    mapping = {
        "DFL-CLU-00000F": team_name_frankfurt,
        "DFL-CLU-00000S": team_name_bochum,
        "DFL-CLU-000017": team_name_leipzig,
        "DFL-CLU-000004": team_name_gladbach,
        "DFL-CLU-00000D": team_name_stuttgart,
        "DFL-CLU-000015": team_name_bielefeld,
        "DFL-CLU-000003": team_name_wolfsburg,
        "DFL-CLU-00000V": team_name_union_berlin,
        "DFL-CLU-000002": team_name_hoffenheim,
        "DFL-CLU-00000A": team_name_freiburg,
        "DFL-CLU-00000B": team_name_leverkusen,
        "DFL-CLU-000008": team_name_koeln,
        "DFL-CLU-00000W": team_name_fuerth,
        "DFL-CLU-00000G": team_name_fcbayern,
        "DFL-CLU-000007": team_name_dortmund,
        "DFL-CLU-000006": team_name_mainz,
        "DFL-CLU-00000Z": team_name_hertha_berlin,
        "DFL-CLU-000010": team_name_augsburg,
        "DFL-CLU-000009": team_name_schalke,
        "DFL-CLU-00000E": team_name_bremen,
        "DFL-CLU-000N5D": team_name_magdeburg,
        "DFL-CLU-000005": team_name_nuernberg,
        "DFL-CLU-000016": team_name_darmstadt,

    }
    return mapping


def get_team_names_places(from_place: int, to_place: int) -> list:
    """
    team names from from_place to to_place (incl) in the final table of bundesliga season 21/22.
    Args:
        from_place: int, position in the table
        to_place: int, position in the table
    Returns: list of team names

    """
    if to_place < from_place:
        raise ValueError("# to_place < from_place")
    if to_place not in range(1, 19) or from_place not in range(1, 19):
        raise ValueError("# to_place or from_place not in range(1,19)")
    teams = [get_buli_results_21_22()[idx] for idx in range(from_place, to_place + 1)]
    return teams


def get_buli_results_21_22():
    results = {1: team_name_fcbayern,
               2: team_name_leipzig,
               3: team_name_dortmund,
               4: team_name_wolfsburg,
               5: team_name_frankfurt,
               6: team_name_leverkusen,
               7: team_name_union_berlin,
               8: team_name_gladbach,
               9: team_name_stuttgart,
               10: team_name_freiburg,
               11: team_name_hoffenheim,
               12: team_name_mainz,
               13: team_name_augsburg,
               14: team_name_hertha_berlin,
               15: team_name_bielefeld,
               16: team_name_koeln,
               17: team_name_bremen,
               18: team_name_schalke
               }
    return results


team_name_fcbayern = "Bayern München"
team_name_frankfurt = "Eintracht Frankfurt"
team_name_bochum = "VfL Bochum"
team_name_leipzig = "RB Leipzig"
team_name_gladbach = "Borussia M'Gladbach"
team_name_stuttgart = "VfB Stuttgart"
team_name_bielefeld = "Arminia Bielefeld"
team_name_wolfsburg = "VfL Wolfsburg"
team_name_union_berlin = "Union Berlin"
team_name_hoffenheim = "TSG Hoffenheim"
team_name_freiburg = "SC Freiburg"
team_name_leverkusen = "Bayer 04 Leverkusen"
team_name_koeln = "FC Köln"
team_name_fuerth = "Greuther Fürth"
team_name_dortmund = "Borussia Dortmund"
team_name_mainz = "FSV Mainz 05"
team_name_hertha_berlin = "Hertha BSC Berlin"
team_name_augsburg = "FC Augsburg"
team_name_bremen = "Werder Bremen"
team_name_schalke = "FC Schalke 04"
team_name_magdeburg = "FC Magdeburg"
team_name_nuernberg = "FC Nürnberg"
team_name_darmstadt = "Darmstadt 98"
