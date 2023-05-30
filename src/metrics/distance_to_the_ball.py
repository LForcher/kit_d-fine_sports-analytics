import pandas as pd
from src import config
from src.utils import utils


def distance_to_the_ball(match_id: str) -> pd.DataFrame:
    """
    Calculation of distance to the ball for every timestamp for every player.
    Args:
        match_id: DFL match ID

    Returns: dataframe with columns ["match_id", "half", "frame", "player_id", "distance_to_the_ball"]

    """
    positions_attackers_defenders = utils.get_positions_attackers_defenders_data([match_id])
    if config.get_reduce_metrics_to_every_fifth_frame():
        positions_attackers_defenders = positions_attackers_defenders[positions_attackers_defenders["frame"] % 5 == 0]
        positions_attackers_defenders.reset_index(inplace=True)
    if positions_attackers_defenders.empty:
        print(f"Warning: No position data, thus skipping metric defensive_pressure, for match {match_id}")
        return pd.DataFrame()
    df = calc_distances(positions_attackers_defenders)
    return df


def calc_distances(positions_attacker_defenders: pd.DataFrame) -> pd.DataFrame:
    """
    Calculation of distance to the ball for every timestamp for every player.
    Args:
        positions_attacker_defenders: as given by table table_name_positions_of_attackers_and_defenders

    Returns: dataframe with columns ["match_id", "half", "frame", "player_id", "distance_to_the_ball"]

    """
    all_results = []
    for a_d in ["attacker", "defender"]:
        for idx in range(11):
            df = positions_attacker_defenders.copy()

            df["distance_to_the_ball"] = utils.euclid_metric(
                positions_attacker_defenders[f"{a_d}_{idx}_val_x"],
                positions_attacker_defenders[f"{a_d}_{idx}_val_y"],
                positions_attacker_defenders["ball_x"],
                positions_attacker_defenders["ball_y"]
            )
            df.rename(columns={f"{a_d}_{idx}_id": "player_id"}, inplace=True)
            df = df[["match_id", "half", "frame", "player_id", "distance_to_the_ball"]]
            all_results.append(df)
    df_final = pd.concat(all_results, axis=0)
    return df_final
