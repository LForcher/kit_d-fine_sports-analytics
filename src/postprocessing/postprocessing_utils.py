import pandas as pd
from src.utils import db_handler, utils

no_metric_vals_available_for_attackers = ["dis_def_att_defending_team", "dis_def_mid_defending_team",
                                          "dis_mid_att_defending_team", "number_defenders_10m_fromball",
                                          "number_defenders_20m_fromball", "number_defenders_15m_lastdefender",
                                          "number_defenders_own_final_third", "number_defenders_own_half",
                                          "numerical_superiority_10m_fromball",
                                          "numerical_superiority_15m_lastdefender",
                                          "numerical_superiority_20m_fromball", "numerical_superiority_own_final_third",
                                          "numerical_superiority_own_half"
                                          ]
no_player_metric_vals_available_for_defenders = ["defensive_pressure"]


def get_all_team_metrics() -> list:
    """ Get all team metrics from database"""
    metrics = db_handler.get_distinct_col_values(utils.table_name_calculated_team_values, "metric")[
        "metric"].values.tolist()
    return metrics


def get_all_player_metrics() -> list:
    """ Get all player metrics from database"""
    metrics = db_handler.get_distinct_col_values(utils.table_name_calculated_player_values, "metric")[
        "metric"].values.tolist()
    return metrics


def get_config_all_possible_metrics() -> pd.DataFrame:
    """
    helper method that creates dataframe with all implemented metric combinations
    Returns: df with all implemented metric combinations

    """
    final_metrics_list = list()

    player_metrics = get_all_player_metrics()
    for base_metric in player_metrics:
        metric = pd.Series({"team_or_player": "player",
                            "base_metric": base_metric})
        for n_players in [1, 3, 5]:
            for time_shift_seconds in [0, 1, 3, 5]:
                metric["time_shift(seconds)"] = time_shift_seconds
                metric["len_period"] = 0
                for a_d in ["attackers", "defenders"]:
                    if a_d == "defenders" and base_metric in no_player_metric_vals_available_for_defenders:
                        continue
                    f_m = metric.copy()
                    f_m[a_d] = n_players
                    f_m["final_metric"] = f"{base_metric}_shift_{time_shift_seconds}_sec_{n_players}_{a_d}"
                    final_metrics_list.append(f_m.copy())
            for len_period in [1, 3, 5]:
                metric["time_shift(seconds)"] = 0
                metric["len_period"] = len_period
                for period_calc in ["average", "difference"]:
                    for a_d in ["attackers", "defenders"]:
                        if a_d == "defenders" and base_metric in no_player_metric_vals_available_for_defenders:
                            continue
                        f_m = metric.copy()
                        f_m[a_d] = n_players
                        f_m["period_calculator"] = period_calc
                        f_m["final_metric"] = f"{base_metric}_{period_calc}_{len_period}_sec_{n_players}_{a_d}"
                        final_metrics_list.append(f_m.copy())
    team_metrics = get_all_team_metrics()
    for base_metric in team_metrics:
        metric = pd.Series({"team_or_player": "team",
                            "base_metric": base_metric})
        for time_shift_seconds in [0, 1, 3, 5]:
            metric["time_shift(seconds)"] = time_shift_seconds
            metric["len_period"] = 0
            for a_d in ["attacking", "defending"]:
                if base_metric in no_metric_vals_available_for_attackers and a_d == "attacking":
                    continue  # these metrics are only calculated for the defending team!
                f_m = metric.copy()
                f_m["defending_or_attacking"] = a_d
                f_m["final_metric"] = f"{base_metric}_shift_{time_shift_seconds}_sec_{a_d}_team"
                final_metrics_list.append(f_m.copy())
        for len_period in [1, 3, 5]:
            metric["time_shift(seconds)"] = 0
            metric["len_period"] = len_period
            for period_calculator in ["average", "difference"]:
                for a_d in ["attacking", "defending"]:
                    if base_metric in no_metric_vals_available_for_attackers and a_d == "attacking":
                        continue  # these metrics are only calculated for the defending team!
                    f_m = metric.copy()
                    f_m["defending_or_attacking"] = a_d
                    f_m["period_calculator"] = period_calculator
                    f_m["final_metric"] = f"{base_metric}_{period_calculator}_{len_period}_sec_{a_d}_team"
                    final_metrics_list.append(f_m.copy())

    final_metrics = pd.concat(final_metrics_list, ignore_index=True, axis=1).T
    final_metrics["attacker_weights"] = None
    final_metrics["defender_weights"] = None
    return final_metrics


if __name__ == "__main__":
    # save this dataframe to see which metrics exist
    all_possible_metrics = get_config_all_possible_metrics()
    print(all_possible_metrics)
