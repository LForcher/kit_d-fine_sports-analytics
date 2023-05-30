import src.utils.tools as tools
from src import config
from src.utils import utils
import pandas as pd
import numpy as np
import scipy.signal as signal

pd.options.mode.chained_assignment = "raise"


def velocities(match_id, smoothing=True, filter_='Savitzky-Golay', window=7, polyorder=1, maxspeed=12):
    """ main method calculating velocities for given match using calc_player_velocities"""
    position_data_for_match = utils.get_position_data([match_id])
    if position_data_for_match.empty:
        print(f"Warning: No position data, thus skipping metric velocities, for match {match_id}")
        return pd.DataFrame()

    present_columns = set(config.tracking_status_column_names) - set(position_data_for_match.columns)

    if not (present_columns == set()):
        raise ValueError(f"missing column {present_columns} in {match_id}")

    positions_and_velocities_for_match = calc_player_velocities(position_data_for_match, smoothing=smoothing,
                                                                filter_=filter_, window=window,
                                                                polyorder=polyorder, maxspeed=maxspeed)

    return positions_and_velocities_for_match


def calc_player_velocities(team, smoothing=True, filter_='Savitzky-Golay', window=7, polyorder=1, maxspeed=12):
    """ calc_player_velocities( tracking_data )

    Calculate player velocities in x & y direciton, and total player speed at each timestamp of the tracking data

    Parameters
    -----------
        team: the tracking DataFrame for home or away team
        smoothing: boolean variable that determines whether velocity measures are smoothed. Default is True.
        filter_: type of filter to use when smoothing the velocities. Default is Savitzky-Golay,
            which fits a polynomial of order 'polyorder' to the data within each window
        window: smoothing window size in # of frames
        polyorder: order of the polynomial for the Savitzky-Golay filter. Default is 1 - a linear fit to the velcoity,
            so gradient is the acceleration
        maxspeed: the maximum speed that a player can realisitically achieve (in meters/second).
            Speed measures that exceed maxspeed are tagged as outliers and set to maxspeed.

    Returns
    -----------
       team : the tracking DataFrame with columns for speed in the x & y direction and total speed added
    """
    # remove any velocity data already in the dataframe
    team = remove_player_velocities(team)
    team = tools.fill_time_column(team)
    # Get the player ids
    player_ids = np.unique([c[:-2] for c in team.columns if c[:4] in ['home', 'away']])

    # Calculate the timestep from one frame to the next. Should always be 0.04 within the same half
    dt = team['Time [s]'].diff()

    # index of first frame in second half
    second_half_idx = (team.sort_values(by=['half', 'frame']).loc[team.half == 1, 'frame'].max() + 1)

    # estimate velocities for players in team
    for player in player_ids:  # cycle through players individually
        # difference player positions in timestep dt to get unsmoothed estimate of velicity
        vx = team[player + "_x"].diff() / dt
        vy = team[player + "_y"].diff() / dt

        if maxspeed > 0:
            # remove unsmoothed data points that exceed the maximum speed (these are most likely position errors)
            raw_speed = np.sqrt(vx ** 2 + vy ** 2)
            vx[raw_speed > maxspeed] = maxspeed
            vy[raw_speed > maxspeed] = maxspeed

        if smoothing:
            if filter_ == 'Savitzky-Golay':
                vx.loc[:second_half_idx] = signal.savgol_filter(vx.loc[:second_half_idx], window_length=window,
                                                                polyorder=polyorder, mode="constant")
                vy.loc[:second_half_idx] = signal.savgol_filter(vy.loc[:second_half_idx], window_length=window,
                                                                polyorder=polyorder, mode="constant")
                # calculate second half velocity
                vx.loc[second_half_idx:] = signal.savgol_filter(vx.loc[second_half_idx:], window_length=window,
                                                                polyorder=polyorder, mode="constant")
                vy.loc[second_half_idx:] = signal.savgol_filter(vy.loc[second_half_idx:], window_length=window,
                                                                polyorder=polyorder, mode="constant")
            if filter_ == 'moving average':
                ma_window = np.ones(window) / window
                # calculate first half velocity
                vx.loc[:second_half_idx] = np.convolve(vx.loc[:second_half_idx], ma_window, mode='same')
                vy.loc[:second_half_idx] = np.convolve(vy.loc[:second_half_idx], ma_window, mode='same')
                # calculate second half velocity
                vx.loc[second_half_idx:] = np.convolve(vx.loc[second_half_idx:], ma_window, mode='same')
                vy.loc[second_half_idx:] = np.convolve(vy.loc[second_half_idx:], ma_window, mode='same')

                # put player speed in x,y direction, and total speed back in the data frame
        team[player + "_vx"] = vx
        team[player + "_vy"] = vy
        team[player + "_v"] = np.sqrt(vx ** 2 + vy ** 2)

    return team


def remove_player_velocities(team):
    # remove player velocoties and acceleeration measures that are already in the 'team' dataframe
    columns = [c for c in team.columns if
               c.split('_')[-1] in ['vx', 'vy', 'ax', 'ay', 'speed', 'acceleration']]  # Get the player ids
    team = team.drop(columns=columns)
    return team
