import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
matplotlib.use('nbagg')
import src.utils.tools as tools


def plot_positions(in_df, half, frames, mark_player='none', dimensions=(60, 45)):
    half_length = dimensions[0]
    half_width = dimensions[1]
    ratio = half_width / half_length
    plt.figure(figsize=(14, 14 * ratio))
    plt.title(mark_player)
    fig = plt.gcf()
    ax = fig.gca()
    plt.xlim([-half_length, half_length])
    plt.ylim([-half_width, half_width])
    circle1 = plt.Circle((0, 0), 9.15, color='b', fill=False)

    edge1 = plt.Circle((-half_length, -half_width), 1, color='b', fill=False)
    edge2 = plt.Circle((-half_length, half_width), 1, color='b', fill=False)
    edge3 = plt.Circle((half_length, half_width), 1, color='b', fill=False)
    edge4 = plt.Circle((half_length, -half_width), 1, color='b', fill=False)

    penalty1 = plt.Rectangle((-half_length, -20.16), 16.5, 40.32, color='b', fill=False, clip_on=False)
    penalty2 = plt.Rectangle((half_length - 16.5, -20.16), 16.5, 40.32, color='b', fill=False, clip_on=False)

    goal_area_1 = plt.Rectangle((-half_length, -9.16), 5.5, 18.32, color='b', fill=False, clip_on=False)
    goal_area_2 = plt.Rectangle((half_length - 5.5, -9.16), 5.5, 18.32, color='b', fill=False, clip_on=False)

    goal1 = plt.Rectangle((-half_length, -3.66), -3.66, 7.32, color='b', fill=False, clip_on=False)
    goal2 = plt.Rectangle((half_length + 3.66, -3.66), -3.66, 7.32, color='b', fill=False, clip_on=False)

    plt.axvline(0, color='b')
    plt.scatter(0, 0, color='b')
    plt.scatter(-half_length + 11, 0, color='b')
    plt.scatter(half_length - 11, 0, color='b')

    ax.add_patch(circle1)
    ax.add_patch(penalty1)
    ax.add_patch(goal_area_1)
    ax.add_patch(goal1)
    ax.add_patch(penalty2)
    ax.add_patch(goal_area_2)
    ax.add_patch(goal2)
    ax.add_patch(edge1)
    ax.add_patch(edge2)
    ax.add_patch(edge3)
    ax.add_patch(edge4)



    in_df = in_df.loc[(in_df.half == half) & (in_df.frame.isin(frames)), :]
    player_columns = tools.get_player_id_column_names(in_df)

    if mark_player == 'auto':
        try:
            mark_player = in_df.pID.unique()[0]
        except:
            mark_player = 'none'
    else:
        mark_player = in_df[mark_player].unique()[0]

    for frame in frames:
        df_by_frames = in_df.loc[in_df.frame == frame, :]
        players = tools.get_player_ids(df_by_frames)
        for player in players:
            value_x = df_by_frames[[player + '_x']]
            value_y = df_by_frames[[player + '_y']]
            if player != 'ball':
                plt.scatter(value_x, value_y, marker='o', color='green', alpha=0.3)
                if mark_player != 'none':
                    if player == mark_player:
                        plt.scatter(value_x, value_y, marker='o', color='red', label=mark_player)
            else:
                plt.scatter(value_x, value_y, marker='o', color='orange', label='ball')

    plt.xlabel("distane /m", fontsize=20)
    plt.ylabel("distane /m", fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    return plt
