import geopandas as gpd
import networkx as nx
import os
import copy
import numpy as np
import pandas as pd
from pysal.lib.weights import Queen
from matplotlib.colors import LinearSegmentedColormap as LSC
import matplotlib.pyplot as plt
import seaborn as sns


def color_map(gdf, districting):
    # Takes a few seconds

    # block : distr num
    inv_map = {block: k for k, district in districting.items()
               for block in district}

    gdf['district'] = pd.Series(inv_map)
    shapes = []
    for name, group in gdf.groupby('district'):
        shapes.append(group.geometry.unary_union)
    shape_series = gpd.GeoSeries(shapes)
    G = Queen(shapes).to_networkx()
    color_series = pd.Series(nx.greedy_color(G))
    n_colors = len(set(color_series.values))
    cmap = LSC.from_list("", ["red", "green", "dodgerblue",
                              'yellow', 'darkviolet', 'chocolate'][:n_colors])

    map_gdf = gpd.GeoDataFrame({'geometry': shape_series,
                                'color': color_series})
    ax = map_gdf.plot(column='color', figsize=(15, 15), cmap=cmap, edgecolor='black', lw=.5)
    gdf.plot(ax=ax, facecolor='none', edgecolor='white', lw=.05)
    ax.axis('off')
    return map_gdf, ax


def politics_map(gdf, politics, districting):
    # Takes a few seconds

    # block : distr num
    inv_map = {block: k for k, district in districting.items()
               for block in district}

    gdf['district'] = pd.Series(inv_map)

    shapes = []
    colors = []
    for name, group in gdf.groupby('district'):
        shapes.append(group.geometry.unary_union)
        colors.append(politics[name])
    shape_series = gpd.GeoSeries(shapes)

    map_gdf = gpd.GeoDataFrame({'geometry': shape_series,
                                'color': pd.Series(colors)})
    ax = map_gdf.plot(column='color', figsize=(15, 15), edgecolor='black', lw=1.5,
                      cmap='seismic', vmin=.35, vmax=.65)
    gdf.plot(ax=ax, facecolor='none', edgecolor='white', lw=.05)
    ax.axis('off')
    return map_gdf, ax

def mmd_politics_map(gdf, politics, districting):
    # Takes a few seconds

    # block : distr num
    inv_map = {block: k for k, district in districting.items()
               for block in district}

    gdf['district'] = pd.Series(inv_map)

    shapes = []
    colors = []
    for name, group in gdf.groupby('district'):
        shapes.append(group.geometry.unary_union)
        colors.append(politics[name])
    shape_series = gpd.GeoSeries(shapes)

    map_gdf = gpd.GeoDataFrame({'geometry': shape_series,
                                'color': pd.Series(colors)})
    ax = map_gdf.plot(column='color', figsize=(15, 15), edgecolor='black', lw=1.5,
                      cmap='seismic', vmin=0, vmax=1)
    gdf.plot(ax=ax, facecolor='none', edgecolor='white', lw=.05)
    ax.axis('off')
    return map_gdf, ax


def create_mmd_panel_input_data(results, solution_indices):
    plans = []
    plan_outcomes = []
    for trial, sol_ix in solution_indices:
        scores = results[trial]['district_scores'].thiele_pav
        seats = results[trial]['district_df'].n_seats

        if isinstance(sol_ix, list):
            plan_node_ixs = seats.index.get_indexer(sol_ix)
        else:
            solutions = results[trial]['optimization_results']['fair']['thiele_pav']
            plan_node_ixs = solutions[sol_ix]['solution_ixs']

        seat_series = scores.iloc[plan_node_ixs] 
        politics_series = seat_series / seats.iloc[plan_node_ixs]
        plan = {nid: results[trial]['block_assignment'][nid] for nid in seat_series.index}
        plans.append(plan)
        plan_outcomes.append(politics_series.to_dict())
    return plans, plan_outcomes


def plot_mmd_panel(shapes, plans, plan_outcomes, label_loc=(0.1, 0.1), figsize=(15, 6)):
    gdf = copy.deepcopy(shapes)
    n_maps = len(plans)
    fig, axs = plt.subplots(1, n_maps, figsize=figsize)
    for i in range(n_maps):
        districting = plans[i]
        politics = plan_outcomes[i]
        n_districts = len(politics)
        ax = axs[i]
        inv_map = {block: k for k, district in districting.items()
                for block in district}

        gdf['district'] = pd.Series(inv_map)

        d_shapes = []
        colors = []
        for name, group in gdf.groupby('district'):
            d_shapes.append(group.geometry.unary_union)
            colors.append(politics[name])

        shape_series = gpd.GeoSeries(d_shapes)
        map_gdf = gpd.GeoDataFrame({'geometry': shape_series,
                                    'color': pd.Series(colors)})
        map_gdf.plot(ax=ax, column='color', edgecolor='black', lw=1,
                        cmap='seismic', vmin=0, vmax=1)
        gdf.plot(ax=ax, facecolor='none', edgecolor='white', lw=.025)
        ax.axis('off')
        ax.text(label_loc[0], label_loc[1], f'K={n_districts}', size=24,
                transform=ax.transAxes)

    fig.tight_layout(w_pad=0)
    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.9, 0.24, 0.03, 0.55])
    sm = plt.cm.ScalarMappable(cmap='seismic', norm=plt.Normalize(vmin=0, vmax=1))
    # fake up the array of the scalar mappable. Urgh...
    sm._A = []
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.ax.set_ylabel('expected Republican seat share', rotation=270, fontsize=14)
    cbar.ax.get_yaxis().labelpad = 18
    cbar.ax.tick_params(labelsize=14) 
    return fig