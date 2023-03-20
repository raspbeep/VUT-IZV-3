#!/usr/bin/python3.10
# coding=utf-8
import pandas as pd
import geopandas
import matplotlib.pyplot as plt
import contextily as ctx
import numpy as np
from sklearn.cluster import AgglomerativeClustering


def make_geo(df: pd.DataFrame) -> geopandas.GeoDataFrame:
    """
    Create GeoDataFrame with S-JTSK projected coordinate system from DataFrame df
    :param df: input data frame
    :return: GeoDataFrame
    """
    df["date"] = pd.to_datetime(df["p2a"], cache=True)
    category_columns = ["k", "p", "q", "t", "l", "i", "h"]
    df[category_columns] = df[category_columns].astype("category")
    df = df[(df["d"].notna()) & (df["e"].notna())]
    return geopandas.GeoDataFrame(
        df, geometry=geopandas.points_from_xy(df["d"], df["e"]), crs="EPSG:5514"
    )


def plot_geo(
    gdf: geopandas.GeoDataFrame, fig_location: str = None, show_figure: bool = False
):
    """
    Plot 4 subplots for each year with alcohol and drugs related road accidents in selected regions
    :param gdf: GeoDataFrame with accidents
    :param fig_location: location of the figure
    :param show_figure: show figure
    :return: GeoDataFrame
    """
    selected_region = "JHM"
    years = [2018, 2019, 2020, 2021]
    new_gdf = gdf[(gdf["region"] == selected_region) & gdf["date"].dt.year.isin(years)]
    new_gdf = new_gdf.to_crs("EPSG:3857")

    fig, axs = plt.subplots(2, 2, figsize=(12, 10.5))

    alcohol = [3, 6, 7, 8, 9]
    drugs = [4, 5]

    for ax, year in zip(axs.flatten(), years):
        current_gdf = new_gdf[
            (new_gdf["date"].dt.year == year)
            & (new_gdf["p11"].isin(drugs) | new_gdf["p11"].isin(alcohol))
        ]
        ax.set_xlim(xmin=current_gdf.total_bounds[0], xmax=current_gdf.total_bounds[2])
        ax.set_ylim(ymin=current_gdf.total_bounds[1], ymax=current_gdf.total_bounds[3])
        ax.set_title(
            "{region} ({year})".format(region=selected_region, year=year), fontsize=14
        )
        ax.set_aspect("equal")
        ax.axis("off")

        # create markers at accident locations, color is differentiated by p11(alcohol or drugs)
        current_gdf[(current_gdf["p11"].isin(drugs))].plot(
            ax=ax, color="blue", markersize=1
        )
        current_gdf[(current_gdf["p11"].isin(alcohol))].plot(
            ax=ax, color="red", markersize=1
        )
        ctx.add_basemap(
            ax,
            crs=new_gdf.crs.to_string(),
            source=ctx.providers.Stamen.TonerLite,
            attribution_size=3,
            reset_extent=False,
            alpha=0.8,
        )
        ax.legend(
            ["Drogy", "Alkohol"],
            fancybox=True,
            framealpha=1,
            shadow=True,
            markerscale=5,
        )

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.tight_layout()

    if fig_location is not None:
        plt.savefig(fig_location)
    if show_figure:
        plt.show()
    plt.close()


def plot_cluster(
    gdf: geopandas.GeoDataFrame, fig_location: str = None, show_figure: bool = False
):
    selected_region = "JHM"
    new_gdf = gdf[(gdf["region"] == selected_region)]
    # make clustering base on column p36
    new_gdf = new_gdf[new_gdf["p36"].notna()]
    # 1., 2. and 3. class of roads
    roads = [1, 2, 3]
    new_gdf = new_gdf[new_gdf["p36"].isin(roads)]
    # convert to straight CRS
    new_gdf = new_gdf.to_crs("EPSG:3857")
    # reshape the data to unknown number of pairs ([lat, long], ...)
    coordinates = np.reshape(list(zip(new_gdf["d"], new_gdf["e"])), (-1, 2))
    # Agglomerative clustering was chosen due to expected uneven cluster sizes,
    # suitability for point distance clustering and connectivity constraints caused
    # by the road network and outliers
    new_gdf["road_type"] = (
        AgglomerativeClustering(n_clusters=20).fit(coordinates).labels_
    )
    new_gdf = new_gdf.dissolve(by="road_type", aggfunc={"p1": "count"})

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_axes([0.0, 0.12, 1.0, 0.83])

    new_gdf.plot(ax=ax, markersize=1, column="p1", legend=False, cmap="plasma")
    ax.set_xlim(xmin=new_gdf.total_bounds[0], xmax=new_gdf.total_bounds[2])
    ax.set_ylim(ymin=new_gdf.total_bounds[1], ymax=new_gdf.total_bounds[3])
    ax.axis("off")
    ctx.add_basemap(
        ax,
        crs=new_gdf.crs.to_string(),
        alpha=0.9,
        reset_extent=False,
        source=ctx.providers.Stamen.TonerLite,
    )

    ax.set_title(
        "Pocet nehod v " + selected_region + " na cestach 1., 2. a 3. triedy",
        fontsize=16,
    )

    cb = fig.add_axes([0.015, 0.03, 0.97, 0.5])
    cb.axis("off")
    fig.colorbar(
        ax=cb, mappable=ax.collections[0], orientation="horizontal", label="Pocet nehod"
    )

    if fig_location is not None:
        plt.savefig(fig_location)
    if show_figure:
        plt.show()
    plt.close()


if __name__ == "__main__":
    geo_dataframe = make_geo(pd.read_pickle("accidents.pkl.gz"))
    plot_geo(geo_dataframe, "geo1.png", True)
    plot_cluster(geo_dataframe, "geo2.png", True)
