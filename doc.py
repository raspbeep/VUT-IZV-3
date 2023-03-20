#!/usr/bin/python3.10
# coding=utf-8
import re
import pandas as pd
import matplotlib.pyplot as plt

colors = ["#99ff99", "#ffcc99", "#ff9999", "#66b3ff", "#ffccff"]


def parse_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Changes dataframe column p45a (car brand) to a category type, add date column
    :param df: input data frame
    :return df: pd.DataFrame
    """
    df["date"] = pd.to_datetime(df["p2a"], cache=True)
    df["p45a"] = df["p45a"].astype("category")
    return df


def prevalent_brands(dataframe: pd.DataFrame, n: int, print_data: bool = False):
    """
    Get top brands which are involved in accidents the most. Count their accidents and return it in a list of tuples
    :param print_data: whether to print data
    :param dataframe: input data frame
    :param n: number of top brands
    :return: tuples of top accidents involved brands and their counts
    """
    df = dataframe.copy()
    # remove faulty values
    new_df = df[(df["p45a"] != -1) & (df["p45a"].notna()) & (df["p45a"] != 0)]
    # get most prevalent brands in accidents
    top_b_numbers = new_df["p45a"].value_counts().iloc[:n].index.tolist()
    new_df = new_df[new_df["p45a"].isin(top_b_numbers)]
    top_brands_tuples = []
    b_dictionary = {}
    # open input file with brands and numbers
    with open("brands", "r") as f:
        brands = f.readlines()
        for line in brands:
            number_brand = line.strip().split("\t")
            b_dictionary[int(number_brand[0])] = number_brand[1].lower().capitalize()
    # create tuples
    for b in top_b_numbers:
        top_brands_tuples.append(
            (b_dictionary[b], b, new_df[new_df["p45a"] == b]["p45a"].count())
        )

    brands_dict = {brand[1]: brand[0] for brand in top_brands_tuples}
    # keep only most prevalent brands
    df = df[df["p45a"].isin(top_b_numbers)]

    # create a category new column with brand names
    df["Brand"] = df["p45a"].map(brands_dict)
    df["Brand"] = df["Brand"].astype("category")

    if print_data:
        for i in top_brands_tuples:
            print(f"number: {i[1]} \t n. of accidents: {i[2]} \t brand: {i[0]}")
    return top_brands_tuples, df


def plot_brands(
    dataframe: pd.DataFrame, fig_location: str = None, show_figure: bool = False
):
    """
    Plot bar chart of the most prevalent brands and their accidents
    :param dataframe: input data frame
    :param fig_location: location to save the figure
    :param show_figure: whether to show the figure
    """
    plt.style.context("seaborn-muted")
    df = dataframe.copy()

    # aggregate accidents by region
    df = df.groupby(["region", "Brand"]).size().reset_index(name="counts")
    # plot bar chart by each region and brand

    df.pivot(index="region", columns="Brand", values="counts").plot(
        kind="bar", figsize=(10, 6), alpha=0.75, rot=0, stacked=True, color=colors
    )
    plt.xlabel("Region")
    plt.ylabel("Number of accidents")
    plt.title("Number of accident by region and vehicle brand")

    plt.tight_layout()
    if fig_location:
        plt.savefig(fig_location)
    if show_figure:
        plt.show()
    plt.close()


def plot_pies(
    dataframe: pd.DataFrame, fig_location: str = None, show_figure: bool = False
):
    """
    Plot pie charts of the most prevalent brands and their accidents
    :param dataframe: input data frame
    :param fig_location: location to save the figure
    :param show_figure: whether to show the figure
    :return: None
    """
    plt.style.context("seaborn-muted")
    df = dataframe.copy()

    def cat(val):
        """
        Function to get the category of the value
        :param val:
        :return: String describing the category
        """
        if val == 100:
            return "Not cause by the driver"
        if 201 <= val <= 209:
            return "Excessive driving speed"
        if 301 <= val <= 311:
            return "Incorrect overtaking"
        if 401 <= val <= 414:
            return "Not giving right of way"
        if 501 <= val <= 516:
            return "Improper driving style"
        if 601 <= val <= 615:
            return "Technical failure"
        else:
            # fail to categorize
            return None

    # create a category new column with brand names
    df["Cause"] = df["p12"].apply(cat)
    # remove faulty values
    df = df[df["Cause"].notna()]
    df["Cause"] = df["Cause"].astype("category")

    df = df.groupby(["Brand", "Cause"]).size().reset_index(name="counts")
    labels = [
        "Not cause by the driver",
        "Excessive driving speed",
        "Incorrect overtaking",
        "Not giving right of way",
        "Improper driving style",
        "Technical failure",
    ]
    # create pie charts subplots for each brand and the causes of accidents
    ax = df.pivot(index="Cause", columns="Brand", values="counts").plot(
        kind="pie",
        autopct="%1.0f%%",
        subplots=True,
        figsize=(16, 6),
        shadow=True,
        legend=False,
        labels=None,
        colors=colors,
    )
    # iterate in subplots and set labels
    for i, a in enumerate(ax):
        a.set_title(a.yaxis.label.get_text())
        a.set_ylabel("")

    plt.tight_layout()
    plt.legend(labels=labels, loc="lower center", bbox_to_anchor=(-1.10, -0.19), ncol=3)

    if fig_location:
        plt.savefig(fig_location)
    if show_figure:
        plt.show()
    plt.close()


def make_table(dataframe: pd.DataFrame, brands: list, print_table: bool = False):
    """
    Make a tex table of the most prevalent brands and their accident counts by year
    :param dataframe: input data frame
    :param brands: list of tuples of brands to print
    :param print_table: whether to print the table
    """
    df = dataframe.copy()
    # get accidents for selected brands numbers
    top_brand_numbers = [brand[1] for brand in brands]
    brands_dict = {brand[1]: brand[0] for brand in brands}

    # keep only most prevalent brands
    df = df[df["p45a"].isin(top_brand_numbers)]

    # create a category new column with brand names
    df["Brand"] = df["p45a"].map(brands_dict)
    df["Brand"] = df["Brand"].astype("category")

    df["year"] = df["date"].dt.year
    df["sum"] = 1
    df = df.groupby(["Brand", "year"]).agg({"sum": "sum"}).reset_index()
    df = df.pivot(index="Brand", columns="year", values="sum")

    tex = df.style.to_latex(
        caption="Top vehicle brands involved in accidents by year",
        position="h",
        hrules=True,
    )

    # remove index name and add centering
    tex = re.sub(r"^Brand\b.*\n", "", tex, flags=re.MULTILINE)
    tex = re.sub(
        r"^\\begin\{table\}\[h\].*\n",
        "\\\\begin{table}[h]\n\\\\centering\n",
        tex,
        flags=re.MULTILINE,
    )

    if print_table:
        print("\n% BEGIN TABLE\n")
        print(tex)
        print("\n% END TABLE\n")


if __name__ == "__main__":
    data_frame = parse_data(pd.read_pickle("accidents.pkl.gz"))
    p_brands, data_frame = prevalent_brands(data_frame, n=4, print_data=True)
    plot_brands(data_frame, fig_location="brands_per_region.png", show_figure=True)
    plot_pies(data_frame, fig_location="pies.png", show_figure=True)
    make_table(data_frame, p_brands, print_table=True)
