import pandas as pd
import matplotlib.pyplot as plt


def load_split(inventory, ss):

    li = []
    for i in ss:
        li.append(i)
    ss_df = pd.concat(li, axis=0)
    return inventory, ss_df

def agg_data_split(inventory, ss):
    inventory, ss_df = load_split(inventory, ss)
    inventory_agg = inventory.groupby("time", as_index=True).sum()
    time = inventory["time"].unique()

    ss = ss_df["Safety_Stock"].sum()
    for t in time:
        inventory_agg.loc[t, "ss"] = ss
    inventory_agg.columns = ["Cycle Inventory", "Safety Stock"]
    inventory_agg["Total Inventory"] = inventory_agg["Cycle Inventory"] + inventory_agg["Safety Stock"]
    inventory_agg["Relative Cycle Inventory"] = inventory_agg["Cycle Inventory"] / inventory_agg["Total Inventory"]
    inventory_agg["Relative Safety Stock"] = inventory_agg["Safety Stock"] / inventory_agg["Total Inventory"]
    return inventory_agg

def cycle_ss_barplot_abs(inventory, ss):
    data = agg_data_split(inventory, ss)
    ax = data[["Cycle Inventory", "Safety Stock"]].plot(kind="bar", stacked=True)
    ax.set(ylabel="Pallets", title="Cycle Inventory vs. Safety Stock (Absolute)")
    xl = ax.set_xlabel("Time Period", fontsize=9)
    plt.xticks(rotation=45)
    plt.subplots_adjust(bottom=0.15)
    plt.show()

def cycle_ss_barplot_rel(inventory, ss):
    data = agg_data_split(inventory, ss)

    ax = data[["Relative Cycle Inventory", "Relative Safety Stock"]].plot(kind="bar", stacked=True)
    ax.set(ylabel="Percent", title="Cycle Inventory vs. Safety Stock (Relative)")
    xl = ax.set_xlabel("Time Period", fontsize=9)
    plt.xticks(rotation=45)
    plt.subplots_adjust(bottom=0.15)
    plt.show()

def find_intersection(inventory, baseline_path):
    baseline = pd.read_csv(baseline_path,
                           index_col=False)
    print(baseline)
    print(baseline.columns)
    idx_inv = inventory.set_index(["product", "time"]).index
    idx_bas = baseline.set_index(["sh_ItemId", "period"]).index
    idx_its = idx_inv.intersection(idx_bas)
    print(idx_its)
    return idx_its, baseline