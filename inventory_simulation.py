import pandas as pd
import csv
import numpy as np
import ipdb

orders = pd.read_csv("2018_Transport Calc Slim copy.csv",
                     usecols=[2, 3, 4, 7, 10, 14, 15, 16, 17, 21, 24, 28],
                     dtype={"sh_ItemId": "object"})

wh = pd.read_csv("wh_locations.csv")

location_dic = orders.groupby("lo_LocationId")[
    "sh_OriginLocationMasterLocation"].first()
# get mapping of io_LocationId to general location
location_dic = location_dic.to_dict()
wh_address = {}
for index, row in wh.iterrows():
    # create mapping between address and ShipToID
    wh_address[row["NavID"]] = row["Location"]


# Data cleaning
# only keep orders >0 pallets
orders = orders[orders["sh_18_Ln_PalletsEquivalent"] > 0]

orders = orders[orders["DirectShipmentInOriginal"] == "standard"]  # only keep standard shipment
# remove orders with invalid locations
orders = orders[orders["sh_OriginLocationMasterLocation"].notna()]


# if shipment was internal point to where it was sent (e.g. LASEK if ShipToID = CZ__370 00__Ceské Budejovice)
orders["internal_ship"] = orders["ShipToID"].map(wh_address)

# if shipment was to the client writes "Client"
orders["internal_ship"].fillna("Client", inplace=True)
orders["internal_ship"][orders["internal_ship"] == orders["sh_OriginLocationMasterLocation"]
                        ] = "Client"  # if orgin=destination treat it as a normal order
fact = ["BRN", "VES", "BAB", "DOM", "DOM buffer", "ITT", "ITT buffer", "DOE", "SHO", "DUN"]
orders["internal_ship"][orders["internal_ship"] == "IIT buffer"] = "ITT"


lead_times = pd.read_csv("Lead Time per GIFC.csv",
                         usecols=[0, 1, 3, 4, 6])  # load lead times spreadsheet
# remove sh prefix from GIFC
lead_times["sh_GIFC"] = lead_times["sh_GIFC"].str[4:]

lead_times = lead_times.drop_duplicates()
# lead times based on sh_GIFC and production location are assigned to the orders
orders = orders.merge(
    lead_times, on=["sh_GIFC", "ProducedBy"], how="left", sort=False)


operation = {"sh_18_Ln_PalletsEquivalent": "sum",
             "Effective Lead Time [weeks]": "first",
             "Avg delay Lead time [weeks]": "first",
             }  # assign operations to be performed on each column of interest

orders["sh_ShipmentDate"] = pd.to_datetime(
    orders["sh_ShipmentDate"],
    format="%d/%m/%Y")  # converting dates to datetime objects




test = orders.assign(period=pd.PeriodIndex(orders["sh_ShipmentDate"], freq="W-Sun")).groupby(["period",
                       "sh_ItemId",
                       "sh_OriginLocationMasterLocation",
                       "sh_GIFC",
                       "ProducedBy",
                       "internal_ship"]).agg(operation)  # aggregate based on sku,location, week,... and perform operations defined above

test = test.reset_index()

aggs = orders.assign(period=pd.PeriodIndex(orders["sh_ShipmentDate"], freq="W-Sun")).groupby(["period",
                       "sh_OriginLocationMasterLocation"]).agg({"sh_18_Ln_PalletsEquivalent": "sum"})
aggs = aggs.reset_index()
aggs.to_csv("demand_period_location.csv")
print(test[test.duplicated(subset=["sh_ItemId", "sh_OriginLocationMasterLocation", "period"], keep=False)])

prop = test[test["internal_ship"] != "Client"].groupby(
    ["sh_ItemId", "internal_ship"])["sh_18_Ln_PalletsEquivalent"].sum()  # sums quantity sent internally for each lcoations and SKU

total = test.groupby(["sh_ItemId", "sh_OriginLocationMasterLocation"])[
    "sh_18_Ln_PalletsEquivalent"].sum()  # get total demand for each location/sku
prop = prop.reset_index()
total = total.reset_index()

prop = prop.merge(total, how="left", left_on=["internal_ship", "sh_ItemId"], right_on=[
                  "sh_OriginLocationMasterLocation", "sh_ItemId"])  # match quantity sent internally and total demand


prop["perc_supplied_indirect"] = prop["sh_18_Ln_PalletsEquivalent_x"] / \
    prop["sh_18_Ln_PalletsEquivalent_y"]  # determine proportion of demand that cames from indirect flow (qty that came from other warehouse / total demand)
prop.dropna(inplace=True, subset=["perc_supplied_indirect"])
prop.to_csv("prop.csv")


prop.drop(["sh_18_Ln_PalletsEquivalent_x",
           "sh_18_Ln_PalletsEquivalent_y"], inplace=True, axis=1)

op = {"sh_18_Ln_PalletsEquivalent": ["size", "mean", "std", "sum"],
      "Effective Lead Time [weeks]": "first",
      "Avg delay Lead time [weeks]": "first"}

ss_computation = test.groupby(
    ["sh_OriginLocationMasterLocation", "sh_ItemId", "sh_GIFC"]).agg(op)  # mean demand and std (per week) is computed


ss_computation = ss_computation.reset_index()
ss_computation = ss_computation.merge(prop, how="left", right_on=["sh_ItemId", "internal_ship"],
                                      left_on=["sh_ItemId", "sh_OriginLocationMasterLocation"])  # assign prop for each order with mathcing sku and location

ss_computation.drop(
    ["internal_ship", "sh_OriginLocationMasterLocation"], axis=1, inplace=True)
ss_computation.columns = ["sh_ItemId", "sh_OriginLocationMasterLocation", "copy", "sh_GIFC",
                          "size", "mean", "std", "sum", "lt_mean", "lt_std", "perc_supplied_indirect"]  # renaming columns
ss_computation["perc_supplied_indirect"].fillna(
    0, inplace=True)  # if not internal shipment -> 0%
# fill missing lead times with median values of present values
ss_computation["lt_mean"].fillna(
    ss_computation["lt_mean"].median(), inplace=True)
ss_computation["lt_std"].fillna(
    ss_computation["lt_std"].median(), inplace=True)


''' 
compute SS by using the demand for each sku and location
weighted average is used to assign portion of safety stock that covers for demand variability (using the entire formula for SS) and SS that cover for lead time variability
+1 is added to the factory lead time to account for internal transport lead time
'''
ss_computation["Safety Stock_0.98"] = (1 - ss_computation["perc_supplied_indirect"]) * (
    2.05 * ((ss_computation["lt_mean"] + 1) * ss_computation["std"]**2 + (ss_computation["mean"]**2) * ss_computation["lt_std"]**2)**0.5) + ss_computation["perc_supplied_indirect"] * (2.05 * (1 * ss_computation["std"]**2)**0.5)

ss_computation["Safety Stock_0.98"][ss_computation["sh_OriginLocationMasterLocation"].isin(
    fact)] = 0
ss_computation["Safety Stock_0.98"][ss_computation["size"] < 2] = 0
ss_computation.to_csv("ss.csv")

test = test.merge(ss_computation, how="left", left_on=["sh_OriginLocationMasterLocation", "sh_ItemId", "sh_GIFC"], right_on=[
                  "sh_OriginLocationMasterLocation", "sh_ItemId", "sh_GIFC"])  # Assign SS levels to matching SKU and lcoation

test.drop(["copy", "mean", "std", "sum",
           "lt_mean", "lt_std"], inplace=True, axis=1)


test.rename(
    columns={"sh_18_Ln_PalletsEquivalent": "sh_18_Ln_PalletsEquivalent"}, inplace=True)
test.drop(["Effective Lead Time [weeks]",
           "Avg delay Lead time [weeks]"], axis=1, inplace=True)

holding_costs = pd.read_csv("inventory_data.csv",
                            usecols=["Item No_",
                                     "Cost per UOM",
                                     "KW 39 Quantity",
                                     "Qty per Container",
                                     "EUR pallet indicator",
                                     "Location Code",
                                     "Pallets"])

holding_costs["Qty per Container"] = holding_costs["Qty per Container"].str.strip()
holding_costs["Qty per Container"] = holding_costs["Qty per Container"].str.replace(" ", "")
holding_costs["Qty per Container"] = holding_costs["Qty per Container"].astype(float)



holding_costs["Location"] = holding_costs["Location Code"].map(location_dic)
holding_costs.dropna(inplace=True, subset=["Location Code"])

holding_costs["Location"][holding_costs["Location Code"].str.contains("TABOR")] = "DC Tabor"
holding_costs["Location"][holding_costs["Location Code"].str.contains("TÁBOR")] = "DC Tabor"
holding_costs["Location"][holding_costs["Location Code"].str.contains("SKLAD")] = "DC SK"
holding_costs["Location"][holding_costs["Location Code"].str.contains("312")] = "CoP NL"
holding_costs["Location"][holding_costs["Location Code"].str.contains("WABERE")] = "DC HU DRY"
holding_costs["Location"][holding_costs["Location Code"].str.contains("LASEK")] = "LASEK"
holding_costs["Location"][holding_costs["Location Code"].str.contains("LGI")] = "LGI"
holding_costs["Location"][holding_costs["Location Code"].str.contains("TRANSFER")] = "DC Tabor"
holding_costs["Location"][holding_costs["Location Code"].str.contains("BAL_FG_VYR")] = "DC Tabor"
holding_costs["Location"][holding_costs["Location Code"].str.contains("900")] = "DC NL"
holding_costs["Location"][holding_costs["Location Code"].str.contains("203")] = "DC NL"
holding_costs["Location"][holding_costs["Location Code"].str.contains("405")] = "ITT"


holding_costs["inv_costs_20"] = holding_costs["Cost per UOM"] * 0.2 

holding_costs = holding_costs[["Location", "Item No_", "inv_costs_20", "Location Code"]]
holding_costs = holding_costs.drop_duplicates(subset=["Item No_", "Location", "inv_costs_20"])
# holding_costs.to_csv("hol.csv")
test = test.merge(holding_costs, how="left", left_on=["sh_ItemId", "sh_OriginLocationMasterLocation"], right_on=["Item No_", "Location"])
# test = test.drop_duplicates()

test["inv_costs_20"] = test["inv_costs_20"].fillna(test.groupby("sh_ItemId")["inv_costs_20"].transform("mean"))


test = test.dropna(subset=["size"])

test.to_csv('zdzad.csv')

# test = pd.read_csv("inv.csv",
#                    index_col=[0])


def ABC_segmentation(perc):
    '''
    Creates the 3 classes A, B, and C based
    on quantity percentages (A-60%, B-25%, C-15%)
    '''
    if perc > 0 and perc < 0.6:
        return 'A'
    elif perc >= 0.6 and perc < 0.85:
        return 'B'
    elif perc >= 0.85:
        return 'C'

test_agg = test.groupby(["sh_ItemId"], as_index=False).agg(
    {"sh_18_Ln_PalletsEquivalent": "sum", "inv_costs_20": "mean"})


def ABC_segmentation(perc):
    '''
    Creates the 3 classes A, B, and C based 
    on quantity percentages (A-60%, B-25%, C-15%)
    '''
    if perc > 0 and perc < 0.6:
        return 'A'
    elif perc >= 0.6 and perc < 0.85:
        return 'B'
    elif perc >= 0.85:
        return 'C'


test_agg["AddCost"] = test_agg["sh_18_Ln_PalletsEquivalent"] 

test_agg = test_agg.sort_values(by=["AddCost"], ascending=False)
test_agg["CumCost"] = test_agg["AddCost"].cumsum()
test_agg["TotSum"] = test_agg["AddCost"].sum()
test_agg["RunPerc"] = test_agg["CumCost"] / test_agg["TotSum"]
test_agg["ABC_cluster"] = test_agg["RunPerc"].apply(ABC_segmentation)

abc_map = dict(zip(test_agg["sh_ItemId"], test_agg["ABC_cluster"]))

test["ABC_cluster"] = test["sh_ItemId"].map(abc_map)

test_agg = test.groupby(["sh_ItemId"], as_index=False)[
    "sh_18_Ln_PalletsEquivalent"].agg(["mean", "std", "count"]).reset_index()

test_agg["CV"] = test_agg["std"] / test_agg["mean"]


def XYZ_segmentation(perc):
    '''
    Creates the 3 classes A, B, and C based
    on quantity percentages (A-60%, B-25%, C-15%)
    '''
    if perc > 0 and perc <= 0.5:
        return "X"
    elif perc > 0.5 and perc < 1:
        return 'Y'
    elif perc >= 1:
        return 'Z'


test_agg["XYZ_cluster"] = test_agg["CV"].apply(XYZ_segmentation)
print(test_agg)

xyz_map = dict(zip(test_agg["sh_ItemId"], test_agg["XYZ_cluster"]))


test["XYZ_cluster"] = test["sh_ItemId"].map(xyz_map)


test["Safety Stock_0.98"][(test["ABC_cluster"] == "C") & (test["XYZ_cluster"]=="Z")] = 0



test["Total Inventory"] = test["sh_18_Ln_PalletsEquivalent"] + \
    test["Safety Stock_0.98"].fillna(
    0)  # add demand from orders with SS for weekly inventory
allperiod = test["period"].unique()
combs = []
for a, b in zip(test["sh_ItemId"], test["sh_OriginLocationMasterLocation"]):
    for t in allperiod:
        combs.append((a,b,t))

# index = pd.MultiIndex.from_product([test["sh_ItemId"].unique(), test["sh_OriginLocationMasterLocation"].unique(), test["period"].unique()])
# test = test.set_index(["sh_ItemId", "sh_OriginLocationMasterLocation", "period"])
# result = test.unstack(fill_value=0).stack().reset_index()
ss_ass = dict(zip(zip(test["sh_ItemId"], test["sh_OriginLocationMasterLocation"]), test["Safety Stock_0.98"]))
inv_cost_ass = dict(zip(zip(test["sh_ItemId"], test["sh_OriginLocationMasterLocation"]), test["inv_costs_20"]))
allskus = test["sh_ItemId"].unique()

abc_clust = dict(zip(test["sh_ItemId"], test["ABC_cluster"]))
xyz_clust = dict(zip(test["sh_ItemId"], test["XYZ_cluster"]))
perc = dict(zip(zip(test["sh_ItemId"], test["sh_OriginLocationMasterLocation"]), test["perc_supplied_indirect"]))
dic_df = test.set_index(["sh_ItemId", "sh_OriginLocationMasterLocation", "period"]).to_dict()

for c in combs:
    if c not in dic_df["sh_18_Ln_PalletsEquivalent"]:
        dic_df["sh_18_Ln_PalletsEquivalent"][c] = 0
        dic_df["Safety Stock_0.98"][c] = ss_ass[c[0:2]]
        dic_df["Total Inventory"][c] = ss_ass[c[0:2]]
        dic_df["inv_costs_20"][c] = inv_cost_ass[c[0:2]]
        dic_df["perc_supplied_indirect"][c] = perc[c[0:2]]
        dic_df["ABC_cluster"][c] = abc_clust[c[0]]
        dic_df["XYZ_cluster"][c] = xyz_clust[c[0]]

test = pd.DataFrame.from_dict(dic_df)
test = test.reset_index()
# test.rename(columns={"level_0": "sh_ItemId", "level_1": "sh_OriginLocationMasterLocation", "level_2": "period"}, inplace=True)
# test.sort_values(by=["sh_ItemId", "sh_OriginLocationMasterLocation"], inplace=True, axis=1)
print(test)
test.to_csv("inventory final.csv")



batch = pd.read_csv("PPF Batch Size.csv",
                    index_col=False)
print(batch)

batch = dict(zip(batch["sh_ItemId"], batch["Min. Batch Size (PAL)"]))

def min_batch(qty, sku):

    if qty > 0 and qty < batch[sku]:
        return batch[sku]
    else:
        return qty


test["sh_18_Ln_PalletsEquivalent"][test["level_1"] == test["ProducedBy"]] = np.vectorize(min_batch)(test["sh_18_Ln_PalletsEquivalent"], test["level_0"])
test["Total Inventory"] = test["sh_18_Ln_PalletsEquivalent"] + test["Safety Stock_0.98"].fillna(0)
a = np.vectorize(min_batch)(test["sh_18_Ln_PalletsEquivalent"], test["level_0"])

test["inv_costs_20"][test["Total Inventory"] == 0] = 0

ipdb.set_trace()

print("bla")