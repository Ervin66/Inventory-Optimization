import pandas as pd
import numpy as np
from pulp import *
import ipdb
import time
import os
import visualisation as v
import ujson
import collections
import re
import configparser, os.path, argparse
# import gurobipy
# import cProfile
# import pstats
# import io


class InventoryModel():
    """class used to define and construct the inventory linear program
       data_path: the file path of the data containing the orders
       product_col: string of the column's header containing the product (group) id
       time_col: string of the column's header containing the date of the order
       loc_col: string of the column's header containing the location of the order
       qty_col: string of the column's header containing the quantity of the order
    """

    def __init__(self, sku_col, time_col, loc_col, qty_col, prod_col):
        self.biobj = self.arg_parse()
        self.config_dict = self.read_config(overide="config.ini")
        self.sku_col = sku_col
        self.time_col = time_col
        self.loc_col = loc_col
        self.qty_col = qty_col
        self.prod_col = prod_col
        d = {sku_col: "object"}
        self.raw_data = pd.read_csv(self.config_dict["filenames"]["orders"],
                                    index_col=[0],
                                    nrows=100,
                                    dtype=d)

        self.service_level = float(self.config_dict["model"]["service_level"])
        self.ftl_matrix_path = self.config_dict["filenames"]["ftl_matrix"]
        self.batch_size_path = self.config_dict["filenames"]["batch_size"]
        self.loc_data_path = self.config_dict["filenames"]["location_data"]

        # column's values are converted to datetime_col objects to facilitate weekly aggregation
        # self.raw_data[time_col] = pd.to_datetime(self.raw_data[time_col])

        # remove orders that do not have a location specified
        self.data = self.raw_data[self.raw_data[loc_col].notna()]
        # self.data = self.data["XYZ_cluster"].fillna("Z")

        # self.data = self.data[self.data["sh_ItemId"] == "2C700000000173"]
        self.writer = pd.ExcelWriter(self.config_dict["filenames"]["result_file"], engine="xlsxwriter")
        if not os.path.exists("CSV export files"):
            os.makedirs("CSV export files")
        if not os.path.exists("Saved Models"):
            os.makedirs("Saved Models")

        # privremeno
        # v = self.raw_data[sku_col].value_counts()
        # self.raw_data = self.raw_data[self.raw_data[sku_col].isin(
        #     v.index[v.gt(100)])]
        self.data["XYZ_cluster"] = self.data["XYZ_cluster"].fillna("Z")
        self.data = self.data[~((self.data["ABC_cluster"] == "C") & (self.data["XYZ_cluster"] == "Z"))]
        # self.data = self.data.groupby(["sh_ItemId", "new_shOrigin"]).filter(lambda x: len(x) > 5)
        self.data.to_excel(self.writer, sheet_name="Demand")
        self.data.to_csv("./CSV export files/Demand.csv")
        print(self.data)
        print(self.data.groupby(sku_col)[
              self.qty_col].agg(Mean='mean', Sum='sum', Size="size"))


        # self.data.to_csv("demand_input.csv")
        self.inv_model = pulp.LpProblem("Inventory_Optimization",
                                        LpMinimize)  # creates an object of an LpProblem (PuLP) to which we will assign variables, constraints and objective function
    
    def arg_parse(self):
        parser = argparse.ArgumentParser(description= "Launch a bi-objective optimisation")
        parser.add_argument("--biobj", type=bool)
        args = parser.parse_args()
        return args.biobj

    def read_config(self, overide=None):
        config = configparser.ConfigParser()
        while True:
            if overide is None:
                config_file = input("Please specifiy configuration file ")
                if os.path.exists(config_file):
                    config.read(config_file)
                    config_dict = {}
                    break
                else:
                    print("The configuration file could not be found. Please try again. ")
                    continue
            else:
                config.read(overide)
                config_dict = {}
                break
        for section in config.sections():
            config_dict[section] = dict(config.items(section))
        for k in config_dict:
            for key, value in config_dict[k].items():
                if value == "True":
                    config_dict[k][key] = True
                if value == "False":
                    config_dict[k][key] = False
        return config_dict

    def recycle_model(self, folder):
        while True:
            model_file = input("Please indicate model's name ")
            try:
                variables, model = LpProblem.from_json(folder + model_file + ".json")
                with open(folder + model_file + "constraints.json") as f:
                    cons = ujson.load(f)
                break
            except FileNotFoundError:
                print("The model name cannot be found. Please try again.")
                continue
        model.constraints = OrderedDict(zip(cons.keys(), model.constraints.values()))  # rename the constraint for easier altering
        self.update_constraint_rhs(model.constraints)
        self.call_cplex(model)
        results = self.parse_variables(variables)
        print("Inventory level:--------------")
        self.inv_result = self.export_vars_3d(indices=results["inventory"][0],
                                              variable=results["inventory"][1],
                                              filename="Inventory Level")
        print("Shipments:------------------")
        self.ship_result, self.ftl_result = self.export_vars_4d(indices=results["shipment"][0],
                                                                variable=results["shipment"][1],
                                                                filename="Shipment")
        print("Production:-----------------")
        self.prod_result = self.export_vars_3d(indices=results["production"][0],
                                               variable=results["production"][1],
                                               filename="Production")
        print("Lost Sales:-----------------")
        self.ls_result = self.export_vars_3d(indices=results["lost_sales"][0],
                                               variable=results["lost_sales"][1],
                                             filename="Lost Sales")
        self.writer.save() 

    def parse_var_keys(self, var_key):
        # a = [tuple(re.findall(r"'(\w+)'", text)) for text in variables.keys()]
        a = tuple(re.findall(r"'(\w+)'", var_key))
        return a

    def parse_variables(self, variables):
        inv_dic, inv_idx = {}, []
        ship_dic, ship_idx = {}, []
        prod_dic, prod_idx = {}, []
        ls_dic, ls_idx = {}, []

        for k,v in variables.items():
            if "inventory" in k:
                i = self.parse_var_keys(k)
                inv_idx.append(i)
                inv_dic[i] = v
            if "shipment" in k:
                i = self.parse_var_keys(k)
                ship_idx.append(i)
                ship_dic[i] = v
            if "production" in k:
                i = self.parse_var_keys(k)
                prod_idx.append(i)
                prod_dic[i] = v
            if "lost_sales" in k:
                i = self.parse_var_keys(k)
                ls_idx.append(i)
                ls_dic[i] = v
        master_dic = {"inventory": [inv_idx, inv_dic],
                      "shipment": [ship_idx, ship_dic],
                      "production": [prod_idx, prod_dic],
                      "lost_sales": [ls_idx, ls_dic]}
        return master_dic

    def update_constraint_rhs(self, constraints):
        cons_modified = {"Production Cap.": 0,
                         "Holding Cap.": 0,
                         "Initial Inv.": 0,
                         "Min. Inv.": 0} 
        for fact in self.factory_id:
            if constraints.get(f"('{fact}', '0')ProdCap", False) is not False:
                if constraints[f"('{fact}', '0')ProdCap"].constant !=  - self.loc_da["Prod. Cap."][fact]: #  check if prod. cap. has changed
                    new_cap = - self.loc_da["Prod. Cap."][fact]
                    for t in self.time_id:
                        constraints[f"('{fact}', '{t}')ProdCap"].constant = new_cap
                        cons_modified["Production Cap."] += 1
        for wh in self.loc_id:
            if constraints.get(f"('{wh}', '0')HoldCap", False) is not False:
                if constraints[f"('{wh}', '0')HoldCap"].constant !=  - self.loc_da["Hold. Cap."][wh]:
                    new_cap = - self.loc_da["Hold. Cap."][wh]
                    for t in self.time_id:
                        constraints[f"('{wh}', '{t}')HoldCap"].constant = new_cap
                        cons_modified["Holding Cap."] += 1


        p = [k for k, v in self.cw_ss.items() if v > 0]
        t_ = self.time_id[1]
        for d, i in p:
            if constraints[f"('{d}', '{t_}', '{i}')2ndech_ssreq"].constant != - self.cw_ss[(d, i)]:
                new_ss = - self.cw_ss[(d, i)]
                for t in self.time_id[1:]:
                    constraints[f"('{d}', '{t}', '{i}')2ndech_ssreq"].constant = new_ss
                    cons_modified["Min. Inv."] += 1

                old_initial = constraints[f"('{d}', '0', '{i}')initial"].constant
                constraints[f"('{d}', '0', '{i}')initial"].constant -= (old_initial - new_ss)
                cons_modified["Initial Inv."] += 1


        p = [k for k, v in self.rw_ss.items() if v > 0]
        for w, i in p:
            if constraints[f"('{w}', '{t_}', '{i}')3rdech_ssreq"].constant != - self.rw_ss[(w, i)]:
                new_ss = - self.rw_ss[(w, i)]
                for t in self.time_id[1:]:
                    constraints[f"('{w}', '{t}', '{i}')3rdech_ssreq"].constant = new_ss
                    cons_modified["Min. Inv."] += 1

                old_initial = constraints[f"('{w}', '0', '{i}')initial"].constant
                constraints[f"('{w}', '0', '{i}')initial"].constant -= (old_initial - new_ss)
                cons_modified["Initial Inv."] += 1
        if all(value == 0 for value in cons_modified.values()):
            print("No constraint has been modified")
        else:
            print("Production Capacity Cons. modfied: " + str(cons_modified["Production Cap."]))
            print("Holding Capacity Cons. modfied: " + str(cons_modified["Holding Cap."]))
            print("Initial Inv. modfied: " + str(cons_modified["Initial Inv."]))
            print("Min. Inv. Cons. modfied: " + str(cons_modified["Min. Inv."]))

    def define_indices(self, loc_data_path):
        ''' Returns the unique values for indices from the input data
        Required keywords argument:
        new_time_col: string of the aggregation basis for time (e.g. "week", "month",...)
        '''
        self.loc_data = pd.read_csv(loc_data_path,
                            index_col=[0])
        self.sku_id = self.data[self.sku_col].unique()
        self.time_id = self.data["period"].unique().tolist()
        self.time_id.insert(0,"0")  # create t=0 index
        self.loc_id = self.loc_data.index.tolist()
        self.factory_id = self.loc_data[self.loc_data["Echelon"]
                                        == "Factory"].index.tolist()
        self.cw_id = self.loc_data[self.loc_data["Echelon"]
                                   == "Central"].index.tolist()
        self.rw_id = self.loc_data[self.loc_data["Echelon"]
                                   == "Regional"].index.tolist()
        self.ext_factory_id = list(set(self.data[self.prod_col].unique().tolist()) - set(self.factory_id))

    def define_subsets(self):
        # subset of skus produced internally
        self.int_skus = (self.data[self.data[self.prod_col]
                               .isin(self.factory_id)][self.sku_col]
                               .to_list())

        # subset of external skus
        self.ext_skus = (self.data[self.data[self.prod_col]
                               .isin(self.ext_factory_id)][self.sku_col]
                               .to_list())

        # assign which factory produces which sku
        k = (self.data[self.data[self.prod_col]
             .isin(self.factory_id)]
             .groupby(self.prod_col)[self.sku_col]
             .apply(set).apply(list))
        self.intsku_fact = dict(zip(k.index, k))  # {"factory1": ["SKU1", "SKU2"]}

        # asssign which sku can be supplied by which factory inverse of the previous dic
        k = (self.data[self.data[self.prod_col]
             .isin(self.factory_id)]
             .groupby(self.sku_col)[self.prod_col]
             .apply(set).apply(list))
        self.sku_plan = dict(zip(k.index, k))  # {"sku1": "[factory1]"}

        # create subseet of external skus with demand at 1st echelon
        k = (self.data[(self.data[self.loc_col].isin(self.factory_id)) 
                       & (self.data[self.sku_col].isin(self.ext_skus))]
                       .groupby(self.loc_col)[self.sku_col]
                       .apply(set).apply(list))
        self.extsku_fact = dict(zip(k.index, k))
        # assign which supplier supplies which sku
        k = (self.data[self.data[self.prod_col]
             .isin(self.ext_factory_id)]
             .groupby(self.sku_col)[self.prod_col]
             .apply(set).apply(list))
        self.supplier = dict(zip(k.index, k)) # {"SKU1": "ExtFact1"}

        # create subset of intenral skus held at reginal warehouses
        k = (self.data[(self.data[self.loc_col]
             .isin(self.rw_id)) 
             & (self.data[self.sku_col].isin(self.int_skus))]
             .groupby(self.loc_col)[self.sku_col]
             .apply(set).apply(list))
        self.intsku_RW = dict(zip(k.index, k))

        k = (self.data[(self.data[self.loc_col]
             .isin(self.rw_id)) 
             & (self.data[self.sku_col].isin(self.ext_skus))]
             .groupby(self.loc_col)[self.sku_col]
             .apply(set).apply(list))
        self.extsku_RW = dict(zip(k.index, k))

        #  Creating subsets for special factory to factory balance constraints
        b = self.factory_id.copy()
        b.remove("BRN") # all factories except BRN since already accounted for
        brn = (self.data[(self.data[self.loc_col] == "BRN") & (self.data[self.prod_col].isin(b))]
                .groupby(self.prod_col)[self.sku_col]
                .apply(list).apply(set).tolist())
        if len(brn) > 0:
            self.f2f_sku = dict({"BRN": list(brn[0])})
        i = self.factory_id.copy()
        i.remove("ITT") # all factories except ITT since already accounted for
        itt = (self.data[(self.data[self.loc_col] == "ITT") & (self.data[self.prod_col].isin(i))]
               .groupby(self.prod_col)[self.sku_col]
               .apply(list).apply(set).tolist())
        if len(itt) > 0:
            try:
                self.f2f_sku["ITT"] = list(itt[0])
            except AttributeError:
                self.f2f_sku = dict({"ITT": list(itt[0])})

        if len(itt) == 0 and len(brn) == 0:
            self.f2f_sku = {}

        # self.loc_data = self.loc_data.loc[self.loc_id]
        # self.FTL_matrix = self.FTL_matrix.drop("CoP NL", axis=1) #temp
        print(f"Number of product: {len(self.sku_id)}" + "\n",
              f"Number of locations: {len(self.loc_id)}" + "\n", 
              f"Number of periods: {len(self.time_id)}" + "\n",
              f"Number of orders: {len(self.data)}")

    def define_paramaters(self, ftl_matrix_path, s_l):
        ''' Returns a dataframe of the paramaters required for the model
        self.loc_data_path: string of the path to the location data
        '''

        self.demand_stats = self.data.groupby(
            self.sku_col)[self.qty_col].agg(["size", "mean", "std"]).fillna(0)
        self.demand_stats["var"] = self.demand_stats["std"]**2
        # self.demand_stats.fillna(0, inplace=True)
        self.demand_stats.to_csv("Demand Stats.csv")
        # self.leadt_data = self.raw_data([self.product_col, self.loc_col])["Effective Lead Time [weeks]" , ""].first()
        # self.holding_costs =self.raw_data([])
        self.FTL_matrix = pd.read_csv(ftl_matrix_path,
                                      index_col=[0])
        self.lt_df = self.data.groupby([self.sku_col])[
            ["lead_time", "std_lead_time"]].first().fillna(0)
        self.lt_df["var_lead_time"] = self.lt_df["std_lead_time"] ** 2
        self.holding_costs = self.data.groupby(
            [self.sku_col, self.loc_col])["hold_cost_pallet"].first()

        # extract demand to dictionary
        self.demand = self.data.set_index([self.time_col,
                                          self.sku_col,
                                          self.loc_col])[self.qty_col].to_dict()

        self.cw_ss_df, self.rw_ss_df = self.compute_ss(s_l)

        k = self.cw_ss_df.reset_index()
        k = (k[k[self.sku_col].isin(self.int_skus)]
                              .groupby("new_shOrigin")[self.sku_col]
                              .apply(set).apply(list))
        self.intsku_CW = dict(zip(k.index, k))  # holds all possible SKUs
        k = self.cw_ss_df.reset_index()

        k = (k[~k[self.sku_col].isin(self.int_skus)]
                               .groupby("new_shOrigin")[self.sku_col]
                               .apply(set).apply(list))
        self.extsku_CW = dict(zip(k.index, k))

        self.cw_ss, self.rw_ss = self.cw_ss_df["Safety_Stock"].fillna(0).to_dict(), self.rw_ss_df["Safety_Stock"].fillna(0).to_dict()
        self.loc_da = self.loc_data.to_dict() 

        self.sku_LOC = self.union_subset()

        self.total_demand = self.data[self.qty_col].sum()

        # self.data.set_index(["new_shOrigin", "sh_ItemId"], inplace=True)
        # abc = pd.concat([self.cw_ss_df, self.rw_ss_df])
        # abc.fillna(0, inplace=True)
        # da = pd.merge(self.data, abc, left_index=True, right_index=True)
        # da.loc[:, "Safety_Stock"] = abc["Safety_Stock"]
        # da = da.reset_index()
        # da["Total"] = da["Pallets"] + da["Safety_Stock"]
        # t = da.groupby(["new_shOrigin", "period"], as_index=False)["Total"].sum()

        # a = t.groupby(["new_shOrigin"]).agg({"Total":["max", "mean", "median"]})
        # a.to_csv("inv_load.csv")
        # t = da.groupby(["ProducedBy", "period"], as_index=False)["Total"].sum()

        # p = t.groupby(["ProducedBy"]).agg({"Total":["max", "mean", "median"]})
        # p.to_csv("prod_load.csv")
        # a = self.data[(self.data["ProducedBy"].isin(self.factory_id) & self.data["new_shOrigin"].isin(self.factory_id))]
        # a = a[a["ProducedBy"] != a["new_shOrigin"]]
        # a.to_csv("fact_to_fact_orders.csv")

    def define_variables(self):
        '''Defines the variable and their nature whcih are then added to the model
        '''

        # generating only stricly necessary indices
        self.inv_idx = [(i, w, t) for w in self.loc_id for i in self.sku_LOC.get(w, []) for t in self.time_id]
        self.inv_level = pulp.LpVariable.dicts("inventory",
                                               self.inv_idx,
                                               lowBound=0)
        dep = self.departure_allocation()

        self.ship_idx = [(o,d,i,t) for o in (self.factory_id + self.cw_id + self.ext_factory_id) for d in dep.get(o,[]) for i in (self.sku_LOC.get(o, [])) for t in self.time_id]


        self.shipment = pulp.LpVariable.dicts("shipments",
                                              self.ship_idx,
                                              lowBound=0)
        self.ftl_idx = [(o,d,t) for o,d,i,t in self.ship_idx]
        self.FTL = pulp.LpVariable.dicts("FTL",
                                         self.ftl_idx,
                                         lowBound=0)
        self.prod_idx = [(i,f,t) for f in self.factory_id for i in self.intsku_fact.get(f, [] ) for t in self.time_id]
        self.production = pulp.LpVariable.dicts("production",
                                                self.prod_idx,
                                                lowBound=0)
        self.ls_idx = self.demand.keys()
        self.lost_sales = pulp.LpVariable.dicts("lost sales",
                                                  self.ls_idx,
                                                   lowBound=0)

        # self.slack = pulp.LpVariable.dicts("slack",
        #                                    (self.sku_id,
        #                                     self.loc_id,
        #                                     self.time_id),
        #                                    lowBound=0)

    def define_objective(self, shortage=True):
        ''' Defines the objective funciton
        '''
        start_time = time.time()
        hc = self.holding_costs.to_dict()
                
        ftl = self.FTL_matrix.to_dict()
        default_hc = self.holding_costs.groupby("sh_ItemId").mean().to_dict()
        holding_costs = LpAffineExpression(((self.inv_level[i], hc.get((i[0], i[1]), default_hc[i[0]]))
                                           for i in self.inv_idx))
        trans_costs_echelon = LpAffineExpression(((self.FTL[(o,d,t)], ftl[d].get(o, 9999))
                                                  for o, d, t in self.ftl_idx))

        production_costs = LpAffineExpression(((self.production[i], self.loc_da["Prod. Costs"][i[1]])
                                                for i in self.prod_idx))
        shortage_costs = 0
        if shortage:
            shortage_costs = LpAffineExpression((self.lost_sales[i], 9999)
                                                for i in self.ls_idx)
        # slack_costs = lpSum((self.slack[i][w][t] * 50000
        #                      for i in self.sku_id
        #                      for w in self.loc_id
        #                      for t in self.time_id))

        self.inv_model += holding_costs + \
            trans_costs_echelon + production_costs  + shortage_costs  # + slack_costs
        print("--- %s seconds ---" % (time.time() - start_time))
        print("objective defined")

    def define_constraints(self):
        '''Defines the constraints to be added to the model
        '''

        factory_to_rw, cw_to_rw, fact_to_fact = self.arrival_allocation()
        dep = self.departure_allocation()
        self.sku_LOC = self.union_subset()
        constr_dic = {}
                            
        constr_dic.update({f"{f,t}ProdCap": LpAffineExpression(((self.production[(i, f, t)], 1)
                                              for i in self.intsku_fact.get(f, []))) <= self.loc_da["Prod. Cap."][f]
                                                                                    for x,f,t in self.prod_idx})

        constr_dic.update({f"{w,t}HoldCap": LpAffineExpression(((self.inv_level[(i,w,t)], 1)
                                                                  for i in self.sku_LOC.get(w, []))) <= self.loc_da["Hold. Cap."][w] 
                                                                    for x, w, t in self.inv_idx})

        constr_dic.update({f"{o,d,t}FTL":LpAffineExpression(((self.shipment[(o,d,i,t)] ,1)
                                                                            for i in self.sku_LOC.get(o, []))) == 33 * self.FTL[(o,d,t)]
                                                            for o,d,x,t in self.ship_idx})

        lt_dic = self.data["lead_time"].to_dict()
        lt = {(i,t): self.time_id[int(
                ind - lt_dic.get(i,1))] for i in self.sku_id
                                                           for ind, t in enumerate(self.time_id)} # store lead time for each sku
        inital_inv = {t:( 1 if ind>0 else 0) for ind, t in enumerate(self.time_id)} # dummy binary variable
        prevt = {t :self.time_id[ind-1]  for ind, t in enumerate(self.time_id)}

        constr_dic.update({f"{f,t, i}1stech_InvBal":LpAffineExpression(((self.production[(i,f,lt[i,t])], 1),
                                                                        *((self.shipment[(f,w,i,t)] , -1) for w in dep[f])))
                                                                        == self.inv_level[(i,f,t)] + self.demand.get((t,i,f), 0) - self.lost_sales.get((t,i,f), 0)- self.inv_level[(i,f,prevt[t])]*inital_inv[t]
                                                                        for f in self.factory_id
                                                                        for i in self.intsku_fact.get(f, []) 
                                                                        for t in self.time_id})

        constr_dic.update({f"{f,t, i}1stech_InvBal":LpAffineExpression([*((self.shipment[(x, f, i, prevt[t])], 1) for x in self.sku_plan.get(i, []))])
                                                                == self.inv_level[(i,f,t)] + self.demand.get((t,i,f), 0) - self.lost_sales.get((t,i,f), 0) - self.inv_level[(i,f,prevt[t])]*inital_inv[t]
                                                                for f in self.factory_id
                                                                for i in self.f2f_sku.get(f, []) 
                                                                for t in self.time_id})
        p = [(i,w) for w in self.loc_id for i in self.sku_LOC.get(w, []) if (self.cw_ss.get((w, i), self.rw_ss.get((w, i), 0)) + self.demand.get((self.time_id[1],i,w), 0)) > 0]
        constr_dic.update({f"{w,t, i}initial":LpAffineExpression([(self.inv_level[(i,w,t)], 1)])
                                                                  ==  self.cw_ss.get((w, i), self.rw_ss.get((w, i), 0)) + self.demand.get((self.time_id[1],i,w), 0)
                                                                        for i,w in p
                                                                        for t in self.time_id[:1]})


        constr_dic.update({f"{d,t, i}2ndech_InvBal":LpAffineExpression((*((self.shipment[(f,d,i,prevt[t])], 1) for f in self.sku_plan[i]),
                                                                        *((self.shipment[(d,w,i,t)], -1) for w in self.rw_id if d in cw_to_rw[w])))
                                                                        == self.inv_level[(i,d,t)] + self.demand.get((t,i,d), 0) - self.lost_sales.get((t,i,d), 0)- self.inv_level[(i,d,prevt[t])]*inital_inv[t]
                                                                        for d in self.cw_id                 
                                                                        for i in self.intsku_CW.get(d,[])
                                                                        for t in self.time_id[1:]})
        last_t = self.minimize_constraint()
        
        constr_dic.update({f"{w, t, i}3rdech_InvBal":LpAffineExpression((*((self.shipment[(f,w,i,prevt[t])], 1) for f in factory_to_rw[w] if i in self.intsku_fact.get(f, [])),
                                                                        *((self.shipment[(d,w,i,prevt[t])], 1) for d in cw_to_rw[w])))  
                                                                         == self.inv_level[(i,w,t)] + self.demand.get((t,i,w), 0)  - self.lost_sales.get((t,i,w), 0) - self.inv_level[(i,w,prevt[t])]*inital_inv[t]
                                                                            for w in self.rw_id
                                                                            for i in self.intsku_RW.get(w, [])
                                                                            for t in self.time_id[1:last_t[(w, i)]]})


        constr_dic.update({f"{f,t,i}1stech_ext_sku_InvBal": LpAffineExpression([*((self.shipment[(e,f,i,prevt[t])], 1) for e in self.supplier[i])])
                                                                        == self.demand.get((t,i,f), 0) - self.lost_sales.get((t,i,f), 0) + self.inv_level[(i,f,t)] - self.inv_level[(i,f,prevt[t])]*inital_inv[t]
                                                                        for f in self.factory_id
                                                                        for i in self.extsku_fact.get(f, [])
                                                                        for t in self.time_id[1:]}) 

        constr_dic.update({f"{d,t,i}2ndech_ext_sku_InvBal": LpAffineExpression((*((self.shipment[(e,d,i,prevt[t])], 1) for e in self.supplier[i]),
                                                                               *((self.shipment[(d,w,i,t)], -1) for w in self.rw_id if d in cw_to_rw[w])))
                                                                        == self.demand.get((t,i,d), 0) - self.lost_sales.get((t,i,d), 0) + self.inv_level[(i,d,t)] - self.inv_level[(i,d,prevt[t])]*inital_inv[t]
                                                                        for d in self.cw_id
                                                                        for i in self.extsku_CW.get(d,[])
                                                                        for t in self.time_id[1:]})
        # gather indices where SS > 0
        p = [k for k,v in self.cw_ss.items() if v > 0]

        constr_dic.update({f"{d, t, i}2ndech_ssreq":LpAffineExpression([(self.inv_level[(i,d,t)], 1)]) 
                                                                            >= self.cw_ss.get((d, i), 0) - self.lost_sales.get((t,i,d), 0)
                                                                            for d, i in p
                                                                            for t in self.time_id[1:]})
        p = [k for k,v in self.rw_ss.items() if v > 0]

        constr_dic.update({f"{w, t, i}3rdech_ssreq":LpAffineExpression([(self.inv_level[(i,w,t)], 1)]) 
                                                                        >= self.rw_ss.get((w,i), 0) - self.lost_sales.get((t,i,w), 0)
                                                                        for w,i in p
                                                                        for t in self.time_id[1:]})
        self.inv_model.extend(constr_dic)
        return constr_dic


    def direct_sh_constraint(self):
        constr_dic = {}
        direct_sh_extsku = dict(zip(self.loc_data.index, self.loc_data["Direct shipment ext. SKU"]))

        inital_inv = {t: (1 if ind > 0 else 0) for ind, t in enumerate(self.time_id)}
        prevt = {t: self.time_id[max(ind - 1, 0)] for ind, t in enumerate(self.time_id)}
        factory_to_rw, cw_to_rw, fact_to_fact = self.arrival_allocation()
        last_t = self.minimize_constraint()


        constr_dic.update({f"{w,t,i}3rdech_ext_sku_InvBal": LpAffineExpression([*((self.shipment[(e,w,i,prevt[t])], 1) for e in self.supplier[i])])
                                                                    == self.demand.get((t,i,w), 0) - self.lost_sales.get((t,i,w), 0) + self.inv_level[(i,w,t)] - self.inv_level[(i,w,prevt[t])]*inital_inv[t]
                                                                    for w in self.rw_id if direct_sh_extsku[w] == 1
                                                                    for i in self.extsku_RW.get(w, [])
                                                                    for t in self.time_id[1:]})

        constr_dic.update({f"{w,t,i}3rdech_ext_sku_InvBal": LpAffineExpression([*((self.shipment[(d,w,i,prevt[t])], 1) for d in cw_to_rw[w])])
                                                                    == self.demand.get((t,i,w), 0) - self.lost_sales.get((t,i,w), 0) + self.inv_level[(i,w,t)] - self.inv_level[(i,w,prevt[t])]*inital_inv[t]
                                                                    for w in self.rw_id if direct_sh_extsku[w] == 0
                                                                    for i in self.extsku_RW.get(w, [])
                                                                    for t in self.time_id[1:]})
        self.inv_model.extend(constr_dic)
        return constr_dic

    def min_batch_size_constraint(self, path):
        constr_dic = {}
        sku_min_batch = []
        m = pd.read_csv(self.batch_size_path,
                        index_col=[0])
        m = m[m["min_batch_active"] > 0 ] #  only keep min batch size for active skus
        m = m[m.index.isin(self.sku_id)]
        min_batch_size = dict(zip(m.index, m["Min. Batch Size (PAL)"]))
        min_batch_size = self.minimize_bin(self.data, min_batch_size)
        big_M = self.compute_big_M(self.data, min_batch_size)
        self.prod_on = pulp.LpVariable.dicts("ProdSetup",
                                             (list(min_batch_size),
                                              self.time_id),
                                             cat="Binary")
        # big M = max demand per SKU 
        constr_dic.update({f"{i, t}Min_batch_size": LpAffineExpression([(self.production.get((i,f,t),0), 1)])
                                                                        >= self.prod_on[i][t] * min_batch_size[i]
                                                                        for i in min_batch_size
                                                                        for f in self.factory_id if i in self.intsku_fact.get(f, [])
                                                                        for t in self.time_id})
        constr_dic.update({f"{i, t}Max_batch_size": LpAffineExpression([(self.production.get((i,f,t),0), 1)]) 
                                                                         <= self.prod_on[i][t] * big_M[i]
                                                                        for i in min_batch_size
                                                                        for f in self.factory_id if i in self.intsku_fact.get(f, [])
                                                                        for t in self.time_id})
        self.inv_model.extend(constr_dic)
        return constr_dic

    def minimize_bin(self, demand, mbs):
        temp = demand.groupby([self.sku_col, self.time_col]).agg({self.qty_col: "sum"})
        temp = temp[temp[self.qty_col] != 0].groupby([self.sku_col]).agg({self.qty_col: "min"})
        temp = dict(zip(temp.index, temp[self.qty_col]))
        l = []
        for i in list(mbs):
            if temp[i] > mbs[i]:
                l.append(i)
                del mbs[i]
        r = str(len(l) * len(self.time_id))
        if r != "0":
            print(r + " Binaries variables have been removed")
        return mbs

    def minimize_constraint(self):
        '''
        return the index last time period where a demand occur for each sku at 3rd echelon
        in order to reduce the number of constraint being generated 
        => no need to define a constraint after the last demand has occured for that sku
        returns: a dictionary -> {(Location1, SKU1): index(period1)}
        '''
        last = self.data.sort_values(by=[self.time_col]).drop_duplicates(subset=[self.sku_col, self.loc_col], keep="last")
        last = last[last[self.loc_col].isin(self.rw_id)]
        last_t = dict(zip(zip(last[self.loc_col], last[self.sku_col]),last[self.time_col]))
        last_t = {k: self.time_id.index(v)
                  for k, v in last_t.items()}
        return last_t

    def compute_big_M(self, demand, mbs):
        temp = demand.groupby([self.sku_col, self.time_col]).agg({self.qty_col: "sum"})
        temp = temp[temp[self.qty_col] != 0].groupby([self.sku_col]).agg({self.qty_col: "max"})
        temp = dict(zip(temp.index, temp[self.qty_col]))
        big_M = {}
        for i in mbs.keys():
            big_M[i] = max((temp[i] + self.cw_ss.get(i, 0) + self.rw_ss.get(i, 0)), mbs[i])
        return big_M

    def compute_ss(self, service_level):
        cent_wh_alloc = {}  # store central warehouse - regional warehouse allocation
        cw_ss = pd.DataFrame()  # store mean and std of demand aggregated per central warehouse and respective reg. warehouse
        rw_ss = pd.DataFrame()
        op = {self.qty_col: ["mean", "std"],
              "lead_time": "first",
              "std_lead_time": "first"}
        for c_wh in self.cw_id:
            # extracting whih central warehosue is responsible for which regional warehouse
            cent_wh_alloc[c_wh] = [c_wh] + self.loc_data.loc[self.loc_data["Resp. Central WH"] == c_wh].index.tolist()
            # computing demand mean and std accroding to the allocationc
            stats = self.data[self.data["new_shOrigin"].isin(cent_wh_alloc[c_wh])].groupby("sh_ItemId", as_index=False).agg(op)
            stats["new_shOrigin"] = c_wh
            cw_ss = cw_ss.append(stats)
        cw_ss.set_index(["new_shOrigin", "sh_ItemId"], inplace=True)
        cw_ss.columns = ["Pallets_mean", "Pallets_std", "lead_time_mean", "lead_time_std"]  # renaming the columns
        cw_ss["Safety_Stock"] = service_level * (cw_ss["lead_time_mean"] * cw_ss["Pallets_std"]**2
                                      + cw_ss["Pallets_mean"]**2 * cw_ss["lead_time_std"]**2)**0.5
        for r_wh in self.rw_id:
            stats = self.data[self.data["new_shOrigin"] == r_wh].groupby("sh_ItemId", as_index=False).agg(op)
            stats["new_shOrigin"] = r_wh
            rw_ss = rw_ss.append(stats)
        rw_ss.set_index(["new_shOrigin", "sh_ItemId"], inplace=True)
        rw_ss.columns = ["Pallets_mean", "Pallets_std", "lead_time_mean", "lead_time_std"] # renaming the columns
        rw_ss["Safety_Stock"] = service_level * (rw_ss["lead_time_mean"] * rw_ss["Pallets_std"]**2)**0.5

        #  Capping the max SS
        max_ss_cw = float(self.config_dict["model"]["max_ss_cw"])
        max_ss_rw = float(self.config_dict["model"]["max_ss_rw"])

        cw_ss.loc[cw_ss["Safety_Stock"] > max_ss_cw, "Safety_Stock"] = max_ss_cw
        rw_ss.loc[rw_ss["Safety_Stock"] > max_ss_rw, "Safety_Stock"] = max_ss_rw
        try:
            cw_ss.to_excel(self.writer, sheet_name="Central Warehouse SS")
            rw_ss.to_excel(self.writer, sheet_name="Regional Warehouse SS")
            cw_ss.to_csv("./CSV export files/Central Warehouse SS.csv")
            cw_ss.to_csv("./CSV export files/Regional Warehouse SS.csv")
        except IndexError:
            pass

        return cw_ss, rw_ss

    def arrival_allocation(self):
        ''' dicitoanry holding for each destination in the network the location that it could be supplied from
        example: {'Reg. Warehouse1': ['factory1, factory2']}
        '''
        factory_to_rw = {}
        cw_to_rw = {}
        fact_to_fact = {}
        for w in self.rw_id:
            factory_to_rw[w] = self.loc_data.loc[w, "Factory_allocation"].split(",")
            
            try:
                cw_to_rw[w] = self.loc_data.loc[w, "Resp. Central WH"].split(",")
            except AttributeError:
                cw_to_rw[w] = []
        for f in self.factory_id:
            try:
                fact_to_fact[f] = self.loc_data.loc[f, "Factory_to_Factory"].split(",")
            except AttributeError:
                fact_to_fact[f] = []

        return factory_to_rw, cw_to_rw, fact_to_fact

    def union_subset(self):
        '''
        for each lcoation combine all possible skus that could be held/produced at that location
        {'location1': ['SKU1', 'SKU2', 'SKU3']}
        '''
        sku_LOC = {}
        for l in self.loc_id:
            if l in self.factory_id:
                sku_LOC[l] = self.intsku_fact.get(l, []) + self.extsku_fact.get(l, []) + self.f2f_sku.get(l, [])
            if l in self.cw_id:
                sku_LOC[l] = self.intsku_CW.get(l, []) + self.extsku_CW.get(l, [])
            if l in self.rw_id:
                sku_LOC[l] = self.intsku_RW.get(l, []) + self.extsku_RW.get(l, [])

        k = (self.data[self.data[self.prod_col]
             .isin(self.ext_factory_id)]
            .groupby(self.prod_col)[self.sku_col]
            .apply(set).apply(list))
        ext = dict(zip(k.index, k))
        sku_LOC.update(ext)
        return sku_LOC

    def departure_allocation(self):
        ''' dictionarry holding for each departure point in the network the set of facilities it is allow to serve
        example: {'factory1': ['Central Warehoouse1', 'Regional Warehouse2']}
        '''
        loc = self.loc_data.fillna("")
        dep = {}
        for f in self.factory_id:
            dep[f] = self.cw_id + loc.index[loc["Factory_allocation"].str.contains(f)].tolist() + loc.index[loc["Factory_to_Factory"].str.contains(f)].tolist()
        for d in self.cw_id:
            dep[d] = loc.index[loc["Resp. Central WH"].str.contains(d)].tolist()
        for e in self.ext_factory_id:
            dep[e] = self.data[self.loc_col][self.data[self.prod_col] == e].unique().tolist() + self.cw_id
        return dep

    def bi_obj(self):
        folder = "./Saved Models/"
        model_file = "biobj"
        epsilon = [0]
        bi_obj_results = {}
        self.define_indices(loc_data_path=self.loc_data_path)
        self.define_subsets()
        self.define_paramaters(ftl_matrix_path=self.ftl_matrix_path, s_l=self.service_level)
        start_time = time.time()
        self.define_variables()
        self.define_objective(shortage=False)
        self.service_level = pulp.LpVariable("service level")
        constr_dic = {}
        constr_dic = self.define_constraints()
        constr_dic.update(self.direct_sh_constraint())
        mbsc = self.min_batch_size_constraint(self.batch_size_path)
        constr_dic.update(mbsc)
        self.inv_model.extend({"service_level_measure": LpAffineExpression((self.lost_sales[(i, w, t)], 1)
                                                                            for i,w,t in self.ls_idx) == (1- self.service_level) * self.total_demand})
        self.inv_model.extend({"epsilon": LpAffineExpression(self.service_level) >= epsilon[-1]})
        constr_dic.update({"service_level_measure": LpAffineExpression((self.lost_sales[(i, w, t)], 1)
                                                                            for i,w,t in self.ls_idx) == (1- self.service_level) * self.total_demand})
        constr_dic.update({"epsilon": LpAffineExpression(self.service_level) >= epsilon[-1]})
        self.save_model_json(self.inv_model, constraints=constr_dic)
        while epsilon[-1] < 2.33:
            variables, model = LpProblem.from_json(folder + model_file + ".json")
            with open(folder + model_file + "constraints.json") as f:
                cons = ujson.load(f)
            self.call_cplex(model)
            z2 = self.service_level.varValue
            bi_obj_results[epsilon[-1]] = value(model.objective)
            epsilon.append(epsilon[-1] + 0.1)
            self.define_paramaters(ftl_matrix_path=self.ftl_matrix_path, s_l=epsilon[-1])


            model.constraints = OrderedDict(zip(cons.keys(), model.constraints.values())) # rename the constraint for easier altering
            model.constraints["epsilon"].constant = epsilon[-1]
            self.update_constraint_rhs(model.constraints)
            self.save_model_json(model, constraints=model.constraints)

    def build_model(self):
        ''' calls all the required function to initialize, solve and export the model
        '''
        self.define_indices(loc_data_path=self.loc_data_path)
        self.define_subsets()
        self.define_paramaters(ftl_matrix_path=self.ftl_matrix_path, s_l=self.service_level)

        if any(os.scandir("./Saved Models")):
            print("existing model(s) found")
            while True:
                old_mod = input("Use existing model? (y/n) ")
                if old_mod == "y":
                    self.recycle_model("./Saved Models/")
                    break
                if old_mod == "n" or any(os.scandir("./Saved Models")) is False:
                    start_time = time.time()
                    self.define_variables()
                    self.define_objective()
                    # pr = cProfile.Profile()
                    # pr.enable()

                    constr_dic = {}
                    constr_dic = self.define_constraints()
                    constr_dic.update(self.direct_sh_constraint())
                    mbsc = self.min_batch_size_constraint(self.batch_size_path)
                    constr_dic.update(mbsc)

                    # pr.disable()
                    # s = io.StringIO()
                    # ps = pstats.Stats(pr, stream=s).sort_stats('tottime')
                    # ps.print_stats()

                    # with open('test.txt', 'w+') as f:
                    #     f.write(s.getvalue())

                    
                    self.mc_time = time.time() - start_time
                    print("Model Creation Time:-------------",
                          "\n" + str(self.mc_time))
                    self.save_model_json(model=self.inv_model, constraints=constr_dic, ask=True)

                    start_time = time.time()
                    self.call_cplex(self.inv_model)
                    self.solv_time = time.time() - start_time

                    print(f"number of variables: {self.inv_model.numVariables()}")
                    if bool(self.config_dict["debugging"]["write_lp"]):
                        self.inv_model.writeLP("test.LP")
                    if bool(self.config_dict["debugging"]["write_lp"]):

                        self.inv_model.writeMPS("test.mps")
                    print("Objective value: ", value(self.inv_model.objective))
                    print("Inventory level:--------------")
                    self.inv_result = self.export_vars_3d(indices=self.inv_idx,
                                                          variable=self.inv_level,
                                                          filename="Inventory Level")
                    print("Shipments:------------------")
                    self.ship_result, self.ftl_result = self.export_vars_4d(indices=self.ship_idx,
                                                                            variable=self.shipment,
                                                                            filename="Shipment")

                    print("Production:-----------------")
                    self.prod_result = self.export_vars_3d(indices=self.prod_idx,
                                                           variable=self.production,
                                                           filename="Production")
                    print("Lost Sales:-----------------")
                    self.ls_result = self.export_vars_3d(indices=self.ls_idx,
                                                         variable=self.lost_sales,
                                                         filename="Lost Sales")
                    self.writer.save()
                    break

                else:
                    print("Invalid input. ")
                    continue


    def call_cplex(self, model):
        param = self.config_dict["cplex"]

        solver = CPLEX_PY()

        solver.buildSolverModel(model)
        try:
            # solver.solverModel.parameters.emphasis.memory.set(int(param["memory_emphasis"]))
            solver.solverModel.parameters.workmem.set(int(param["working_memory"]))
            solver.solverModel.parameters.mip.strategy.file.set(int(param["node_file"]))
            solver.solverModel.parameters.mip.tolerances.mipgap.set(float(param["mipgap"]))

            solver.callSolver(model)
        except cplex.exceptions.errors.CplexSolverError:
            print("One of the Cplex parametrs specified is invalid. ")
        status = solver.findSolutionValues(model)
        solver.solverModel.parameters.conflict.display.set(2) # if model is unsolvable will display the problematic constraint(s)

    def export_vars_3d(self, indices, variable, filename):
        ''' Formats the solution and returns it embedded in a pandas dataframe
        time_ind: time index used by the variable
        wh_ind: location index used by the varaible
        prod_ind: product (group) index used by the variable
        variable: variable name to be exported
        filename: under which name to save the varaible to be exported to csv
        '''
        start_time = time.time()
        dic = {"time": [], "warehouse": [], "product": [], "value": []}
        for i,w,t in indices:
            dic["time"].append(t)
            dic["warehouse"].append(w)
            dic["product"].append(i)
            dic["value"].append(variable[(i,w,t)].varValue)

        self.var_df = pd.DataFrame.from_dict(dic)
        self.var_df = self.var_df.fillna(0)
        try:
            if variable == (self.lost_sales | ls_dic):
                self.var_df.columns = ["warehouse", "sh_ItemId", "time", "value"]
        except AttributeError:
            pass
        except NameError: 
            pass
        self.var_df.to_excel(self.writer, sheet_name=filename)
        self.var_df.to_csv(f"./CSV export files/{filename}.csv")
        self.var_df.loc[self.var_df["value"] < 0, "value"] = 0
        print(self.var_df)
        print("--- %s seconds ---" % (time.time() - start_time))
        print(f"{filename} exported")
        return self.var_df

    def export_vars_4d(self, indices, variable, filename):
        start_time = time.time()
        dic = {"time": [], "origin": [],
               "destination": [], "product": [], "value": []}
        for o,d,i,t in indices:
            if variable[(o,d,i,t)].varValue is not None:
                if variable[(o,d,i,t)].varValue > 0:
                    dic["origin"].append(o)
                    dic["destination"].append(d)
                    dic["product"].append(i)
                    dic["time"].append(t)
                    dic["value"].append(variable[(o,d,i,t)].varValue)

        self.var_df = pd.DataFrame.from_dict(dic)
        self.var_df = self.var_df.fillna(0)
        self.var_df.to_excel(self.writer, sheet_name=filename)
        self.var_df.to_csv(f"./CSV export files/{filename}.csv")
        self.FTL_df = self.var_df.groupby(["origin", "destination", "time"], as_index=True).sum()
        self.FTL_df.value = self.FTL_df.value / 33
        self.FTL_df.value = np.ceil(self.FTL_df.value)

        try:
            self.FTL_df.to_excel(self.writer, sheet_name="Full-Truck Loads")
            self.FTL_df.to_csv("./CSV export files/Full-Truck Loads.csv")
        except IndexError:
            pass

        print(self.var_df)
        print("FTL----------------------")
        print(self.FTL_df)
        print("--- %s seconds ---" % (time.time() - start_time))
        print(f"{filename} exported")
        return self.var_df, self.FTL_df

    def save_model_json(self, model, constraints, ask=False):
        if ask is False:
            model.toJson("./Saved Models/biobj.json")  # saving the model to json
            with open("./Saved Models/biobjconstraints.json", "w") as cons_out:
                ujson.dump(constraints, cons_out, sort_keys=False)   # saving the constraints separetely in order to regain the constraints naming convention
        else:
            if bool(self.config_dict["model"]["save_model"]):
                model_name = self.config_dict["model"]["model_name"]                 
                model.toJson(f"./Saved Models/{model_name}.json")  # saving the model to json
                with open(f"./Saved Models/{model_name}constraints.json", "w") as cons_out:
                    ujson.dump(constraints, cons_out, sort_keys=False) # saving the constraints separetely in order to regain the constraints naming convention


# Actually creating an instance of the classes in which the model is defined

t = 53
mctime_dic = {}
solvtime_dic = {}
gen = {}
start_time = time.time()
I = InventoryModel(sku_col="sh_ItemId",
                   time_col="period",
                   loc_col="new_shOrigin",
                   qty_col="Pallets",
                   prod_col="ProducedBy")
if I.arg_parse():
    I.bi_obj()
else:
    I.build_model()

    if I.config_dict["visualisation"]["abs_cycle_ss_barplot"]:
        print(I.config_dict["visualisation"])
        v.cycle_ss_barplot_abs(inventory=I.inv_result[I.inv_result["time"].isin(I.time_id[1:])],
                               ss=[I.cw_ss_df.reset_index(), I.rw_ss_df.reset_index()])
    if I.config_dict["visualisation"]["rel_cycle_ss_barplot"]:
        v.cycle_ss_barplot_rel(inventory=I.inv_result[I.inv_result["time"].isin(I.time_id[1:])],
                               ss=[I.cw_ss_df.reset_index(), I.rw_ss_df.reset_index()])

# mctime_dic[i] = I.mc_time
# solvtime_dic[i] = I.solv_time
# gen["model_creation"] = mctime_dic
# gen["solving time"] = solvtime_dic
# with open("times.json", "w") as f:
#     json.dump(gen, f)
# v.find_intersection(inventory=I.inv_result,
#                     baseline_path="./CSV input files/baseline.csv")

time_taken = time.time() - start_time
print("Total Time: ", time_taken)
   


# calling the function that will assemble the model together
# pr = cProfile.Profile()
# pr.enable()

# pr.disable()
# s = io.StringIO()
# ps = pstats.Stats(pr, stream=s).sort_stats('tottime')
# ps.print_stats()

# with open('test.txt', 'w+') as f:
#     f.write(s.getvalue())


