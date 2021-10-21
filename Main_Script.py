import pandas as pd
import numpy as np
from pulp import *
import time
import os
import visualisation as v
import ujson
import collections
from collections import defaultdict
import re
import configparser, os.path, argparse
from statistics import NormalDist



class InventoryModel():
    """class used to define and construct the inventory linear program
       data_path: the file path of the data containing the orders
       product_col: string of the column's header containing the product (group) id
       time_col: string of the column's header containing the date of the order
       loc_col: string of the column's header containing the location of the order
       qty_col: string of the column's header containing the quantity of the order
    """

    def __init__(self, sku_col, time_col, loc_col, qty_col, prod_col):
        args = self.arg_parse().parse_args()
        self.biobj = args.biobj
        self.tune = args.tune
        self.config_dict = self.read_config()
        self.sku_col = sku_col
        self.time_col = time_col
        self.loc_col = loc_col
        self.qty_col = qty_col
        self.prod_col = prod_col
        self.decimal = self.config_dict["filenames"]["decimal"]
        d = {sku_col: "object"}
        self.raw_data = pd.read_csv(self.config_dict["filenames"]["orders"],
                                    index_col=[0],
                                    nrows=20000,
                                    dtype=d,
                                    sep=None,
                                    engine="python",
                                    decimal=self.decimal)

        self.service_level = NormalDist().inv_cdf(float(self.config_dict["model"]["service_level"]))
        self.ftl_matrix_path = self.config_dict["filenames"]["ftl_matrix"]
        self.batch_size_path = self.config_dict["filenames"]["batch_size"]
        self.loc_data_path = self.config_dict["filenames"]["location_data"]


        # remove orders that do not have a location specified
        self.data = self.raw_data[self.raw_data[loc_col].notna()]

        self.writer = pd.ExcelWriter(self.config_dict["filenames"]["result_file"], engine="xlsxwriter")
        if not os.path.exists("CSV export files"):
            os.makedirs("CSV export files")
        if not os.path.exists("Saved Models"):
            os.makedirs("Saved Models")

        try:
            self.data["XYZ_cluster"] = self.data["XYZ_cluster"].fillna("Z")
            self.data = self.data.loc[~((self.data["ABC_cluster"] == "C") & (self.data["XYZ_cluster"] == "Z"))]
        except KeyError:
            print("No clusters columns found")
        self.data.to_excel(self.writer, sheet_name="Demand")
        self.data.to_csv("./CSV export files/Demand.csv")
        print(self.data)
        print(self.data.groupby(sku_col)[
              self.qty_col].agg(Mean='mean', Sum='sum', Size="size"))

        self.inv_model = pulp.LpProblem("Inventory_Optimization",
                                        LpMinimize)  # creates an object of an LpProblem (PuLP) to which we will assign variables, constraints and objective function
    
    def arg_parse(self):
        '''
        defines argument of cmd flags
        '''
        parser = argparse.ArgumentParser(description= "Launch a bi-objective optimisation")
        parser.add_argument("--biobj", type=bool)
        parser.add_argument("--tune", type=bool)
        return parser

    def read_config(self, overide=None):
        '''
        extract the paramaters defined nto config file into a dictionary for easier access troughout code
        '''
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
        '''
        contains the logic when loading an existing model and re-solving it
        '''
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
        model.constraints = OrderedDict(zip(cons.keys(), model.constraints.values()))  # restore constraint nomenclature
        self.update_constraint_rhs(model.constraints)
        if self.tune:
            self.call_cplex_tuning_tool(model)
        else:
            if self.config_dict["cplex"]["cplex"]:
                self.call_cplex(model)
            if self.config_dict["gurobi"]["gurobi"]:
                self.call_gurobi(model)
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
        '''
        allows to restore variable names after re-solve
        '''
        a = tuple(re.findall(r"'(\w+)'", var_key))
        return a

    def parse_variables(self, variables):
        '''
        allows to tidily format variables after resolve
        '''
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
        '''
        this method allows to modify the RHS (for some) coefficient upon loading an existing model
        '''
        cons_modified = {"Production Cap.": 0,
                         "Holding Cap.": 0,
                         "Initial Inv.": 0,
                         "Min. Inv.": 0} 
        for fact in self.factory_id:
            if constraints.get(f"('{fact}', '0')ProdCap", False) is not False:
                if constraints[f"('{fact}', '0')ProdCap"].constant !=  - self.loc_da["prod. cap."][fact]: #  check if prod. cap. has changed
                    new_cap = - self.loc_da["prod. cap."][fact]
                    for t in self.time_id:
                        constraints[f"('{fact}', '{t}')ProdCap"].constant = new_cap
                        cons_modified["Production Cap."] += 1
        for wh in self.loc_id:
            if constraints.get(f"('{wh}', '0')HoldCap", False) is not False:
                if constraints[f"('{wh}', '0')HoldCap"].constant !=  - self.loc_da["hold. cap."][wh]:
                    new_cap = - self.loc_da["hold. cap."][wh]
                    for t in self.time_id:
                        constraints[f"('{wh}', '{t}')HoldCap"].constant = new_cap
                        cons_modified["Holding Cap."] += 1

        last_t = self.find_last_t()
        p = [k for k, v in self.cw_ss.items() if v > 0]
        t_ = self.time_id[1]
        for d, i in p:
            if constraints[f"('{d}', '{t_}', '{i}')2ndech_ssreq"].constant != - self.cw_ss[(d, i)]:
                new_ss = - self.cw_ss[(d, i)]
                for t in self.time_id[:last_t[i]]:
                    constraints[f"('{d}', '{t}', '{i}')2ndech_ssreq"].constant = new_ss
                    cons_modified["Min. Inv."] += 1

                old_initial = constraints[f"('{d}', '0', '{i}')initial"].constant
                constraints[f"('{d}', '0', '{i}')initial"].constant -= (old_initial - new_ss)
                cons_modified["Initial Inv."] += 1


        p = [k for k, v in self.rw_ss.items() if v > 0]
        for w, i in p:
            if constraints[f"('{w}', '{t_}', '{i}')3rdech_ssreq"].constant != - self.rw_ss[(w, i)]:
                new_ss = - self.rw_ss[(w, i)]
                for t in self.time_id[:last_t[i]]:
                    constraints[f"('{w}', '{t}', '{i}')3rdech_ssreq"].constant = new_ss
                    cons_modified["Min. Inv."] += 1

                old_initial = constraints[f"('{w}', '0', '{i}')initial"].constant
                constraints[f"('{w}', '0', '{i}')initial"].constant -= (old_initial - new_ss)
                cons_modified["Initial Inv."] += 1

                old_initial = constraints[f"{w,str(self.time_id[1]), i}initial"].constant
                constraints[f"{w,self.time_id[1], i}initial"].constant -= (old_initial - new_ss)
                cons_modified["Initial Inv."] += 1

        if all(value == 0 for value in cons_modified.values()):
            print("No constraint has been modified")
        else:
            print("Production Capacity Cons. modfied: " + str(cons_modified["Production Cap."]))
            print("Holding Capacity Cons. modfied: " + str(cons_modified["Holding Cap."]))
            print("Initial Inv. modfied: " + str(cons_modified["Initial Inv."]))
            print("Min. Inv. Cons. modfied: " + str(cons_modified["Min. Inv."]))

    def define_indices(self, loc_data_path):
        ''' Returns the unique values for sets that can be derived from the input location data
        Required keywords argument:
        - loc_data_path: path to the location data file
        '''
        self.loc_data = pd.read_csv(loc_data_path,
                            index_col=[0],
                            sep=None,
                            engine="python",
                            decimal=self.decimal)
        self.sku_id = self.data[self.sku_col].unique()  # Set of all SKUs I
        self.time_id = self.data[self.time_col].unique().tolist()  # Set of all time periods T
        self.time_id.insert(0,"0")  # create t=0 index
        self.loc_id = self.loc_data.index.tolist()  # Set of all network facilities W
        self.factory_id = (self.loc_data[self.loc_data["echelon"] == "Factory"]
                           .index.tolist())  # Set of factories F
        self.cw_id = (self.loc_data[self.loc_data["echelon"] == "Central"]
                      .index.tolist())  # Set of CW D
        self.rw_id = (self.loc_data[self.loc_data["echelon"] == "Regional"]
                      .index.tolist())  # set of RW J
        self.ext_factory_id = list(set(self.data[self.prod_col].unique().tolist()) - set(self.factory_id))  # Set of ext suppliers O

    def define_subsets(self):
        ''' Returns the values of the different subset some of which are user-specified, 
        some of which are automatically derived
        Also returns non-essential subsets which facilititate a tighter formulation of variables, 
        objective function and constraints 
        '''
        # subset of skus produced internally
        self.int_skus = (self.data[self.data[self.prod_col]
                               .isin(self.factory_id)][self.sku_col]
                               .to_list())  # [intsku1, intsku2]

        # subset of external skus
        self.ext_skus = (self.data[self.data[self.prod_col]
                         .isin(self.ext_factory_id)][self.sku_col]
                         .to_list())  # [extsku1, extsku2]

        # assign which factory produces which sku
        k = (self.data[self.data[self.prod_col]
             .isin(self.factory_id)]
             .groupby(self.prod_col)[self.sku_col]
             .apply(set).apply(list))
        self.intsku_fact = dict(zip(k.index, k))  # {"factory1": ["intSKU1", "intSKU2"]}

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
        self.extsku_fact = dict(zip(k.index, k))  # {"factory1": [extSKU1, extSKU2]}
        # assign which supplier supplies which sku
        k = (self.data[self.data[self.prod_col]
             .isin(self.ext_factory_id)]
             .groupby(self.sku_col)[self.prod_col]
             .apply(set).apply(list))
        self.supplier = dict(zip(k.index, k)) # {"SKU1": "ExtFact1"}

        # create subset of intenral skus held at reginal warehouses
        k = (self.data[(self.data[self.loc_col]
             .isin(self.rw_id)) & (self.data[self.sku_col]
             .isin(self.int_skus))]
             .groupby(self.loc_col)[self.sku_col]
             .apply(set).apply(list))
        self.intsku_RW = dict(zip(k.index, k))  # {"RW1": ["intsku1", "intsku2"]}

        k = (self.data[(self.data[self.loc_col]
             .isin(self.rw_id)) 
             & (self.data[self.sku_col].isin(self.ext_skus))]
             .groupby(self.loc_col)[self.sku_col]
             .apply(set).apply(list))
        self.extsku_RW = dict(zip(k.index, k)) # {"RW1": ["extsku1", "extsku2"]}

        #  Creating subsets for special factory to factory balance constraints
        f2f_fact_id = list(self.loc_data.loc[~self.loc_data["factory_to_factory"].isna()].index)  # extract factories in need of special constraints
        self.f2f_sku = {}
        for f in f2f_fact_id:
            temp = self.loc_data.loc[f, "factory_to_factory"].split(",")  # extract factories that can supply ["fact1", "fact2"]
            skus = (self.data[(self.data[self.loc_col] == f) & (self.data[self.prod_col].isin(temp))]
                    .groupby(self.loc_col)[self.sku_col]
                    .apply(set).tolist())  # extract SKUs that 1. factory f 2. is produced by a factory in "factory_to_factory" column, {f: ["sku1", "sku2"]}

            if len(skus) > 0:
                skus = [s for s in skus[0]]
                self.f2f_sku[f] = skus  # {fact1: ["sku1", "sku2"]}

        print(f"Number of product: {len(self.sku_id)}" + "\n",
              f"Number of locations: {len(self.loc_id)}" + "\n", 
              f"Number of periods: {len(self.time_id)}" + "\n",
              f"Number of orders: {len(self.data)}")

    def define_paramaters(self, ftl_matrix_path, s_l):
        ''' Returns different dictionaries and dataframes of parameters
        - Arguments:
            self.loc_data_path: string of the path to the location data
            s_l: z-score of service elevel to be used when computing ss
        -Output:
        self.demand_stats: general statistics about the demand
        self.FTL_matrix: loading ftl matrix into a df
        self.lt_df: loading lead timess and lt std into a df
        self.holding_costs: loading holding costs int o a df
        self.demand: extracting demand and converting to df
        self.cw_ss_df: computing ss for cws
        self.rw_ss_df: compitong ss for rws
        self.loc_da: converting paramaters specified in location data file into a dict
        self.intsku_CW: int. SKUs that could be held at each cw
        self.extsku_CW: ext. SKUs that could be held at each cw
        '''

        self.demand_stats = self.data.groupby(
            self.sku_col)[self.qty_col].agg(["size", "mean", "std"]).fillna(0)
        self.demand_stats["var"] = self.demand_stats["std"]**2
        self.demand_stats.to_csv("Demand Stats.csv")

        self.FTL_matrix = pd.read_csv(ftl_matrix_path,
                                      index_col=[0],
                                      sep=None,
                                      engine="python",
                                      decimal=self.decimal)
        # putting different parametrs into dictionaries
        self.lt_df = self.data.groupby([self.sku_col])[
            ["lead_time", "std_lead_time"]].first().fillna(0)
        self.lt_df["var_lead_time"] = self.lt_df["std_lead_time"] ** 2
        self.holding_costs = self.data.groupby(
            [self.sku_col, self.loc_col])["hold_cost_pallet"].first()

        # extract demand to dictionary
        self.demand = self.data.set_index([self.time_col,
                                          self.sku_col,
                                          self.loc_col])[self.qty_col].to_dict()
        print("service level: " + str(s_l))
        self.cw_ss_df, self.rw_ss_df = self.compute_ss(s_l)  # computing safety stock

        k = self.cw_ss_df.reset_index()
        k = (k[k[self.sku_col].isin(self.int_skus)]
                              .groupby(self.loc_col)[self.sku_col]
                              .apply(set).apply(list))
        self.intsku_CW = dict(zip(k.index, k))  # subset of int SKUs held at central warehouses {"CW1": ["intSKU1", "intSKU2"]}
        k = self.cw_ss_df.reset_index()

        k = (k[~k[self.sku_col].isin(self.int_skus)]
                               .groupby(self.loc_col)[self.sku_col]
                               .apply(set).apply(list))
        self.extsku_CW = dict(zip(k.index, k))  # subset of ext SKUs held at central warehouses {"CW1": ["extSKU1", "extSKU2"]}

        self.cw_ss, self.rw_ss = self.cw_ss_df["Safety_Stock"].fillna(0).to_dict(), self.rw_ss_df["Safety_Stock"].fillna(0).to_dict()
        self.loc_da = self.loc_data.to_dict()

        self.sku_LOC = self.union_subset()

        self.total_demand = self.data[self.qty_col].sum()

    def define_variables(self):
        '''Defines the variable and their nature which are then added to the model
        before each variable strictly necessary indices are genrated within a list of tuples
        '''

        # generating only stricly necessary indices
        last_t = self.find_last_t()
        self.inv_idx = [(i, w, t) for w in self.loc_id for i in self.sku_LOC.get(w, []) for t in self.time_id[:last_t[i]]]
        self.inv_level = pulp.LpVariable.dicts("inventory",
                                               self.inv_idx,
                                               lowBound=0)
        dep = self.departure_allocation() 

        origins = self.factory_id + self.cw_id + self.ext_factory_id
        self.ship_idx = [(o, d, i, t) for i, d, t in self.inv_idx for o in origins if i in self.sku_LOC.get(o, []) and d in dep.get(o, [])]

        self.shipment = pulp.LpVariable.dicts("shipments",
                                              self.ship_idx,
                                              lowBound=0)
        self.ftl_idx = [(o, d, t) for o,d,i,t in self.ship_idx]
        self.sum_idx = defaultdict(list)
        for a in self.ship_idx:  # isolate all SKUs that occur for each origin-destination-period triplet
            k = (a[0], a[1], a[3])
            self.sum_idx[k].append(a[2])  # {(o,d,t): ["SKU1", "SKU2"]}

        self.FTL = pulp.LpVariable.dicts("FTL",
                                         self.ftl_idx,
                                         lowBound=0,
                                         cat="Integer")
        self.prod_idx = [(i,f,t) for f in self.factory_id for i in self.intsku_fact.get(f, [] ) for t in self.time_id[:last_t[i] + 1]]
        self.production = pulp.LpVariable.dicts("production",
                                                self.prod_idx,
                                                lowBound=0)
        self.ls_idx = self.demand.keys()
        self.lost_sales = pulp.LpVariable.dicts("lost sales",
                                                 self.ls_idx,
                                                 lowBound=0)
        # self.slack_ind = [(w, t) for w in self.loc_id for t in self.time_id]
        # self.slack = pulp.LpVariable.dicts("slack",
        #                                    self.slack_ind,
        #                                    lowBound=0)

    def define_objective(self, shortage=True):
        ''' Defines the objective funciton
        '''
        start_time = time.time()
        costs = self.config_dict["model"]
        hc = self.holding_costs.to_dict()

        ftl = self.FTL_matrix.to_dict()
        default_hc = self.holding_costs.groupby(self.sku_col).mean().to_dict()
        holding_costs = LpAffineExpression(((self.inv_level[i], hc.get((i[0], i[1]), default_hc[i[0]]))
                                            for i in self.inv_idx))
        trans_costs_echelon = LpAffineExpression(((self.FTL[(o, d, t)], ftl[d].get(o, int(costs["ftl_def"])))
                                                  for o, d, t in self.ftl_idx))
        prod_costs = dict(zip(zip(self.data[self.sku_col], self.data[self.prod_col]), self.data["prod_costs"]))
        production_costs = LpAffineExpression(((self.production[(i, f, t)], prod_costs[i, f])
                                               for i, f, t in self.prod_idx))
        shortage_costs = 0
        if shortage:
            shortage_costs = LpAffineExpression((self.lost_sales[i], int(costs["short_costs"]))
                                                for i in self.ls_idx)

        # slack_costs = LpAffineExpression(((self.slack[(w, t)], 9999)
        #                                          for w in self.loc_id
        #                                          for t in self.time_id))

        self.inv_model += holding_costs + \
            trans_costs_echelon + production_costs  + shortage_costs  # + slack_costs
        print("--- %s seconds ---" % (time.time() - start_time))
        print("objective defined")

    def define_constraints(self):
        '''Defines the constraints to be added to the model
        Constraints are consturcted and added to constr_dic
        constr_dic is the passed to be added as constraints to LpProblem object
        '''
        last_t = self.find_last_t()  # for each sku store last time period where demand occurs

        factory_to_rw, cw_to_rw, fact_to_fact = self.arrival_allocation()  # store allowed shiments destination
        dep = self.departure_allocation()  # store allowed shipment departure
        self.sku_LOC = self.union_subset()  # for each facility which sku should be held there
        constr_dic = {}
               
        constr_dic.update({f"{f,t}ProdCap": LpAffineExpression(((self.production[(i, f, t)], 1)
                                                                for i in self.intsku_fact.get(f, []) if last_t[i] > self.time_id.index(t))) <= self.loc_da["prod. cap."][f]
                           for x, f, t in self.prod_idx})

        last_ss = self.sum_ss(self.cw_ss, self.rw_ss, last_t)
        constr_dic.update({f"{w,t}HoldCap": LpAffineExpression(((self.inv_level[(i,w,t)], 1)
                                                                  for i in self.sku_LOC.get(w, []) if last_t[i] > self.time_id.index(t))) <= self.loc_da["hold. cap."][w] - last_ss.get((w, t), 0)
                                                                    for x, w, t in self.inv_idx})

        constr_dic.update({f"{o,d,t}FTL":LpAffineExpression(((self.shipment[(o,d,i,t)] ,1)
                                                                            for i in self.sum_idx.get((o,d,t), []))) == 33 * self.FTL[(o,d,t)]
                                                            for o,d,x,t in self.ship_idx})

        lt_dic = self.data["lead_time"].to_dict()
        lt = {(i,t): self.time_id[max(int(ind - lt_dic.get(i,1)), ind-1)]
                                    for i in self.sku_id
                                    for ind, t in enumerate(self.time_id)} # store lead time for each sku
        prevt = {t :self.time_id[ind-1]  for ind, t in enumerate(self.time_id)}  # store the previous chronological time period
        constr_dic.update({f"{f, t, i}1stech_InvBal": LpAffineExpression(((self.production[(i, f, lt[i, t])], 1),
                                                                        *((self.shipment[(f,w,i,t)] , -1) for w in dep[f] if i in self.sku_LOC.get(w, []))))
                                                                        == self.inv_level[(i,f,t)] + self.demand.get((t,i,f), 0) - self.lost_sales.get((t,i,f), 0)- self.inv_level.get((i,f,prevt[t]), 0)
                                                                        for f in self.factory_id
                                                                        for i in self.intsku_fact.get(f, []) 
                                                                        for t in self.time_id[1: last_t[i]]})
        constr_dic.update({f"{f, t}initial_shipments": LpAffineExpression([*((self.shipment[(f,d,i,t)] , -1) for d in dep[f] for i in self.sku_LOC.get(f, []) if i in self.sku_LOC.get(d, []))])
                                                                        == 0
                                                                        for f in self.factory_id + self.cw_id
                                                                        for t in self.time_id[0]})

        constr_dic.update({f"{f,t, i}1stech_InvBal": LpAffineExpression([*((self.shipment[(x, f, i, prevt[t])], 1) for x in self.sku_plan.get(i, []))])
                                                                == self.inv_level[(i,f,t)] + self.demand.get((t,i,f), 0) - self.lost_sales.get((t,i,f), 0) - self.inv_level.get((i,f,prevt[t]), 0)
                                                                for f in self.factory_id
                                                                for i in self.f2f_sku.get(f, []) 
                                                                for t in self.time_id[1:last_t[i]]})

        constr_dic.update({f"{w,'0', i}initial": LpAffineExpression([(self.inv_level[(i, w, "0")], 1)])
                                                                  ==  self.cw_ss.get((w, i), self.rw_ss.get((w, i), 0)) + self.demand.get((t,i,w), 0) 
                                                                        for i,w, t in self.inv_idx if t == self.time_id[1]})
        try:
            constr_dic.update({f"{w,str(self.time_id[1]), i}initial":LpAffineExpression([(self.inv_level[(i, w, self.time_id[1])], 1)])
                                                                      ==  self.cw_ss.get((w, i), self.rw_ss.get((w, i), 0)) + self.demand.get((t,i,w), 0) 
                                                                            for i,w, t in self.inv_idx if t == self.time_id[2] if w in self.rw_id})
        except IndexError:
            pass
        constr_dic.update({f"{d,t, i}2ndech_InvBal":LpAffineExpression((*((self.shipment[(f,d,i,prevt[t])], 1) for f in self.sku_plan[i]),
                                                                        *((self.shipment[(d,w,i,t)], -1) for w in self.rw_id if d in cw_to_rw[w] and i in self.sku_LOC.get(w, []))))
                                                                        == self.inv_level[(i,d,t)] + self.demand.get((t,i,d), 0) - self.lost_sales.get((t,i,d), 0)- self.inv_level.get((i,d,prevt[t]), 0)
                                                                        for d in self.cw_id                 
                                                                        for i in self.intsku_CW.get(d,[])
                                                                        for t in self.time_id[1:last_t[i]]})
        last_t_ = self.minimize_constraint()
        
        constr_dic.update({f"{w, t, i}3rdech_InvBal":LpAffineExpression((*((self.shipment[(f,w,i,prevt[t])], 1) for f in factory_to_rw[w] if i in self.intsku_fact.get(f, [])),
                                                                        *((self.shipment[(d,w,i,prevt[t])], 1) for d in cw_to_rw[w])))  
                                                                         == self.inv_level[(i,w,t)] + self.demand.get((t,i,w), 0)  - self.lost_sales.get((t,i,w), 0) - self.inv_level.get((i,w,prevt[t]), 0)
                                                                            for w in self.rw_id
                                                                            for i in self.intsku_RW.get(w, [])
                                                                            for t in self.time_id[2:last_t_[(w, i)]]})


        constr_dic.update({f"{f,t,i}1stech_ext_sku_InvBal": LpAffineExpression([*((self.shipment[(e,f,i,prevt[t])], 1) for e in self.supplier[i])])
                                                                        == self.demand.get((t,i,f), 0) - self.lost_sales.get((t,i,f), 0) + self.inv_level[(i,f,t)] - self.inv_level.get((i,f,prevt[t]), 0)
                                                                        for f in self.factory_id
                                                                        for i in self.extsku_fact.get(f, [])
                                                                        for t in self.time_id[1:last_t[i]]}) 

        constr_dic.update({f"{d,t,i}2ndech_ext_sku_InvBal": LpAffineExpression((*((self.shipment[(e,d,i,prevt[t])], 1) for e in self.supplier[i]),
                                                                               *((self.shipment[(d,w,i,t)], -1) for w in self.rw_id if d in cw_to_rw[w] and i in self.sku_LOC.get(w, []))))
                                                                        == self.demand.get((t,i,d), 0) - self.lost_sales.get((t,i,d), 0) + self.inv_level[(i,d,t)] - self.inv_level.get((i,d,prevt[t]), 0)
                                                                        for d in self.cw_id
                                                                        for i in self.extsku_CW.get(d,[])
                                                                        for t in self.time_id[1:last_t[i]]})
        # gather indices where SS > 0
        p = [k for k,v in self.cw_ss.items() if v > 0]

        constr_dic.update({f"{d, t, i}2ndech_ssreq":LpAffineExpression([(self.inv_level[(i,d,t)], 1)]) 
                                                                            >= self.cw_ss.get((d, i), 0) - self.lost_sales.get((t,i,d), 0)
                                                                            for d, i in p
                                                                            for t in self.time_id[:last_t[i]]})
        p = [k for k,v in self.rw_ss.items() if v > 0]

        constr_dic.update({f"{w, t, i}3rdech_ssreq":LpAffineExpression([(self.inv_level[(i,w,t)], 1)]) 
                                                                        >= self.rw_ss.get((w,i), 0) - self.lost_sales.get((t,i,w), 0)
                                                                        for w,i in p
                                                                        for t in self.time_id[:last_t[i]]})
        self.inv_model.extend(constr_dic)
        return constr_dic


    def direct_sh_constraint(self):
        constr_dic = {}
        direct_sh_extsku = dict(zip(self.loc_data.index, self.loc_data["direct shipment ext. SKU"]))

        inital_inv = {t: (1 if ind > 0 else 0) for ind, t in enumerate(self.time_id)}
        prevt = {t: self.time_id[max(ind - 1, 0)] for ind, t in enumerate(self.time_id)}
        factory_to_rw, cw_to_rw, fact_to_fact = self.arrival_allocation()
        last_t = self.minimize_constraint()


        constr_dic.update({f"{w,t,i}3rdech_ext_sku_InvBal": LpAffineExpression([*((self.shipment[(e,w,i,prevt[t])], 1) for e in self.supplier[i])])
                                                                    == self.demand.get((t,i,w), 0) - self.lost_sales.get((t,i,w), 0) + self.inv_level[(i,w,t)] - self.inv_level.get((i,w,prevt[t]), 0)*inital_inv[t]
                                                                    for w in self.rw_id if direct_sh_extsku[w] == 1
                                                                    for i in self.extsku_RW.get(w, [])
                                                                    for t in self.time_id[2:last_t[(w, i)]]})

        constr_dic.update({f"{w,t,i}3rdech_ext_sku_InvBal": LpAffineExpression([*((self.shipment[(d,w,i,prevt[t])], 1) for d in cw_to_rw[w])])
                                                                    == self.demand.get((t,i,w), 0) - self.lost_sales.get((t,i,w), 0) + self.inv_level[(i,w,t)] - self.inv_level.get((i,w,prevt[t]), 0)*inital_inv[t]
                                                                    for w in self.rw_id if direct_sh_extsku[w] == 0
                                                                    for i in self.extsku_RW.get(w, [])
                                                                    for t in self.time_id[2:last_t[(w, i)]]})
        self.inv_model.extend(constr_dic)
        return constr_dic

    def min_batch_size_constraint(self, path):
        constr_dic = {}
        sku_min_batch = []
        m = pd.read_csv(self.batch_size_path,
                        index_col=[0],
                        sep=None,
                        engine="python",
                        decimal=self.decimal)
        m = m[m["min_batch_active"] > 0 ] #  only keep min batch size for active skus
        m = m[m.index.isin(self.sku_id)]
        last_t = self.find_last_t()
        min_batch_size = dict(zip(m.index, m["min. Batch Size (PAL)"]))
        min_batch_size = self.minimize_bin(self.data, min_batch_size)
        big_M, big_M_alt = self.compute_big_M(self.data, min_batch_size)
        bin_idx = [(i, t) for i in min_batch_size for t in self.time_id[:last_t[i]]]
        self.prod_on = pulp.LpVariable.dicts("ProdSetup",
                                             bin_idx,
                                             cat="Binary")
        constr_dic.update({f"{i, t}Min_batch_size": LpAffineExpression([(self.production.get((i,f,t),0), 1)])
                                                                        >= self.prod_on[(i,t)] * min_batch_size[i]
                                                                        for i in list(big_M)
                                                                        for f in self.factory_id if i in self.intsku_fact.get(f, [])
                                                                        for t in self.time_id[:last_t[i]]})
        constr_dic.update({f"{i, t}Max_batch_size": LpAffineExpression([(self.production.get((i,f,t),0), 1)]) 
                                                                         <= self.prod_on[(i,t)] * big_M[i]
                                                                        for i in list(big_M)
                                                                        for f in self.factory_id if i in self.intsku_fact.get(f, [])
                                                                        for t in self.time_id[:last_t[i]]})
        constr_dic.update({f"{i, t}batch_size": LpAffineExpression([(self.production.get((i,f,t),0), 1)]) 
                                                                         == self.prod_on[(i,t)] * big_M_alt[i]
                                                                        for i in list(big_M_alt)
                                                                        for f in self.factory_id if i in self.intsku_fact.get(f, [])
                                                                        for t in self.time_id[:last_t[i]]})
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
        last_t = dict(zip(zip(last[self.loc_col], last[self.sku_col]), last[self.time_col]))
        last_t = {k: self.time_id.index(v)
                  for k, v in last_t.items()}
        return last_t

    def find_last_t(self):
        '''
        for each SKU find index of last time period where an order occurs
        returns a dictionary with last time period for each SKU
        {"SKU1": "t1", "SKU2": "t2"}
        '''
        last = self.data.sort_values(by=[self.time_col]).drop_duplicates(subset=[self.sku_col], keep="last")
        last_t = dict(zip(last[self.sku_col], last[self.time_col]))
        last_t = {k: self.time_id.index(v) + 1
                  for k, v in last_t.items()}
        return last_t

    def sum_ss(self, cw_ss, rw_ss, time_skus_dic):
        ss_sum = {}
        # find out for each time period which skus has last demand in time for that time period
        d = {n: [k for k in time_skus_dic.keys() if time_skus_dic[k] == n] for n in set(time_skus_dic.values())}

        for w in self.loc_id:
            for k, v in d.items():
                t = self.time_id[k-1]
                ss_sum[(w, t)] = sum(cw_ss.get((w, sku), rw_ss.get((w, sku), 0)) for sku in v)
        return ss_sum

    def compute_big_M(self, demand, mbs):
        '''
        big_M:
        compute big M as to tighten the bound
        Production will never be > Sum of demand for entire network and all periods
        big_M_alt:
        if sum of the demand is smaller than batch size -> production can only be min. batch size or 0
        -> this allows to reduce solution space
        '''
        # temp = demand.groupby([self.sku_col, self.time_col]).agg({self.qty_col: "sum"})
        # temp = temp[temp[self.qty_col] != 0].groupby([self.sku_col]).agg({self.qty_col: "max"})
        temp = demand.groupby(self.sku_col).agg({self.qty_col: "sum"})
        temp = dict(zip(temp.index, temp[self.qty_col]))
        big_M = {}
        big_M_alt = {}
        for i in mbs.keys():
            # big_M[i] = max((temp[i] + self.cw_ss.get(i, 0) + self.rw_ss.get(i, 0)), mbs[i])
            if temp[i] > mbs[i]:
                big_M[i] = temp[i]
            else:
                big_M_alt[i] = mbs[i] 
        return big_M, big_M_alt

    def compute_ss(self, service_level):
        cent_wh_alloc = {}  # store central warehouse - regional warehouse allocation
        cw_ss = pd.DataFrame()  # store mean and std of demand aggregated per central warehouse and respective reg. warehouse
        rw_ss = pd.DataFrame()
        op = {self.qty_col: ["mean", "std"],
              "lead_time": "first",
              "std_lead_time": "first"}
        for c_wh in self.cw_id:
            # extracting whih central warehosue is responsible for which regional warehouse
            cent_wh_alloc[c_wh] = [c_wh] + self.loc_data.loc[self.loc_data["resp. central WH"] == c_wh].index.tolist()
            # computing demand mean and std accroding to the demand at that cw and the rw it oversees
            stats = self.data[self.data[self.loc_col].isin(cent_wh_alloc[c_wh])].groupby(self.sku_col, as_index=False).agg(op)
            stats[self.loc_col] = c_wh
            cw_ss = cw_ss.append(stats)
        cw_ss.set_index([self.loc_col, self.sku_col], inplace=True)
        cw_ss.columns = ["Pallets_mean", "Pallets_std", "lead_time_mean", "lead_time_std"]  # renaming the columns
        cw_ss["Safety_Stock"] = service_level * (cw_ss["lead_time_mean"] * cw_ss["Pallets_std"]**2
                                      + cw_ss["Pallets_mean"]**2 * cw_ss["lead_time_std"]**2)**0.5
        for r_wh in self.rw_id:
            stats = self.data[self.data[self.loc_col] == r_wh].groupby(self.sku_col, as_index=False).agg(op)
            stats[self.loc_col] = r_wh
            rw_ss = rw_ss.append(stats)
        rw_ss.set_index([self.loc_col, self.sku_col], inplace=True)
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
            factory_to_rw[w] = self.loc_data.loc[w, "factory_allocation"].split(",")
            
            try:
                cw_to_rw[w] = self.loc_data.loc[w, "resp. central WH"].split(",")
            except AttributeError:
                cw_to_rw[w] = []
        for f in self.factory_id:
            try:
                fact_to_fact[f] = self.loc_data.loc[f, "factory_to_factory"].split(",")
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
            dep[f] = self.cw_id + loc.index[loc["factory_allocation"].str.contains(f)].tolist() + loc.index[loc["factory_to_factory"].str.contains(f)].tolist()
        for d in self.cw_id:
            dep[d] = loc.index[loc["resp. central WH"].str.contains(d)].tolist()
        for e in self.ext_factory_id:
            dep[e] = self.data[self.loc_col][self.data[self.prod_col] == e].unique().tolist() + self.cw_id
        return dep

    def bi_obj(self):
        folder = "./Saved Models/"
        model_file = "biobj"
        epsilon = [0.51]
        bi_obj_results = {}
        self.define_indices(loc_data_path=self.loc_data_path)
        self.define_subsets()
        self.define_paramaters(ftl_matrix_path=self.ftl_matrix_path, s_l=NormalDist().inv_cdf(epsilon[-1]))
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
                                                                            for i,w,t in self.ls_idx) <= (1- self.service_level) * self.total_demand})
        self.inv_model.extend({"epsilon": self.service_level >= 0})

        constr_dic.update({"epsilon": self.service_level >= 0})
        self.inv_model.writeLP("biobj.lp")
        self.save_model_json(self.inv_model, constraints=constr_dic)
        while epsilon[-1] < 1.05:
            variables, model = LpProblem.from_json(folder + model_file + ".json")
            with open(folder + model_file + "constraints.json") as f:
                cons = ujson.load(f)
            self.call_cplex(model)
            print(self.inv_model.status)
            bi_obj_results[epsilon[-1]] = value(model.objective)
            epsilon.append(epsilon[-1] + 0.05)
            z_score = NormalDist().inv_cdf(min(epsilon[-1], 0.99999))
            print(z_score)
            self.define_paramaters(ftl_matrix_path=self.ftl_matrix_path, s_l=z_score)
            model.constraints = OrderedDict(zip(cons.keys(), model.constraints.values())) # rename the constraint for easier altering
            model.constraints["epsilon"].constant = epsilon[-1]
            self.update_constraint_rhs(model.constraints)
            self.save_model_json(model, constraints=model.constraints)
        print(bi_obj_results) 
        bi_res = pd.DataFrame.from_dict(bi_obj_results, orient="index")
        bi_res.reset_index(inplace=True)
        bi_res.columns = ["service level", "Costs"]
        v.bi_obj_vis(bi_res)

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

                    constr_dic = {}
                    constr_dic = self.define_constraints()
                    constr_dic.update(self.direct_sh_constraint())
                    mbsc = self.min_batch_size_constraint(self.batch_size_path)
                    constr_dic.update(mbsc)
           
                    self.mc_time = time.time() - start_time
                    print("Model Creation Time:-------------",
                          "\n" + str(self.mc_time))
                    self.save_model_json(model=self.inv_model, constraints=constr_dic, ask=True)

                    start_time = time.time()
                    if self.tune:
                        self.call_cplex_tuning_tool(self.inv_model)
                        break
                    else:
                        if self.config_dict["cplex"]["cplex"]:
                            self.call_cplex(self.inv_model)
                            self.solv_time = time.time() - start_time
                        if self.config_dict["gurobi"]["gurobi"]:
                            self.call_gurobi(self.inv_model)
                        print(f"number of variables: {self.inv_model.numVariables()}")
                        if bool(self.config_dict["debugging"]["write_lp"]):
                            self.inv_model.writeLP("model.LP")
                        if bool(self.config_dict["debugging"]["write_lp"]):
                            self.inv_model.writeMPS("model.mps")
                        
                        print("Objective value: ", value(self.inv_model.objective))
                        print("Inventory level:--------------")
                        self.inv_result = self.export_vars_3d(indices=self.inv_idx,
                                                              variable=self.inv_level,
                                                              filename="Inventory Level")
                        self.export_inv(indices=self.inv_idx,
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
                        # self.export_vars_2d(indices=self.slack_ind,
                        #                     variable=self.slack,
                        #                     filename="Slack")
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
            solver.solverModel.parameters.emphasis.memory.set(int(param["memory_emphasis"]))
            solver.solverModel.parameters.workmem.set(int(param["working_memory"]))
            solver.solverModel.parameters.mip.strategy.file.set(int(param["node_file"]))
            solver.solverModel.parameters.mip.cuts.cliques.set(int(param("cuts_clique")))
            solver.solverModel.parameters.mip.cuts.covers.set(int(param("cuts_covers")))
            solver.solverModel.parameters.mip.cuts.flowcovers.set(int(param("cuts_flowcovers")))
            solver.solverModel.parameters.mip.cuts.gomory.set(int(param("cuts_gomory")))
            solver.solverModel.parameters.mip.cuts.gubcovers.set(int(param("cuts_gubcovers")))
            solver.solverModel.parameters.mip.cuts.implied.set(int(param("cuts_implied")))
            solver.solverModel.parameters.mip.cuts.mircut.set(int(param("cuts_mircut")))
            solver.solverModel.parameters.mip.cuts.pathcut.set(int(param("cuts_path")))
            solver.solverModel.parameters.mip.limits.cutsfactor.set(int(param("cuts_factor")))
            solver.solverModel.parameters.mip.strategy.branch.set(int(param("branch_strategy")))
            solver.solverModel.parameters.mip.strategy.probe.set(int(param("strategy_probe")))
            solver.solverModel.parameters.mip.tolerances.mipgap.set(float(param["mipgap"]))
            solver.callSolver(model)
        except cplex.exceptions.errors.CplexSolverError:
            print("One of the Cplex parameters specified is invalid. ")
        status = solver.findSolutionValues(model)
        solver.solverModel.parameters.conflict.display.set(2)  # if model is unsolvable will display the problematic constraint(s)

    def call_cplex_tuning_tool(self, model):
        param = self.config_dict["cplex"]

        solver = CPLEX_PY()
        solver.buildSolverModel(model)
        c = solver.solverModel
        ps = [(c.parameters.workmem, int(param["working_memory"])),
              (c.parameters.mip.tolerances.mipgap, float(param["mipgap"])),
              (c.parameters.emphasis.memory, int(param["memory_emphasis"]))]

        c.parameters.tune.display.set(3)
        c.parameters.tune.timelimit.set(int(self.config_dict["tuning tool"]))
        m = solver.solverModel.parameters.tune_problem(ps)
        if m == solver.solverModel.parameters.tuning_status.completed:
            print("modified parameters: ")
            for param, value in solver.solverModel.parameters.get_changed():
                print(f"{repr(param)}: {value}")

        else:
            print("tuning status was: " + str(solver.solverModel.parameters.tuning_status[status]))

    def call_gurobi(self, model):
        solver = GUROBI(epgap=0.05, cuts=2, presolve=1)
        solver.buildSolverModel(model)
        solver.callSolver(model)
        status = solver.findSolutionValues(model)

    def export_vars_2d(self, indices, variable, filename):
        ''' Formats the solution and returns it embedded in a pandas dataframe
        indices: list of indices over which the variable is defined
        variable: variable name to be exported
        filename: under which name to save the varaible to be exported to csv
        '''
        start_time = time.time()
        dic = {"time": [], "warehouse": [], "value": []}
        for w, t in indices:
            dic["time"].append(t)
            dic["warehouse"].append(w)
            dic["value"].append(variable[(w, t)].varValue)

        self.var_df = pd.DataFrame.from_dict(dic)
        self.var_df = self.var_df.fillna(0)

        self.var_df.to_excel(self.writer, sheet_name=filename)
        self.var_df.to_csv(f"./CSV export files/{filename}.csv")
        self.var_df.loc[self.var_df["value"] < 0, "value"] = 0
        print(self.var_df)
        print("--- %s seconds ---" % (time.time() - start_time))
        print(f"{filename} exported")
        return self.var_dff

    def export_vars_3d(self, indices, variable, filename):
        ''' Formats the solution and returns it embedded in a pandas dataframe
        indices: list of indices over which the variable is defined
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
            if filename == "Lost Sales":
                self.var_df.columns = ["warehouse", "SKU_id", "time", "value"]
        except AttributeError:
            pass
        except NameError: 
            pass
        self.var_df.to_excel(self.writer, sheet_name=filename, float_format="%.2f")
        self.var_df.to_csv(f"./CSV export files/{filename}.csv")
        self.var_df.loc[self.var_df["value"] < 0, "value"] = 0
        print(self.var_df)
        print("--- %s seconds ---" % (time.time() - start_time))
        print(f"{filename} exported")
        return self.var_df


    def export_inv(self, indices, variable, filename):
        ''' Formats the solution and returns it embedded in a pandas dataframe
        indices: list of indices over which the variable is defined
        variable: variable name to be exported
        filename: under which name to save the varaible to be exported to csv
        '''
        start_time = time.time()
        dic = {"time": [], "warehouse": [], "product": [], "value": []}
        for i, w, t in indices:
            dic["time"].append(t)
            dic["warehouse"].append(w)
            dic["product"].append(i)
            dic["value"].append(variable[(i,w,t)].varValue)

        self.var_df = pd.DataFrame.from_dict(dic)
        self.var_df = self.var_df.set_index(["warehouse", "product", "time"])
        ind = [(w, i, t) for w in self.loc_id for i in self.sku_LOC.get(w, []) for t in self.time_id]  # create complete indices (including all time periods)
        ind = pd.Index(ind).difference(self.var_df.index)  # extract missing indices
        add_df = pd.DataFrame(index=ind, columns=self.var_df.columns)
        self.var_df = pd.concat([self.var_df, add_df])  # merge results with missing indices
        self.var_df.sort_index(inplace=True)
        self.var_df.fillna(method="ffill", inplace=True)  # fill the values with the last available
        self.var_df.reset_index(inplace=True)
        self.var_df.to_excel(self.writer, sheet_name=filename)
        self.var_df.to_csv(f"./CSV export files/{filename}.csv")
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

    def backfill(self, df):
        ind = pd.MultiIndex.from_product([self.time_id, df["warehouse"], df["product"]])
        df = df.reindex(ind, method="bfill")
        print(df)

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

start_time = time.time()
I = InventoryModel(sku_col="SKU_id",
                   time_col="period",
                   loc_col="location",
                   qty_col="pallets",
                   prod_col="producedBy")
if I.biobj:
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



time_taken = time.time() - start_time
print("Total Time: ", time_taken)
   



