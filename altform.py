import pandas as pd
import numpy as np
from pulp import *
import ipdb
import time
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

    def __init__(self, data_path, sku_col, time_col, loc_col, qty_col, prod_col, service_level):
        self.sku_col = sku_col
        self.time_col = time_col
        self.loc_col = loc_col
        self.qty_col = qty_col
        self.prod_col = prod_col
        self.raw_data = pd.read_csv(data_path,
                                    index_col=False,
                                    nrows=500)

        self.service_level = service_level
        # column's values are converted to datetime_col objects to facilitate weekly aggregation
        # self.raw_data[time_col] = pd.to_datetime(self.raw_data[time_col])

        # remove orders that do not have a location specified
        self.data = self.raw_data[self.raw_data[loc_col].notna()]
         # privremeno
        # v = self.raw_data[sku_col].value_counts()
        # self.raw_data = self.raw_data[self.raw_data[sku_col].isin(
        #     v.index[v.gt(100)])]

        print(self.data.groupby(sku_col)[
              self.qty_col].agg(Mean='mean', Sum='sum', Size="size"))
        # self.data.to_csv("demand_input.csv")
        self.inv_model = pulp.LpProblem("Inventory_Optimization",
                                        LpMinimize)  # creates an object of an LpProblem (PuLP) to which we will assign variables, constraints and objective function
        

    def define_indices(self, loc_data_path):
        ''' Returns the unique values for indices from the input data
        Required keywords argument:
        new_time_col: string of the aggregation basis for time (e.g. "week", "month",...)
        '''
        self.loc_data = pd.read_csv(loc_data_path,
                            index_col=[0])
        self.sku_id = self.data[self.sku_col].unique()
        self.time_id = self.data["period"].unique()
        self.loc_id = self.loc_data.index.tolist()
        self.factory_id = self.loc_data[self.loc_data["Echelon"]
                                        == "Factory"].index.tolist()
        self.dc_id = self.loc_data[self.loc_data["Echelon"]
                                   == "DC"].index.tolist()
        self.wh_id = self.loc_data[self.loc_data["Echelon"]
                                   == "Warehouse"].index.tolist()
        self.ext_factory_id = list(set(self.data[self.prod_col].unique().tolist()) - set(self.factory_id))
        # self.loc_data = self.loc_data.loc[self.loc_id]
        # self.FTL_matrix = self.FTL_matrix.drop("CoP NL", axis=1) #temp
        print(f"Number of product: {len(self.sku_id)}" + "\n",
              f"Number of locations: {len(self.loc_id)}" + "\n", 
              f"Number of periods: {len(self.time_id)}" + "\n",
              f"Number of orders: {len(self.data)}")

    def define_paramaters(self, ftl_matrix):
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
        self.FTL_matrix = pd.read_csv(ftl_matrix,
                                      index_col=[0])
        self.lt_df = self.data.groupby(["sh_ItemId"])[
            ["lead_time", "std_lead_time"]].first().fillna(0)
        self.lt_df["var_lead_time"] = self.lt_df["std_lead_time"] ** 2
        self.holding_costs = self.data.groupby(
            ["sh_ItemId", "new_shOrigin"])["hold_cost_pallet"].first()

        # assign which factory produces which sku
        k = (self.data[self.data[self.prod_col]
             .isin(self.factory_id)]
             .groupby(self.prod_col)[self.sku_col]
             .apply(set).apply(list))
        self.prod_plan = dict(zip(k.index, k))

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
            self.f2f_sku["ITT"] = list(itt[0])

        if len(itt) == 0 and len(brn) == 0:
            self.f2f_sku = {}

        #assign which supplier supplies which sku
        k = (self.data[self.data[self.prod_col]
             .isin(self.ext_factory_id)]
             .groupby(self.sku_col)[self.prod_col]
             .apply(set).apply(list))
        self.supplier = dict(zip(k.index, k))

        # asssign which sku can be supplied by which factory
        k = (self.data[self.data[self.prod_col]
             .isin(self.factory_id)]
             .groupby(self.sku_col)[self.prod_col]
             .apply(set).apply(list))
        self.sku_plan = dict(zip(k.index, k))

        # subset of skus produced internally
        self.int_skus = (self.data[self.data[self.prod_col]
                               .isin(self.factory_id)][self.sku_col]
                               .to_list())

        # isolate demand for external skus
        self.ext_skus = (self.data[self.data[self.prod_col]
                               .isin(self.ext_factory_id)][self.sku_col]
                               .to_list())
        # extract demand to dictionary
        self.demand = self.data.set_index([self.time_col,
                                          self.sku_col,
                                          self.loc_col])[self.qty_col].to_dict()



        # a = self.data[(self.data["ProducedBy"].isin(self.factory_id) & self.data["new_shOrigin"].isin(self.factory_id))]
        # a = a[a["ProducedBy"] != a["new_shOrigin"]]
        # a.to_csv("fact_to_fact_orders.csv")

    def define_variables(self):
        '''Defines the variable and their nature whcih are then added to the model
        '''

        self.inv_level = pulp.LpVariable.dicts("inventory level",
                                               (self.sku_id,
                                                self.loc_id,
                                                self.time_id),
                                               lowBound=0)
        # self.max_stock = pulp.LpVariable.dicts("maxium stock level",
        #                                        (self.sku_id,
        #                                         self.loc_id,
        #                                         self.time_id),
        #                                        lowBound=0)
        self.shipment = pulp.LpVariable.dicts("shipments",
                                              (self.loc_id + self.ext_factory_id,
                                               self.loc_id,
                                               self.sku_id,
                                               self.time_id),
                                              lowBound=0)
        self.FTL = pulp.LpVariable.dicts("FTL",
                                         (self.factory_id + self.dc_id,
                                          self.dc_id + self.wh_id,
                                          self.time_id),
                                         lowBound=0)
        self.production = pulp.LpVariable.dicts("production",
                                                (self.sku_id,
                                                 self.factory_id,
                                                 self.time_id),
                                                lowBound=0)
        self.lost_sales = pulp. LpVariable.dicts("lost sales",
                                                (self.sku_id,
                                                self.loc_id,
                                                self.time_id),
                                                lowBound=0)
        # self.slack = pulp.LpVariable.dicts("slack",
        #                                    (self.sku_id,
        #                                     self.loc_id,
        #                                     self.time_id),
        #                                    lowBound=0)

    def define_objective(self):
        ''' Defines the objective funciton
        '''
        start_time = time.time()
        hc = self.holding_costs.to_dict()
        self.loc_da = self.loc_data.to_dict()        
        ftl = self.FTL_matrix.to_dict()
        default_hc = self.holding_costs.groupby("sh_ItemId").mean().to_dict()

        holding_costs = LpAffineExpression(((self.inv_level[i][w][t], hc.get((i, w), default_hc[i]))
                                            for i in self.sku_id
                                           for w in self.loc_id
                                           for t in self.time_id))

        trans_costs_echelon = LpAffineExpression(((self.FTL[o][d][t], ftl[d][o])
                                     for o in self.factory_id + self.dc_id
                                     for d in self.dc_id + self.wh_id
                                     for t in self.time_id))

        production_costs = LpAffineExpression(((self.production[i][f][t], self.loc_da["Prod. Costs"][f])
                                                for i in self.sku_id
                                                for f in self.factory_id
                                                for t in self.time_id))
        shortage_costs = LpAffineExpression((self.lost_sales[i][w][t], 999999)
                                            for i in self.sku_id
                                            for w in self.loc_id
                                            for t in self.time_id)
        # slack_costs = lpSum((self.slack[i][w][t] * 50000
        #                      for i in self.sku_id
        #                      for w in self.loc_id
        #                      for t in self.time_id))

        self.inv_model += holding_costs + \
            trans_costs_echelon + production_costs + shortage_costs  # + slack_costs
        print("--- %s seconds ---" % (time.time() - start_time))
        print("objective defined")

    def define_constraints(self):
        '''Defines the constraints to be added to the model
        '''
        start_time = time.time()

        factory_to_rw, cw_to_rw, fact_to_fact = self.routings_allocation()
        constr_dic = {}
        cw_ss, rw_ss = self.compute_ss(self.service_level)

        cw_ss, rw_ss = cw_ss["Safety_Stock"].fillna(0).to_dict(), rw_ss["Safety_Stock"].fillna(0).to_dict()


                            
        constr_dic.update({f"{f,t}ProdCap": LpAffineExpression(((self.production[i][f][t], 1)
                                              for i in self.prod_plan.get(f, []))) <= self.loc_da["Prod. Cap."][f]
                                                                                    for f in self.factory_id
                                                                                    for t in self.time_id})

        constr_dic.update({f"{w,t}HoldCap": LpAffineExpression(((self.inv_level[x][w][t], 1)
                                                                  for x in self.sku_id)) <= self.loc_da["Hold. Cap."][w] 
                                                                    for w in self.loc_id 
                                                                    for t in self.time_id})

        constr_dic.update({f"{o,d,t}FTL":LpAffineExpression(((self.shipment[o][d][i][t] ,1)
                                                                            for i in self.sku_id)) == 33 * self.FTL[o][d][t]
                                                            for o in self.factory_id + self.dc_id
                                                            for d in self.dc_id + self.wh_id
                                                            for t in self.time_id})

        lt_dic = self.data["lead_time"].to_dict()
        lt = {(i,t): self.time_id[max(int(
                ind - lt_dic.get(i,1)), 0)] for i in self.sku_id
                                                           for ind, t in enumerate(self.time_id)}
        inital_inv = {t:( 1 if ind>0 else 0) for ind, t in enumerate(self.time_id)}
        prevt = {t :self.time_id[max(ind - 1, 0)]  for ind, t in enumerate(self.time_id)}

        constr_dic.update({f"{f,t, i}1stech_InvBal":LpAffineExpression(((self.production[i][f][lt[i,t]], 1),
                                                                        *((self.shipment[f][dc][i][t], -1) for dc in self.dc_id),
                                                                        *((self.shipment[f][w][i][t], -1) for w in self.wh_id)))
                                                                        == self.inv_level[i][f][t] + self.demand.get((t,i,f), 0) - self.lost_sales[i][f][t] - self.inv_level[i][f][prevt[t]]*inital_inv[t]
                                                                        for f in self.factory_id
                                                                        for i in self.prod_plan.get(f, [])
                                                                        for t in self.time_id})

        constr_dic.update({f"{x,t,i}1stehch_facttofact_InvBal": LpAffineExpression([*((self.shipment[f][x][i][prevt[t]], 1) for f in self.sku_plan[i])]) 
                                                                                    == self.inv_level[i][x][t] + self.demand.get((t,i,x), 0) - self.lost_sales[i][x][t] - self.inv_level[i][x][prevt[t]]*inital_inv[t]
                                                                                    for x in self.f2f_sku.keys()
                                                                                    for i in self.f2f_sku[x]
                                                                                    for t in self.time_id})

        constr_dic.update({f"{d,t, i}2ndech_InvBal":LpAffineExpression((*((self.shipment[f][d][i][prevt[t]], 1) for f in self.sku_plan[i]),
                                                                        *((self.shipment[d][w][i][t], -1) for w in self.wh_id if d in cw_to_rw[w])))
                                                                        == self.inv_level[i][d][t] + self.demand.get((t,i,d), 0) + cw_ss.get((d, i), 0) - self.lost_sales[i][d][t]- self.inv_level[i][d][prevt[t]]*inital_inv[t]
                                                                        for i in self.int_skus
                                                                        for d in self.dc_id
                                                                        for t in self.time_id})

        constr_dic.update({f"{w, t, i}3rdech_InvBal":LpAffineExpression((*((self.shipment[f][w][i][prevt[t]], 1) for f in factory_to_rw[w]),
                                                                        *((self.shipment[d][w][i][prevt[t]], 1) for d in cw_to_rw[w])))  
                                                                         == self.inv_level[i][w][t] + self.demand.get((t,i,w), 0) + rw_ss.get((w,i), 0) - self.lost_sales[i][w][t] - self.inv_level[i][w][prevt[t]]*inital_inv[t]
                                                                            for i in self.int_skus
                                                                            for w in self.wh_id
                                                                            for t in self.time_id})

        constr_dic.update({f"{f,t,i}1stech_ext_sku_InvBal": LpAffineExpression([*((self.shipment[e][f][i][prevt[t]], 1) for e in self.supplier[i])])
                                                                        == self.demand.get((t,i,f), 0) - self.lost_sales[i][f][t] + self.inv_level[i][f][t] - self.inv_level[i][f][prevt[t]]*inital_inv[t]
                                                                        for i in self.ext_skus
                                                                        for f in self.factory_id
                                                                        for t in self.time_id}) # teest

        constr_dic.update({f"{d,t,i}2ndech_ext_sku_InvBal": LpAffineExpression((*((self.shipment[e][d][i][prevt[t]], 1) for e in self.supplier[i]),
                                                                               *((self.shipment[d][w][i][t], -1) for w in self.wh_id if d in cw_to_rw[w])))
                                                                        == self.demand.get((t,i,d), 0) + cw_ss.get((d, i), 0) - self.lost_sales[i][d][t] + self.inv_level[i][d][t] - self.inv_level[i][d][prevt[t]]*inital_inv[t]
                                                                        for i in self.ext_skus
                                                                        for d in self.dc_id
                                                                        for t in self.time_id})


        constr_dic.update({f"{w,t,i}3rdech_ext_sku_InvBal": LpAffineExpression((*((self.shipment[e][w][i][prevt[t]], 1) for e in self.supplier[i]),
                                                                               *((self.shipment[d][w][i][prevt[t]], 1) for d in cw_to_rw[w])))
                                                                        == self.demand.get((t,i,w), 0) + rw_ss.get((w,i), 0) - self.lost_sales[i][w][t] + self.inv_level[i][w][t] - self.inv_level[i][w][prevt[t]]*inital_inv[t]
                                                                        for i in self.ext_skus
                                                                        for w in self.wh_id
                                                                        for t in self.time_id})

        constr_dic.update({f"{d, t, i}2ndech_ssreq":LpAffineExpression([(self.inv_level[i][d][t], 1)]) 
                                                                            >= cw_ss.get((d, i), 0) - self.lost_sales[i][d][t]
                                                                            for i in self.sku_id
                                                                            for d in self.dc_id
                                                                            for t in self.time_id})

        constr_dic.update({f"{w, t, i}3rdech_ssreq":LpAffineExpression([(self.inv_level[i][w][t], 1)]) 
                                                                        >= rw_ss.get((w,i), 0) - self.lost_sales[i][w][t]
                                                                        for i in self.sku_id
                                                                        for w in self.wh_id
                                                                        for t in self.time_id})
        self.inv_model.extend(constr_dic)

        print("--- %s seconds ---" % (time.time() - start_time))
        print("constraints defined")

    def weighted_avg(self, time_ind, wh_ind, prod_ind):
        '''Computes  the rolling distribution of the demand for each location in order to allocation the safety stock and returns the weight
        time_ind: time index for which the computation is being performed
        wh_ind: location index for which the computation is being performed
        prod_ind: product (group) index for which the computation is being performed
        '''
        try:
            weight = self.data.loc[:time_ind, prod_ind, wh_ind].sum(
            ) / self.data.loc[:time_ind, prod_ind, :].sum()
        except KeyError:
            return 0
        return weight.values

    def compute_ss(self, service_level):
        cent_wh_alloc = {} # store central warehouse - regional warehouse allocation
        ss_cw = {} # central warehouse ss
        cw_ss = pd.DataFrame() # store mean and std of demand aggregated per central warehouse and respective reg. warehouse
        rw_ss = pd.DataFrame()
        op = {self.qty_col:["mean", "std"],
              "lead_time": "first",
              "std_lead_time": "first"}
        for c_wh in self.dc_id:
            # extracting whih central warehosue is responsible for which regional warehouse
            cent_wh_alloc[c_wh] = [c_wh] + self.loc_data.loc[self.loc_data["Resp. Central WH"] == c_wh].index.tolist()
            # computing demand mean and std accroding to the allocation
            stats = self.data[self.data["new_shOrigin"].isin(cent_wh_alloc[c_wh])].groupby("sh_ItemId", as_index=False).agg(op)
            stats["Location"] = c_wh
            cw_ss = cw_ss.append(stats)
        cw_ss.set_index(["Location", "sh_ItemId"], inplace=True)
        cw_ss.columns = ["Pallets_mean", "Pallets_std", "lead_time_mean", "lead_time_std"] # renaming the columns
        cw_ss["Safety_Stock"] = self.service_level * (cw_ss["lead_time_mean"] * cw_ss["Pallets_std"]**2 
                                      + cw_ss["Pallets_mean"]**2 * cw_ss["lead_time_std"]**2)**0.5
        
        for r_wh in self.wh_id:
            stats = self.data[self.data["new_shOrigin"] == r_wh].groupby("sh_ItemId", as_index=False).agg(op)
            stats["Location"] = r_wh
            rw_ss = rw_ss.append(stats)
        rw_ss.set_index(["Location", "sh_ItemId"], inplace=True)
        rw_ss.columns = ["Pallets_mean", "Pallets_std", "lead_time_mean", "lead_time_std"] # renaming the columns
        rw_ss["Safety_Stock"] = self.service_level * (rw_ss["lead_time_mean"] * rw_ss["Pallets_std"]**2)

        return cw_ss, rw_ss

    def routings_allocation(self):
        factory_to_rw = {}
        cw_to_rw = {}
        fact_to_fact = {}
        for w in self.wh_id:
            factory_to_rw[w] = self.loc_data.loc[w, "Factory_allocation"].split(",")
            
            try:
                cw_to_rw[w] = self.loc_data.loc[w, "Resp. Central WH"].split(",")
            except AttributeError:
                cw_to_rw[w]  = []
        for f in self.factory_id:
            try:
                fact_to_fact[f] = self.loc_data.loc[f, "Factory_to_Factory"].split(",")
            except AttributeError:
                fact_to_fact[f] = []

        return factory_to_rw, cw_to_rw, fact_to_fact

    def build_model(self):
        ''' calls all the required function to initialize, solve and export the model
        '''
        self.define_indices(loc_data_path="./CSV input files/wh_data_edited.csv")
        self.define_paramaters(ftl_matrix="./CSV input files/FTLmatrix.csv")
        
        start_time = time.time()
        self.define_variables()
        self.define_objective()
        self.define_constraints()
        self.mc_time = time.time() - start_time

        start_time = time.time()
        solver = CPLEX_PY()
        # self.inv_model.solve(solver)
        solver.buildSolverModel(self.inv_model)
        solver.solverModel.parameters.emphasis.memory.set(1)
        solver.solverModel.parameters.workmem.set(6144)
        solver.callSolver(self.inv_model)
        status = solver.findSolutionValues(self.inv_model)
        solver.solverModel.parameters.conflict.display.set(2)   # in case the model is not solvable display which contraint(s) are conflicting:
        self.solv_time = time.time() - start_time

        # self.inv_model.writeLP("test.LP")
        # self.inv_model.writeMPS("test.mps")
        print("Objective value: ", value(self.inv_model.objective))
        print("Inventory level:--------------")
        self.export_vars_3d(time_ind=self.time_id,
                            wh_ind=self.loc_id,
                            prod_ind=self.sku_id,
                            variable=self.inv_level,
                            filename="Inventory Level")
        print("Shipments:------------------")
        self.export_vars_4d(time_ind=self.time_id,
                            origin_ind=self.loc_id + self.ext_factory_id,
                            destination_ind=self.loc_id,
                            prod_ind=self.sku_id,
                            variable=self.shipment,
                            filename="Shipment")

        print("Production:-----------------")
        self.export_vars_3d(time_ind=self.time_id,
                            wh_ind=self.factory_id,
                            prod_ind=self.sku_id,
                            variable=self.production,
                            filename="Production")
        print("Lost Sales:-----------------")
        self.export_vars_3d(time_ind=self.time_id,
                            wh_ind=self.factory_id,
                            prod_ind=self.sku_id,
                            variable=self.lost_sales,
                            filename="Lost Sales")

        # self.export_vars_3d(time_ind=self.time_id,
        #                     wh_ind=self.loc_id,
        #                     prod_ind=self.sku_id,
        #                     variable=self.slack,
        #                     filename="Slack")

    def export_vars_3d(self, time_ind, wh_ind, prod_ind, variable, filename):
        ''' Formats the solution and returns it embedded in a pandas dataframe
        time_ind: time index used by the variable
        wh_ind: location index used by the varaible
        prod_ind: product (group) index used by the variable
        variable: variable name to be exported
        filename: under which name to save the varaible to be exported to csv
        '''
        #dic domprehension
        start_time = time.time()
        dic = {"time": [], "warehouse": [], "product": [], "value": []}
        for t in time_ind:
            for w in wh_ind:
                for i in prod_ind:
                    dic["time"].append(t)
                    dic["warehouse"].append(w)
                    dic["product"].append(i)
                    dic["value"].append(variable[i][w][t].varValue)

        self.var_df = pd.DataFrame.from_dict(dic)
        self.var_df = self.var_df.fillna(0)
        self.var_df.to_csv(filename + ".csv")
        print(self.var_df)
        print("--- %s seconds ---" % (time.time() - start_time))
        print(f"{filename} exported")
                   
    def export_vars_4d(self, time_ind, origin_ind, destination_ind, prod_ind, variable, filename):
        start_time = time.time()
        dic = {"time": [], "origin": [],
               "destination": [], "product": [], "value": []}
        for t in time_ind:
            for o in origin_ind:
                for d in destination_ind:
                    for i in prod_ind:
                        if variable[o][d][i][t].varValue is not None:
                            if variable[o][d][i][t].varValue > 0:
                                dic["origin"].append(o)
                                dic["destination"].append(d)
                                dic["product"].append(i)
                                dic["time"].append(t)
                                dic["value"].append(variable[o][d][i][t].varValue)

        self.var_df = pd.DataFrame.from_dict(dic)
        self.var_df = self.var_df.fillna(0)
        self.var_df.to_csv(filename + ".csv")
        self.FTL_df = self.var_df.groupby(["origin", "destination", "time"], as_index=True).sum()
        self.FTL_df.value = self.FTL_df.value / 33
        self.FTL_df.value = np.ceil(self.FTL_df.value)
        self.FTL_df.to_csv("FTL.csv")
        print(self.var_df)
        print("FTL----------------------")
        print(self.FTL_df)
        print("--- %s seconds ---" % (time.time() - start_time))
        print(f"{filename} exported")


# Actually creating an instance of the classes in which the model is defined
s = 1.95

t = 53
mctime_dic = {}
solvtime_dic = {}

start_time = time.time()
 
I = InventoryModel(data_path="./CSV input files/orders_newloc.csv",
                   sku_col="sh_ItemId",
                   time_col="period",
                   loc_col="new_shOrigin",
                   qty_col="Pallets",
                   prod_col="ProducedBy",
                   service_level=s)
I.build_model()


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


'''
TODO:
- automate time index
- improve export functions (make indices argmuent list like)
- generate default holding costs
- default value for holding costs
'''

