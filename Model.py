import pandas as pd
from pulp import *
import ipdb
import random
import time


class InventoryModel():
    """class used to define and construct the inventory linear program
       data_path: the file path of the data containing the orders
       product_col: string of the column's header containing the product (group) id
       time_col: string of the column's header containing the date of the order
       loc_col: string of the column's header containing the location of the order
       qty_col: string of the column's header containing the quantity of the order
    """

    def __init__(self, data_path, product_col, time_col, loc_col, qty_col, service_level):
        self.product_col = product_col
        self.time_col = time_col
        self.loc_col = loc_col
        self.qty_col = qty_col

        self.raw_data = pd.read_csv(data_path,
                                    nrows=100)
        self.service_level = service_level
        # column's values are converted to datetime_col objects to facilitate weekly aggregation
        self.raw_data[time_col] = pd.to_datetime(self.raw_data[time_col])

        # remove orders that do not have a location specified
        self.data = self.raw_data[self.raw_data[loc_col].notna()]
        self.data = self.data.set_index([self.product_col, self.time_col, self.loc_col])

        # privremeno
        # v = self.raw_data[product_col].value_counts()
        # self.raw_data = self.raw_data[self.raw_data[product_col].isin(
        #     v.index[v.gt(1000)])]


        print(self.data.groupby(product_col)[
              self.qty_col].agg(Mean='mean', Sum='sum', Size="size"))
        self.data.to_csv("demand_input.csv")
        self.inv_model = pulp.LpProblem("Inventory_Optimization",
                                        LpMinimize)  # creates an object of an LpProblem (PuLP) to which we will assign variables, constraints and objective function

    def define_indices(self):
        ''' Returns the unique values for indices from the input data
        Required keywords argument:
        new_time_col: string of the aggregation basis for time (e.g. "week", "month",...)
        '''
        self.prod_id = self.data.index.get_level_values(
            self.product_col).unique().tolist()
        self.time_id = self.data.index.get_level_values(
            self.time_col).unique().tolist()
        self.loc_id = self.loc_data.index
        self.factory_id = self.loc_data[self.loc_data["Echelon"]
                                        == "Factory"].index.tolist()
        self.dc_id = self.loc_data[self.loc_data["Echelon"]
                                   == "DC"].index.tolist()
        self.wh_id = self.loc_data[self.loc_data["Echelon"]
                                   == "Warehouse"].index.tolist()


        # self.loc_data = self.loc_data.loc[self.loc_id]
        # self.FTL_matrix = self.FTL_matrix.drop("CoP NL", axis=1) #temp
        print(f"Number of product: {len(self.prod_id)}" + "\n",
                        f"Number of locations: {len(self.loc_id)}" + "\n",
                        f"Number of periods: {len(self.time_id)}")

    def define_paramaters(self, loc_data_path, ftl_matrix):
        ''' Returns a dataframe of the paramaters required for the model
        loc_data_path: string of the path to the location data
        '''
        self.loc_data = pd.read_csv(loc_data_path,
                                    index_col=[0])

        self.demand_stats = self.data.groupby(
            self.product_col)[self.qty_col].agg(["size", "mean", "std"]).fillna(0)
        self.demand_stats["var"] = self.demand_stats["std"]**2
        # self.demand_stats.fillna(0, inplace=True)
        self.demand_stats.to_csv("Demand Stats.csv")
        # self.leadt_data = self.raw_data([self.product_col, self.loc_col])["Effective Lead Time [weeks]" , ""].first()
        # self.holding_costs =self.raw_data([])
        self.FTL_matrix = pd.read_csv(ftl_matrix,
                                      index_col=[0])
        self.lt_df = self.data.groupby(["sh_ItemId"])[
            ["lead_time", "std_lead_time"]].first()
        self.lt_df["var_lead_time"] = self.lt_df["std_lead_time"] ** 2
        self.holding_costs = self.data.groupby(
            ["sh_ItemId", "sh_OriginLocationMasterLocation"])["hold_cost_pallet"].first()

        self.ss_df = self.compute_ss(self.service_level)
        
    def define_variables(self):
        '''Defines the variable and their nature whcih are then added to the model
        '''
        self.inv_level = pulp.LpVariable.dicts("inventory level",
                                               (self.prod_id,
                                                self.loc_id,
                                                self.time_id),
                                               lowBound=0)
        # self.max_stock = pulp.LpVariable.dicts("maxium stock level",
        #                                        (self.prod_id,
        #                                         self.loc_id,
        #                                         self.time_id),
        #                                        lowBound=0)
        self.shipment = pulp.LpVariable.dicts("shipments",
                                              (self.loc_id,
                                               self.loc_id,
                                               self.prod_id,
                                               self.time_id),
                                              lowBound=0)
        self.FTL = pulp.LpVariable.dicts("FTL",
                                         (self.factory_id + self.dc_id,
                                          self.dc_id + self.wh_id,
                                          self.time_id),
                                         cat="Integer",
                                         lowBound=0)
        self.production = pulp.LpVariable.dicts("production",
                                                (self.prod_id,
                                                 self.factory_id,
                                                 self.time_id),
                                                lowBound=0)

        # self.slack = pulp.LpVariable.dicts("slack",
        #                                    (self.prod_id,
        #                                     self.loc_id,
        #                                     self.time_id),
        #                                    lowBound=0)

    def define_objective(self):
        ''' Defines the objective funciton
        '''
        start_time = time.time()
        holding_costs = lpSum([self.inv_level[i][w][t] * self.holding_costs.loc[i, w]
                               for i in self.prod_id
                               for w in self.loc_id
                               for t in self.time_id
                               if self.data.index.isin([(t, i, w)]).any()])

        trans_costs_echelon1 = lpSum([self.FTL[o][d][t] * self.FTL_matrix.loc[o, d]
                                      for o in self.factory_id + self.dc_id
                                      for d in self.dc_id + self.wh_id
                                      for t in self.time_id])

        production_costs = lpSum([self.production[i][f][t] * self.loc_data.loc[f, "Prod. Costs"]
                                  for i in self.prod_id
                                  for f in self.factory_id
                                  for t in self.time_id])

        # slack_costs = lpSum((self.slack[i][w][t] * 50000
        #                      for i in self.prod_id
        #                      for w in self.loc_id
        #                      for t in self.time_id))

        self.inv_model += holding_costs + \
            trans_costs_echelon1 + production_costs  # + slack_costs
        print("--- %s seconds ---" % (time.time() - start_time))
        print("objective defined")

    def define_constraints(self):
        '''Defines the constraints to be added to the model
        '''
        start_time = time.time()

        for f in self.factory_id:
            for ind, t in enumerate(self.time_id):
                self.inv_model += lpSum([self.inv_level[i][f][t]
                                         for i in self.prod_id]) <= self.loc_data.loc[f, "Hold. Cap."]
                for i in self.prod_id:
                    lt = self.time_id[max(int(
                        ind - self.lt_df.loc[i, "lead_time"]), 0)]
                    self.inv_model += lpSum([self.production[i][f][t]
                                             for i in self.prod_id]) <= self.loc_data.loc[f, "Prod. Cap."]
                    prevt = self.time_id[ind - 1]
                    self.inv_model += lpSum([self.inv_level[i][f][prevt] -
                                             self.data["Pallets"].get(
                                                 (t, i, f), 0)
                                             - self.shipment[f][dc][i][t]
                                             - self.shipment[f][w][i][t]
                                             + self.production[i][f][lt]
                                             for dc in self.dc_id
                                             for w in self.wh_id]) >= self.inv_level[i][f][t]

        for d in self.dc_id:
            for ind, t in enumerate(self.time_id):
                self.inv_model += lpSum([self.inv_level[i][d][t] + self.ss_df.loc[i, "central_wh"]
                                         for i in self.prod_id]) <= self.loc_data.loc[d, "Hold. Cap."]
                for i in self.prod_id:
                    prevt = self.time_id[max(ind - 1, 0)]
                    self.inv_model += lpSum([self.shipment[o][d][i][prevt] +
                                             self.inv_level[i][d][prevt] -
                                             self.data["Pallets"].get(
                                             (t, i, d), 0)
                                             - self.shipment[d][wh][i][t]
                                             for o in self.factory_id
                                             for wh in self.wh_id]) >= self.inv_level[i][d][t] + self.ss_df["central_wh"].get(i, 0)

        for d in self.wh_id:
            for ind, t in enumerate(self.time_id):
                self.inv_model += lpSum([self.inv_level[i][d][t] + self.ss_df.loc[i, "regional_wh"]
                                         for i in self.prod_id]) <= self.loc_data.loc[d, "Hold. Cap."]
                for i in self.prod_id:
                    prevt = self.time_id[max(ind - 1, 0)]
                    self.inv_model += lpSum([self.shipment[o][d][i][prevt] +
                                             self.shipment[f][d][i][prevt] +
                                             self.inv_level[i][d][prevt] -
                                             self.data["Pallets"].get(
                                             (t, i, d), 0)
                                             for f in self.factory_id
                                             for o in self.dc_id]) >= self.inv_level[i][d][t] + self.ss_df["regional_wh"].get(i, 0)

        for o in self.factory_id + self.dc_id:
            for d in self.dc_id + self.wh_id:
                for t in self.time_id:
                    self.inv_model += lpSum([
                                            self.shipment[o][d][i][t] *
                                            (1 / 33)
                                            for i in self.prod_id]) == self.FTL[o][d][t]
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

        ss_cw = self.service_level * (self.lt_df["lead_time"] * self.demand_stats["var"] +
                                      self.demand_stats["mean"]**2 * self.lt_df["var_lead_time"])**0.5
        ss_rw = self.service_level * \
            (self.lt_df["lead_time"] * self.demand_stats["var"])**0.5
        ss_dic = {"central_wh": ss_cw, "regional_wh": ss_rw}
        return pd.DataFrame(data=ss_dic, index=self.demand_stats.index)

    def build_model(self):
        ''' calls all the required function to initialize, solve and export the model
        '''

        self.define_paramaters(loc_data_path="./CSV input files/wh_data.csv",
                               ftl_matrix="./CSV input files/FTLmatrix.csv")
        self.define_indices(new_time_col="week")
        self.define_variables()
        self.define_objective()
        self.define_constraints()
        solver = CPLEX_PY()
        self.inv_model.solve(solver)
        # in case the model is not solvable display which contraint(s) are conflicting:
        solver.solverModel.parameters.conflict.display.set(2)
        self.inv_model.writeLP("test.LP")
        self.inv_model.writeMPS("test.mps")
        print("Objective value: ", value(self.inv_model.objective))
        print("Inventory level:--------------")
        self.export_vars_3d(time_ind=self.time_id,
                            wh_ind=self.loc_id,
                            prod_ind=self.prod_id,
                            variable=self.inv_level,
                            filename="Inventory Level")
        print("Shipments:------------------")
        self.export_vars_4d(time_ind=self.time_id,
                            origin_ind=self.loc_id,
                            destination_ind=self.loc_id,
                            prod_ind=self.prod_id,
                            variable=self.shipment,
                            filename="Shipment")
        # print("Full Truck Loads:-----------")
        # self.export_vars_3d(time_ind=self.time_id,
        #                     wh_ind=self.loc_id,
        #                     prod_ind=self.loc_id,
        #                     variable=self.FTL,
        #                     filename="FTL")
        print("Production:-----------------")
        self.export_vars_3d(time_ind=self.time_id,
                            wh_ind=self.factory_id,
                            prod_ind=self.prod_id,
                            variable=self.production,
                            filename="Production")
        # self.export_vars_3d(time_ind=self.time_id,
        #                     wh_ind=self.loc_id,
        #                     prod_ind=self.prod_id,
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
                        dic["origin"].append(o)
                        dic["destination"].append(d)
                        dic["product"].append(i)
                        dic["time"].append(t)
                        dic["value"].append(variable[o][d][i][t].varValue)
        self.var_df = pd.DataFrame.from_dict(dic)
        self.var_df = self.var_df.fillna(0)
        self.var_df = self.var_df[self.var_df["value"] > 0]
        self.var_df.to_csv(filename + ".csv")
        print(self.var_df)
        print("--- %s seconds ---" % (time.time() - start_time))
        print(f"{filename} exported")


# Actually creating an instance of the classes in which the model is defined
s = 1.95
time_dic = {}

start_time = time.time()
I = InventoryModel(data_path="./CSV input files/orders_abc_xyz.csv",
                   product_col="sh_ItemId",
                   time_col="sh_ShipmentDate",
                   loc_col="new_shOrigin",
                   qty_col="Pallets",
                   service_level=s)
I.build_model()
time_taken = time.time() - start_time
print("Total Time: ",time_taken)
time_dic[n] = time_taken



'''
TODO:
- automate time index
- automate removal in distance matrix
- improve export functions (make indices argmuent list like)
- 
'''

# see how to index lead time and holding costs
