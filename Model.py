import pandas as pd
from pulp import *
import ipdb


class InventoryModel():
    """class used to define and construct the inventory linear program
       data_path: the file path of the data containing the orders
       prodcut_col: string of the column's header containing the product (group) id
       time_col: string of the column's header containing the date of the order
       loc_col: string of the column's header containing the location of the order
       qty_col: string of the column's header containing the quantity of the order
    """

    def __init__(self, data_path, product_col, time_col, loc_col, qty_col):
        self.product_col = product_col
        self.time_col = time_col
        self.loc_col = loc_col
        self.raw_data = pd.read_csv(data_path,
                                    usecols=[product_col,
                                             time_col,
                                             loc_col,
                                             qty_col])

        # column's values are converted to datetime_col objects to facilitate weekly aggregation
        self.raw_data[time_col] = pd.to_datetime(self.raw_data[time_col])

        # remove orders that do not have a location specified
        self.raw_data = self.raw_data[self.raw_data[loc_col].notna()]
        self.data = self.raw_data.groupby([self.raw_data[time_col].dt.isocalendar().week,
                                           product_col,
                                           loc_col], dropna=False).sum()  # aggregate the order per week, product group and location
        print(self.data)
        print(self.data.groupby(product_col).agg(["size", "mean", "std"]))

        self.inv_model = pulp.LpProblem("Inventory_Optimization",
                                        LpMinimize)  # creates an object of an LpProblem (PuLP) to which we will assign variables, constraints and objective function

    def define_indices(self, new_time_col):
        ''' Returns the unique values for indices from the input data
        Required keywords argument:
        new_time_col: string of the aggregation basis for time (e.g. "week", "month",...)
        '''
        self.prod_id = self.data.index.get_level_values(
            self.product_col).unique().tolist()
        self.time_id = self.data.index.get_level_values(
            new_time_col).unique().tolist()
        self.loc_id = self.data.index.get_level_values(
            self.loc_col).unique().tolist()

    def define_paramaters(self, loc_data_path, demand_data_path):
        ''' Returns a dataframe of the paramaters required for the model
        loc_data_path: string of the path to the location data
        demand_data_path: string of the path of pre-computed statistics of the demand
        '''
        self.loc_data = pd.read_csv(loc_data_path,
                                    index_col=[0])

        self.demand_stats = self.data.groupby(
            self.product_col, as_index=False).agg(["size", "mean", "std"])
        self.demand_stats.to_csv("Demand Stats.csv")
        self.order_cost = 33
        print(self.demand_stats)
        print(self.demand_stats.loc[0, ("Pallets", "std")])

    def define_variables(self):
        '''Defines the variable and their nature whcih are then added to the model
        '''
        self.inv_level = pulp.LpVariable.dicts("inventory level",
                                               (self.prod_id,
                                                self.loc_id,
                                                self.time_id),
                                               lowBound=0)
        self.safety_stock = pulp.LpVariable.dicts("safety stock",
                                                  (self.prod_id,
                                                   self.loc_id,
                                                   self.time_id),
                                                  lowBound=0)
        self.max_stock = pulp.LpVariable.dicts("maxium stock level",
                                               (self.prod_id,
                                                self.loc_id,
                                                self.time_id),
                                               lowBound=0)
        self.shipment = pulp.LpVariable.dicts("maximum stock level",
                                              (self.prod_id,
                                               self.loc_id,
                                               self.time_id),
                                              lowBound=0)
        self.order_dec = pulp.LpVariable.dicts("Order decision",
                                               (self.prod_id,
                                                self.loc_id,
                                                self.time_id),
                                               cat="Integer",
                                               lowBound=0)

    def define_objective(self):
        ''' Defines the objective funciton
        '''
        holding_costs = pulp.lpSum((self.inv_level[i][w][t] * self.loc_data.loc[w, "holding costs"]
                                    for i in self.prod_id
                                    for w in self.loc_id
                                    for t in self.time_id))
        ordering_costs = pulp.lpSum([self.order_dec[i][w][t] * self.order_cost
                                     for i in self.prod_id
                                     for w in self.loc_id
                                     for t in self.time_id])
        self.inv_model += holding_costs + ordering_costs

    def define_constraints(self):
        '''Defines the constraints to be added to the model
        '''
        for w in self.loc_id:
            for ind, t in enumerate(self.time_id):
                for i in self.prod_id:
                    try:
                        if ind != 0:
                            prevt = self.time_id[ind - 1]
                            self.inv_model += lpSum(self.shipment[i][w][prevt] +
                                                    self.inv_level[i][w][prevt] -
                                                    self.data.loc[(t, i, w)]) == self.inv_level[i][w][t]

                            self.inv_model += lpSum(self.safety_stock[i][w][t]) == self.demand_stats.loc[i, (
                                "Pallets", "std")] * 2.33 * self.weighted_avg(prevt, w, i)
                        else:
                            self.inv_model += lpSum(
                                self.data.loc[(t, i, w)] - self.inv_level[i][w][t]) == 0

                            self.inv_model += lpSum(self.safety_stock[i][w][t]) == self.demand_stats.loc[i, (
                                "Pallets", "std")] * 2.33 * (1 / len(self.loc_id))

                            self.inv_model += lpSum(
                                self.order_dec[i][w][t]) == self.shipment[i][w][t] * (1 / 33)

                    except KeyError:

                        continue
                    self.inv_model += lpSum((self.inv_level[i][w][t] + self.safety_stock[i][w][t]
                                             for i in self.prod_id)) <= self.loc_data.loc[w, "Capacity"]
                # self.shipment += lpSum((self.data.loc[(t, i, w)] + self.safety_stock[i][w][t]))
        for t in self.time_id:
            for i in self.prod_id:
                self.inv_model += lpSum((self.safety_stock[i][w][t]
                                         for w in self.loc_id)) == self.demand_stats.loc[i, ("Pallets", "std")] * 2.33

    def weighted_avg(self, time_ind, wh_ind, prod_ind):
        '''Computes  the rolling distribution of the demand for each location in order to allocation the safety stock and returns the weight
        time_ind: time index for which the computation is being performed
        wh_ind: location index for which the computation is being performed
        prod_ind: product (group) index for which the computation is being performed
        '''
        weight = self.data.loc[:time_ind, prod_ind, wh_ind].sum(
        ) / self.data.loc[:time_ind, prod_ind, :].sum()
        return weight.values


    def build_model(self):
        ''' calls all the required function to initialize, solve and export the model
        '''
        self.define_indices(new_time_col="week")
        self.define_variables()
        self.define_paramaters(loc_data_path="./CSV input files/wh_data.csv",
                               demand_data_path="./CSV input files/varvol_cluster.csv")
        self.define_objective()
        self.define_constraints()
        solver = CPLEX_PY()
        self.inv_model.solve(solver)
        solver.solverModel.parameters.conflict.display.set(2)
        self.inv_model.writeLP("test.LP")
        self.inv_model.writeMPS("test.mps")
        print(value(self.inv_model.objective))
        self.export_vars(time_ind=self.time_id,
                         wh_ind=self.loc_id,
                         prod_ind=self.prod_id,
                         variable=self.inv_level,
                         name="Inventory Level")
        self.export_vars(time_ind=self.time_id,
                         wh_ind=self.loc_id,
                         prod_ind=self.prod_id,
                         variable=self.inv_level,
                         name="Safety Stock")
        self.export_vars(time_ind=self.time_id,
                         wh_ind=self.loc_id,
                         prod_ind=self.prod_id,
                         variable=self.shipment,
                         name="Shipment")
        self.export_vars(time_ind=self.time_id,
                         wh_ind=self.loc_id,
                         prod_ind=self.prod_id,
                         variable=self.order_dec,
                         name="Full Truck Loads")

    def export_vars(self, time_ind, wh_ind, prod_ind, variable, filename):
        ''' Formats the solution and returns it embedded in a pandas dataframe
        time_ind: time index used by the variable
        wh_ind: location index used by the varaible
        prod_ind: product (group) index used by the variable
        variable: variable name to be exported
        filename: under which name to save the varaible to be exported to csv
        '''
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


# Actually creating an instance of the classes in which the model is defined
I = InventoryModel(data_path="./CSV input files/clustered_orders.csv",
                   product_col="varvol_cluster",
                   time_col="sh_ShipmentDate",
                   loc_col="sh_OriginLocationMasterLocation",
                   qty_col="Pallets")
# calling the function that will assemble the model together
I.build_model()


'''
TODO:
- add ordering costs
- comment code
- Back-up
- automate time index
'''
