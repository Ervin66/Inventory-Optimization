import pandas as pd
import numpy as np
import time
import ipdb


class ExportVariables():
    """docstring for ExportVariables"""

    def __init__(self, model, id_dict, initial):
        super(ExportVariables, self).__init__()
        # self.indices = indices
        # self.variable = variable
        # self.filename = filename
        self.model = model
        self.i_d = id_dict
        self.writer = initial.create_writer()

    def export_solutions(self):
        self.export_ss()
        # export inv
        model = self.model
        self.indices, self.variable, self.filename = model.inv_idx, model.inv_level, "Inventory Level"
        self.inv_results = self.export_inv()

        self.indices, self.variable, self.filename = model.ship_idx, model.shipment, "Shipments"
        self.export_vars_4d()

        self.indices, self.variable, self.filename = model.prod_idx, model.prod, "Production"
        self.export_vars_3d()

        self.indices, self.variable, self.filename = model.ls_idx, model.lost_sales, "Lost Sales"
        self.export_vars_3d()
        self.writer.save()

    def export_rec_solutions(self, results):
        self.export_ss()
        # export inv
        model = self.model
        self.indices, self.variable, self.filename = results["inventory"][0], results["inventory"][1], "Inventory Level"
        self.inv_results = self.export_inv()

        self.indices, self.variable, self.filename = results["shipment"][0], results["shipment"][1], "Shipments"
        self.export_vars_4d()

        self.indices, self.variable, self.filename = results["production"][0], results["production"][1], "Production"
        self.export_vars_3d()

        self.indices, self.variable, self.filename = results["lost_sales"][0], results["lost_sales"][1], "Lost Sales"
        self.export_vars_3d()
        self.writer.save()

    def detect_var_dimensions(self):
        for i in self.indices:
            return len(i)

    def export_ss(self):
        self.model.cw_ss.to_excel(self.writer, sheet_name="Central Warehouse SS")
        self.model.rw_ss.to_excel(self.writer, sheet_name="Regional Warehouse SS")

    def export_vars(self):
        dim = self.detect_var_dimensions()
        if dim == 2:
            res = self.export_vars_2d()
            return res
        if dim == 3:
            if self.filename == "Inventory Level":
                res = self.export_inv()
            else:
                res = self.export_vars_3d()
            return res
        if dim == 4:
            ship, ftl = self.export_vars_4d()
            return ship, ftl

    def export_vars_2d(self):
        ''' Formats the solution and returns it embedded in a pandas dataframe
        self.indices: list of self.indices over which the variable is defined
        variable: variable name to be exported
        filename: under which name to save the varaible to be exported to csv
        '''
        start_time = time.time()
        dic = {"time": [], "warehouse": [], "value": []}
        for w, t in self.indices:
            dic["time"].append(t)
            dic["warehouse"].append(w)
            dic["value"].append(self.variable[(w, t)].varValue)

        self.var_df = pd.DataFrame.from_dict(dic)
        self.var_df = self.var_df.fillna(0)

        self.var_df.to_excel(self.writer, sheet_name=self.filename)
        self.var_df.to_csv(f"./CSV export files/{self.filename}.csv")
        self.var_df.loc[self.var_df["value"] < 0, "value"] = 0
        print(self.var_df)
        print("--- %s seconds ---" % (time.time() - start_time))
        print(f"{self.filename} exported")
        return self.var_df

    def export_vars_3d(self):
        ''' Formats the solution and returns it embedded in a pandas dataframe
        self.indices: list of self.indices over which the variable is defined
        variable: variable name to be exported
        filename: under which name to save the varaible to be exported to csv
        '''
        start_time = time.time()
        dic = {"time": [], "warehouse": [], "product": [], "value": []}
        for i, w, t in self.indices:
            dic["time"].append(t)
            dic["warehouse"].append(w)
            dic["product"].append(i)
            dic["value"].append(self.variable[(i, w, t)].varValue)

        self.var_df = pd.DataFrame.from_dict(dic)
        self.var_df = self.var_df.fillna(0)
        try:
            if self.filename == "Lost Sales":
                self.var_df.columns = ["warehouse", "SKU_id", "time", "value"]
        except AttributeError:
            pass
        except NameError:
            pass
        self.var_df.to_excel(
            self.writer, sheet_name=self.filename, float_format="%.2f")
        self.var_df.to_csv(f"./CSV export files/{self.filename}.csv")
        self.var_df.loc[self.var_df["value"] < 0, "value"] = 0
        print(self.var_df)
        print("--- %s seconds ---" % (time.time() - start_time))
        print(f"{self.filename} exported")
        return self.var_df

    def export_inv(self):
        ''' Formats the solution and returns it embedded in a pandas dataframe
        indices: list of indices over which the variable is defined
        variable: variable name to be exported
        filename: under which name to save the varaible to be exported to csv
        '''
        start_time = time.time()
        dic = {"time": [], "warehouse": [], "product": [], "value": []}
        for i, w, t in self.indices:
            dic["time"].append(t)
            dic["warehouse"].append(w)
            dic["product"].append(i)
            dic["value"].append(self.variable[(i, w, t)].varValue)

        self.var_df = pd.DataFrame.from_dict(dic)
        self.var_df = self.var_df.set_index(["warehouse", "product", "time"])
        ind = [(w, i, t) for w in self.i_d["loc_id"] for i in self.model.sku_LOC.get(w, [])
               for t in self.i_d["time_id"]]  # create complete self.indices (including all time periods)
        # extract missing self.indices
        ind = pd.Index(ind).difference(self.var_df.index)
        add_df = pd.DataFrame(index=ind, columns=self.var_df.columns)
        # merge results with missing self.indices
        self.var_df = pd.concat([self.var_df, add_df])
        self.var_df.sort_index(inplace=True)
        # fill the values with the last available
        self.var_df.fillna(method="ffill", inplace=True)
        self.var_df.reset_index(inplace=True)
        self.var_df.to_excel(self.writer, sheet_name=self.filename)
        self.var_df.to_csv(f"./CSV export files/{self.filename}.csv")
        print(self.var_df)
        print("--- %s seconds ---" % (time.time() - start_time))
        print(f"{self.filename} exported")
        return self.var_df

    def export_vars_4d(self):
        start_time = time.time()
        dic = {"time": [], "origin": [],
               "destination": [], "product": [], "value": []}
        for o, d, i, t in self.indices:
            if self.variable[(o, d, i, t)].varValue is not None:
                if self.variable[(o, d, i, t)].varValue > 0:
                    dic["origin"].append(o)
                    dic["destination"].append(d)
                    dic["product"].append(i)
                    dic["time"].append(t)
                    dic["value"].append(self.variable[(o, d, i, t)].varValue)

        self.var_df = pd.DataFrame.from_dict(dic)
        self.var_df = self.var_df.fillna(0)
        self.var_df.to_excel(self.writer, sheet_name=self.filename)
        self.var_df.to_csv(f"./CSV export files/{self.filename}.csv")
        self.FTL_df = self.var_df.groupby(
            ["origin", "destination", "time"], as_index=True).sum()
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
        print(f"{self.filename} exported")
        return self.var_df, self.FTL_df

    def backfill(self, df):
        ind = pd.MultiIndex.from_product(
            [self.time_id, df["warehouse"], df["product"]])
        df = df.reindex(ind, method="bfill")
        print(df)
