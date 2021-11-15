import configparser
import os.path
import argparse
import pandas as pd
from Safety import SafetyStock
from collections import defaultdict
import ipdb


class Initialisation():
    """This task verifies file integrity and create necessary objects"""

    def __init__(self):
        # super(Initialisation, self).__init__()
        self.config_dict = self.read_config()
        self.biobj, self.tune = self.read_args()
        self.decimal = self.config_dict["filenames"]["decimal"]

    def read_config(self):
        '''
        extract the paramaters defined nto config file into a dictionary for easier access troughout code
        '''
        config = self.config_dialog()
        config_dict = {}
        for section in config.sections():
            config_dict[section] = dict(config.items(section))
        for k in config_dict:
            for key, value in config_dict[k].items():
                if value == "True":
                    config_dict[k][key] = True
                if value == "False":
                    config_dict[k][key] = False
        if self.verify_files(config_dict["filenames"]):
            return config_dict

    def config_dialog(self, overide=None):
        config = configparser.ConfigParser()
        while True:
            config_file = input("Please specifiy configuration file ")
            if os.path.exists(config_file):
                config.read(config_file)
                break
            else:
                print("The configuration file could not be found. Please try again. ")
                continue
        return config

    def create_files(self):
        self.writer = pd.ExcelWriter(
            self.config_dict["filenames"]["result_file"], engine="xlsxwriter")
        if not os.path.exists("CSV export files"):
            os.makedirs("CSV export files")
        if not os.path.exists("Saved Models"):
            os.makedirs("Saved Models")

    def arg_parse(self):
        '''
        defines argument of cmd flags
        '''
        parser = argparse.ArgumentParser(
            description="Launch a bi-objective optimisation")
        parser.add_argument("--biobj", type=bool)
        parser.add_argument("--tune", type=bool)
        return parser

    def read_args(self):
        # need test
        biobj = self.arg_parse().parse_args().biobj
        tune = self.arg_parse().parse_args().tune
        return biobj, tune

    def verify_files(self, files):
        for f in list(files.values())[:4]:
            if os.path.isfile(f) is False:
                print(f"Please check the file path for {f} could not be found")
            else:
                return True

    def create_writer(self):
        writer = pd.ExcelWriter(
            self.config_dict["filenames"]["result_file"], engine="xlsxwriter")
        return writer

    def read_batch_size(self):
        decimal = self.config_dict["filenames"]["decimal"]
        m = pd.read_csv(self.config_dict["filenames"]["batch_size"],
                        index_col=[0],
                        sep=None,
                        engine="python",
                        decimal=decimal)
        return m


class Subsetting():
    """docstring for Subsetting"""

    def __init__(self, initial, data):
        self.data = data
        self.initial = initial
        loc_data_path = self.initial.config_dict["filenames"]["location_data"]
        self.loc_data = pd.read_csv(loc_data_path,
                               index_col=[0],
                               sep=None,
                               engine="python",
                               decimal=self.initial.decimal)
        self.define_indices()

    def compute_ss(self):
        e = self.echelons()
        loc = self.loc_data
        s = SafetyStock(self.data, loc, e["cw"],  e["rw"], self.initial.config_dict)
        ss = s.ss_allocation()
        return ss

    def define_indices(self):
        loc_data_path = self.initial.config_dict["filenames"]["location_data"]
        loc_data = pd.read_csv(loc_data_path,
                               index_col=[0],
                               sep=None,
                               engine="python",
                               decimal=self.initial.decimal)
        self.sku_id = self.data["SKU_id"].unique()  # Set of all SKUs I
        self.time_id = self.data["period"].unique(
        ).tolist()  # Set of all time periods T
        self.time_id.insert(0, "0")  # create t=0 index
        self.loc_id = self.loc_data.index.tolist()  # Set of all network facilities W
        self.factory_id = (self.loc_data[self.loc_data["echelon"] == "Factory"]
                           .index.tolist())  # Set of factories F
        self.cw_id = (self.loc_data[self.loc_data["echelon"] == "Central"]
                      .index.tolist())  # Set of CW D
        self.rw_id = (self.loc_data[self.loc_data["echelon"] == "Regional"]
                      .index.tolist())  # set of RW J
        self.ext_factory_id = list(set(self.data["producedBy"].unique(
        ).tolist()) - set(self.factory_id))  # Set of ext suppliers O
        idx_dict = {"time_id": self.time_id,
                    "sku_id": self.sku_id,
                     "loc_id": self.loc_id,
                     "fact_id": self.factory_id,
                     "cw_id": self.cw_id,
                     "rw_id": self.rw_id,
                     "ext_fact_id": self.ext_factory_id}
        return idx_dict

    def read_loc_data(self):
        loc_data_path = self.initial.config_dict["filenames"]["location_data"]
        loc_data = pd.read_csv(loc_data_path,
                               index_col=[0],
                               sep=None,
                               engine="python",
                               decimal=self.initial.decimal)
        return loc_data

    def echelons(self):
        """
        Define and return a dict of echelon memberships
        """
        f = (self.loc_data[self.loc_data["echelon"] == "Factory"]
                           .index.tolist())  # Set of factories F
        c = (self.loc_data[self.loc_data["echelon"] == "Central"]
                      .index.tolist())  # Set of CW D
        r = (self.loc_data[self.loc_data["echelon"] == "Regional"]
                      .index.tolist())  # set of RW J
        ech = {"f": f,
               "cw": c,
               "rw": r}
        return ech

    def find_last_t(self):
            '''
        for each SKU find index of last time period where an order occurs
        returns a dictionary with last time period for each SKU
        {"SKU1": "t1", "SKU2": "t2"}
        '''
            last = self.data.sort_values(by=["period"]).drop_duplicates(subset=["SKU_id"], keep="last")
            last_t = dict(zip(last["SKU_id"], last["period"]))
            last_t = {k: self.time_id.index(v) + 1
                      for k, v in last_t.items()}
            return last_t

    def inventory_indices(self):
        last_t = self.find_last_t()
        skus = self.SKU_loc_assignment()
        return [(i, w, t) for w in self.loc_id for i in skus.get(w, []) for t in self.time_id[:last_t[i]]]

    def shipment_indices(self):
        dep = self.departure_allocation()
        origins = self.factory_id + self.cw_id + self.ext_factory_id
        skus = self.SKU_loc_assignment()
        inv_idx = self.inventory_indices()
        return [(o, d, i, t) for i, d, t in inv_idx for o in origins if i in skus.get(o, []) and d in dep.get(o, [])]

    def ftl_indices(self):
        ship_idx = self.shipment_indices()
        return [(o, d, t) for o,d,i,t in ship_idx]

    def production_indices(self):
        intsku_f, _ = self.production()
        last_t = self.find_last_t()
        return [(i, f, t) for f in self.factory_id for i in intsku_f.get(f, []) for t in self.time_id[:last_t[i] + 1]]

    def ls_indices(self):
        demand = self.data.set_index(["period",
                                      "SKU_id",
                                      "location"])["pallets"].to_dict()
        return demand.keys()

    def bin_indices(self):
        last_t = self.find_last_t()
        mbs = self.minimize_bin()
        big_M, big_M_alt = self.compute_M(self.data, mbs)
        return [(i, t) for i in mbs for t in self.time_id[:last_t[i]]]


    def sum_indices(self):
        ship_idx = self.shipment_indices()
        sum_idx = defaultdict(list)
        for a in ship_idx:  # isolate all SKUs that occur for each origin-destination-period triplet
            k = (a[0], a[1], a[3])
            sum_idx[k].append(a[2])
        return sum_idx 

    def minimize_bin(self):
        m = self.initial.read_batch_size()
        m = m[m["min_batch_active"] > 0 ] #  only keep min batch size for active skus
        m = m[m.index.isin(self.sku_id)]
        mbs = dict(zip(m.index, m["min. Batch Size (PAL)"]))
        temp = self.data.groupby(["SKU_id", "period"]).agg({"pallets": "sum"})
        temp = temp[temp["pallets"] != 0].groupby(["SKU_id"]).agg({"pallets": "min"})
        temp = dict(zip(temp.index, temp["pallets"]))
        l = []
        for i in list(mbs):
            if temp[i] > mbs[i]:
                l.append(i)
                del mbs[i]
        r = str(len(l) * len(self.time_id))
        if r != "0":
            print(r + " Binaries variables have been removed")
        return mbs

    def compute_M(self, data, mbs):
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
        temp = self.data.groupby("SKU_id").agg({"pallets": "sum"})
        temp = dict(zip(temp.index, temp["pallets"]))
        big_M = {}
        big_M_alt = {}
        for i in mbs.keys():
            # big_M[i] = max((temp[i] + self.cw_ss.get(i, 0) + self.rw_ss.get(i, 0)), mbs[i])
            if temp[i] > mbs[i]:
                big_M[i] = temp[i]
            else:
                big_M_alt[i] = mbs[i] 
        return big_M, big_M_alt

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
            dep[e] = self.data["location"][self.data["producedBy"] == e].unique().tolist() + self.cw_id
        return dep

    # def function(self):
    #         self.int_skus, self.ext_skus = self.int_ext_skus()
    #         self.intsku_fact, self.sku_plan = self.production()
    #         self.intsku_CW, self.extsku_CW = self.cw_int_ext_skus()

    def int_ext_skus(self):
        int_skus = (self.data[self.data["producedBy"]
                              .isin(self.factory_id)]["SKU_id"]
                    .to_list())  # [intsku1, intsku2]
        ext_skus = (self.data[self.data["producedBy"]
                              .isin(self.ext_factory_id)]["SKU_id"]
                    .to_list())  # [extsku1, extsku2]
        return int_skus, ext_skus

    def production(self):
        # assign which factory produces which sku
        k = (self.data[self.data["producedBy"]
             .isin(self.factory_id)]
             .groupby("producedBy")["SKU_id"]
             .apply(set).apply(list))
        intsku_fact = dict(zip(k.index, k))  # {"factory1": ["intSKU1", "intSKU2"]}
        # asssign which sku can be supplied by which factory inverse of the previous dic
        k = (self.data[self.data["producedBy"]
             .isin(self.factory_id)]
             .groupby("SKU_id")["producedBy"]
             .apply(set).apply(list))
        sku_plan = dict(zip(k.index, k))  # {"sku1": "[factory1]"}
        return intsku_fact, sku_plan

    def fact_ext_skus(self):
        _, ext = self.int_ext_skus()
        k = (self.data[(self.data["location"].isin(self.factory_id)) 
                       & (self.data["SKU_id"].isin(ext))]
                       .groupby("location")["SKU_id"]
                       .apply(set).apply(list))
        extsku_f = dict(zip(k.index, k))
        return extsku_f

    def f2f_skus(self):
        f2f_fact_id = list(self.loc_data.loc[~self.loc_data["factory_to_factory"].isna()].index)  # extract factories in need of special constraints
        f2f_sku = {}
        for f in f2f_fact_id:
            temp = self.loc_data.loc[f, "factory_to_factory"].split(",")  # extract factories that can supply ["fact1", "fact2"]
            skus = (self.data[(self.data["location"] == f) & (self.data["producedBy"].isin(temp))]
                    .groupby("location")["SKU_id"]
                    .apply(set).tolist())  # extract SKUs that 1. factory f 2. is produced by a factory in "factory_to_factory" column, {f: ["sku1", "sku2"]}

            if len(skus) > 0:
                skus = [s for s in skus[0]]
                f2f_sku[f] = skus  # {fact1: ["sku1", "sku2"]}
        return f2f_sku

    def cw_int_ext_skus(self):
        k = self.compute_ss()[0]
        k = k.reset_index() # retrieve ss for cws
        int_s, ext_s = self.int_ext_skus()
        k = (k[k["SKU_id"].isin(int_s)]
                              .groupby("location")["SKU_id"]
                              .apply(set).apply(list))
        intsku_CW = dict(zip(k.index, k)) # subset of int SKUs held at central warehouses {"CW1": ["intSKU1", "intSKU2"]}
        
        k = self.compute_ss()[0]
        k = k.reset_index()
        k = (k[k["SKU_id"].isin(ext_s)]
                               .groupby("location")["SKU_id"]
                               .apply(set).apply(list))
        extsku_CW = dict(zip(k.index, k))  # subset of ext SKUs held at central warehouses {"CW1": ["extSKU1", "extSKU2"]}
        return intsku_CW, extsku_CW

    def rw_int_ext_skus(self):
        int_s, ext_s = self.int_ext_skus()
        k = (self.data[(self.data["location"]
             .isin(self.rw_id)) & (self.data["SKU_id"]
             .isin(int_s))]
             .groupby("location")["SKU_id"]
             .apply(set).apply(list))
        intsku_RW = dict(zip(k.index, k)) # {"RW1": ["intsku1", "intsku2"]}

        k = (self.data[(self.data["location"]
             .isin(self.rw_id)) 
             & (self.data["SKU_id"].isin(ext_s))]
             .groupby("location")["SKU_id"]
             .apply(set).apply(list))
        extsku_RW = dict(zip(k.index, k)) # {"RW1": ["extsku1", "extsku2"]}
        return intsku_RW, extsku_RW
        #  create this subet for skuloc then finish invenotry indices

    def SKU_loc_assignment(self):
        '''
        Store all SKUs that could potentially be held at each location
        '''
        sku_LOC = {}
        int_f, _ = self.production()  # skus produced by each factory
        ext_f, f2f = self.fact_ext_skus(), self.f2f_skus() # external skus at factories, skus produced by other factories
        int_c, ext_c = self.cw_int_ext_skus() # internal + external skus at central w
        int_r, ext_r = self.rw_int_ext_skus() # internal + external skus at regional w
        for l in self.loc_id:
            if l in self.factory_id:
                sku_LOC[l] = int_f.get(l, []) + ext_f.get(l, []) + f2f.get(l, [])
            if l in self.cw_id:
                sku_LOC[l] = int_c.get(l, []) + ext_c.get(l, [])
            if l in self.rw_id:
                sku_LOC[l] = int_r.get(l, []) + ext_r.get(l, [])
        # adding ext suppliers skus
        k = (self.data[self.data["producedBy"]
             .isin(self.ext_factory_id)]
             .groupby("producedBy")["SKU_id"]
             .apply(set).apply(list))
        ext = dict(zip(k.index, k))
        sku_LOC.update(ext)
        return sku_LOC

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
            dep[e] = self.data["location"][self.data["producedBy"] == e].unique().tolist() + self.cw_id
        return dep

    def supplier(self):
        # assign which supplier supplies which sku
        k = (self.data[self.data["producedBy"]
             .isin(self.ext_factory_id)]
             .groupby("SKU_id")["producedBy"]
             .apply(set).apply(list))
        supplier = dict(zip(k.index, k)) # {"SKU1": "ExtFact1"}
        return supplier

    def direct_sh_cons(self):
        direct_sh_extsku = dict(zip(self.loc_data.index, self.loc_data["direct shipment ext. SKU"]))
        return direct_sh_extsku

    def minimize_constraint(self):
        '''
        return the index last time period where a demand occur for each sku at 3rd echelon
        in order to reduce the number of constraint being generated 
        => no need to define a constraint after the last demand has occured for that sku
        returns: a dictionary -> {(Location1, SKU1): index(period1)}
        '''
        last = self.data.sort_values(by=["period"]).drop_duplicates(subset=["SKU_id", "location"], keep="last")
        last = last[last["location"].isin(self.rw_id)]
        last_t = dict(zip(zip(last["location"], last["SKU_id"]), last["period"]))
        last_t = {k: self.time_id.index(v)
                  for k, v in last_t.items()}
        return last_t




class Preprocess():
    """This class is tasked with preparing the data before """

    def __init__(self, n_rows=None):
        self.initial = Initialisation()
        

        self.n_rows = n_rows

        self.data = self.clean_data()

        self.sub = Subsetting(self.initial, self.data)

    def read_raw_data(self):
        decimal = self.initial.config_dict["filenames"]["decimal"]
        d = {"SKU_id": "object"}
        path = self.initial.config_dict["filenames"]["orders"]
        raw_data = pd.read_csv(path,
                               index_col=[0],
                               nrows=self.n_rows,
                               dtype=d,
                               sep=None,
                               engine="python",
                               decimal=decimal)
        return raw_data

    def read_clusters(self, data):
        try:
            data["XYZ_cluster"] = data["XYZ_cluster"].fillna("Z")
            data = data.loc[~((data["ABC_cluster"] == "C") &
                              (data["XYZ_cluster"] == "Z"))]
            return data
        except KeyError:
            print("No clusters found")


    def save_data(self, data):
        writer = self.initial.create_writer()
        data.to_excel(writer, sheet_name="Demand")

    def clean_data(self):
        raw = self.read_raw_data()
        data = raw[raw["location"].notna()]
        self.data = self.read_clusters(data)
        self.save_data(data)
        return self.data   

    def compute_ss(self):
        e = self.sub.echelons()
        loc = self.sub.read_loc_data()
        s = SafetyStock(self.data, loc, e["cw"],  e["rw"], self.initial.config_dict)
        ss = s.ss_allocation()
        return ss

    def holding_costs(self):
        hc = self.data.groupby(["SKU_id", "location"])["hold_cost_pallet"].first()
        default_hc = hc.groupby("SKU_id").mean().to_dict()
        return hc.to_dict(), default_hc

    def prod_costs(self):
        p_c = dict(zip(zip(self.data["SKU_id"], self.data["producedBy"]), self.data["prod_costs"]))
        return p_c

    def ftl_matrix(self):
        ftl_matrix_path = self.initial.config_dict["filenames"]["ftl_matrix"]
        decimal = self.initial.config_dict["filenames"]["decimal"]
        ftl_m = pd.read_csv(ftl_matrix_path,
                            index_col=[0],
                            sep=None,
                            engine="python",
                            decimal=decimal)
        return ftl_m.to_dict()

    def holding_capacity(self):
        loc = self.sub.read_loc_data().to_dict()
        return loc["hold. cap."]

    def prod_capacity(self):
        loc = self.sub.read_loc_data().to_dict()
        return loc["prod. cap."]

    def lead_time(self):
        sku_id = self.sub.define_indices()["sku_id"]
        t_id = self.sub.define_indices()["time_id"]
        lt_dic = self.data["lead_time"].to_dict()
        lt = {(i,t): t_id[max(int(ind - lt_dic.get(i,1)), ind-1)]
                                    for i in sku_id
                                    for ind, t in enumerate(t_id)}
        return lt

    def sum_ss(self):
        ss_sum = {}
        time_skus_dic = self.sub.find_last_t()
        cw_ss, rw_ss = self.compute_ss()
        # find out for each time period which skus has last demand in time for that time period
        d = {n: [k for k in time_skus_dic.keys() if time_skus_dic[k] == n] for n in set(time_skus_dic.values())}

        for w in self.sub.loc_id:
            for k, v in d.items():
                t = self.sub.time_id[k-1]
                ss_sum[(w, t)] = sum(cw_ss.get((w, sku), rw_ss.get((w, sku), 0)) for sku in v)
        return ss_sum

# TODO
# break-up define_indices
    


        


