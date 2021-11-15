import ujson
from Preprocess import Preprocess
import re
from collections import OrderedDict
import os
from pulp import *
import ipdb


class Recycle():
    """docstring for Recycle"""

    def __init__(self, n):
        self.initial = Preprocess(n)
        self.i_d = self.initial.sub.define_indices()
        self.loc_da = self.initial.sub.read_loc_data()
        self.time_id = self.i_d["time_id"]
        self.cw_ss, self.rw_ss = self.initial.compute_ss()
        self.cw_ss, self.rw_ss = self.cw_ss["Safety_Stock"].fillna(0), self.rw_ss["Safety_Stock"].fillna(0)

    def parse_variables(self):
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

    def parse_var_keys(self, var_key):
        a = tuple(re.findall(r"'(\w+)'", var_key))
        return a

    def update_constraint_rhs(self, constraints):
        '''
        this method allows to modify the RHS (for some) coefficient upon loading an existing model
        '''
        cons_modified = {"Production Cap.": 0,
                         "Holding Cap.": 0,
                         "Initial Inv.": 0,
                         "Min. Inv.": 0} 
        for fact in self.i_d["fact_id"]:
            if constraints.get(f"('{fact}', '0')ProdCap", False) is not False:
                if constraints[f"('{fact}', '0')ProdCap"].constant !=  - self.loc_da["prod. cap."][fact]: #  check if prod. cap. has changed
                    new_cap = - self.loc_da["prod. cap."][fact]
                    for t in self.time_id:
                        constraints[f"('{fact}', '{t}')ProdCap"].constant = new_cap
                        cons_modified["Production Cap."] += 1
        for wh in self.i_d["loc_id"]:
            if constraints.get(f"('{wh}', '0')HoldCap", False) is not False:
                if constraints[f"('{wh}', '0')HoldCap"].constant !=  - self.loc_da["hold. cap."][wh]:
                    new_cap = - self.loc_da["hold. cap."][wh]
                    for t in self.time_id:
                        constraints[f"('{wh}', '{t}')HoldCap"].constant = new_cap
                        cons_modified["Holding Cap."] += 1

        last_t = self.initial.sub.find_last_t()
        c_ss = self.cw_ss.to_dict()

        p = [k for k, v in c_ss.items() if v > 0]
        t_ = self.time_id[1]
        for d, i in p:
            if constraints[f"('{d}', '{t_}', '{i}')2ndech_ssreq"].constant != - c_ss[(d, i)]:
                new_ss = - self.cw_ss[(d, i)]
                for t in self.time_id[:last_t[i]]:
                    constraints[f"('{d}', '{t}', '{i}')2ndech_ssreq"].constant = new_ss
                    cons_modified["Min. Inv."] += 1

                old_initial = constraints[f"('{d}', '0', '{i}')initial"].constant
                constraints[f"('{d}', '0', '{i}')initial"].constant -= (old_initial - new_ss)
                cons_modified["Initial Inv."] += 1

        r_ss = self.rw_ss.to_dict()
        p = [k for k, v in r_ss.items() if v > 0]
        for w, i in p:
            if constraints[f"('{w}', '{t_}', '{i}')3rdech_ssreq"].constant != - r_ss[(w, i)]:
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

    def recycle_model(self):
        if any(os.scandir("./Saved Models")):
            while True:
                old_mod = input("Use existing model? (y/n) ")
                if old_mod == "y":
                    variables, model, model.constraints = self.read_model()
                    # maybe need model.constraints=
                    self.update_constraint_rhs(model.constraints)
                    return model.constraints, variables, model
                if old_mod == "n":
                    return True
                    break

    def read_model(self):
        while True:
            model_file = input("Please indicate model's name ")
            try:
                variables, model = LpProblem.from_json("./Saved Models/" + model_file + ".json")
                with open("./Saved Models/" + model_file + "constraints.json") as f:
                    cons = ujson.load(f)

                    return variables, model, OrderedDict(zip(cons.keys(), model.constraints.values()))
            except FileNotFoundError:
                print("The model name cannot be found. Please try again.")
                continue

    def save_model_json(self, model, constraints, ask=False):
        if ask is False:
            model.toJson("./Saved Models/biobj.json")  # saving the model to json
            with open("./Saved Models/biobjconstraints.json", "w") as cons_out:
                ujson.dump(constraints, cons_out, sort_keys=False)   # saving the constraints separetely in order to regain the constraints naming convention
        else:
            if bool(self.initial.initial.config_dict["model"]["save_model"]):
                model_name = self.initial.initial.config_dict["model"]["model_name"]                 
                model.toJson(f"./Saved Models/{model_name}.json")  # saving the model to json
                with open(f"./Saved Models/{model_name}constraints.json", "w") as cons_out:
                    ujson.dump(constraints, cons_out, sort_keys=False) # saving the constraints separetely in order to regain the constraints naming convention

    def load_biobj(self):
        variables, model = LpProblem.from_json("./Saved Models/biobj.json")
        with open("./Saved Models/biobjconstraints.json") as f:
            cons = ujson.load(f)
        return model