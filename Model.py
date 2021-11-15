import pandas as pd
from pulp import *
from Preprocess import Preprocess
from Preprocess import Subsetting
import ipdb

class InventoryModel():
    """docstring for InventoryModel"""

    def __init__(self, pre):
        self.pre = pre
        self.data = self.pre.clean_data()
        self.inv_model = pulp.LpProblem("Inventory_Optimization",
                                        LpMinimize)

    def build_model(self):
        self.define_variables()
        self.define_objective()
        cons = self.define_constraints()
        self.inv_model.extend(cons)
        return self.inv_model

    def define_variables(self):
        self.inv_idx = self.pre.sub.inventory_indices()
        self.inv_level = pulp.LpVariable.dicts("inventory",
                                               self.inv_idx,
                                               lowBound=0)
        self.ship_idx = self.pre.sub.shipment_indices()
        self.shipment = pulp.LpVariable.dicts("shipments",
                                              self.ship_idx,
                                              lowBound=0)
        self.ftl_idx = self.pre.sub.ftl_indices()
        self.FTL = pulp.LpVariable.dicts("FTL",
                                         self.ftl_idx,
                                         lowBound=0,
                                         cat="Integer")
        self.prod_idx = self.pre.sub.production_indices()
        self.prod = pulp.LpVariable.dicts("production",
                                          self.prod_idx,
                                          lowBound=0)
        self.ls_idx = self.pre.sub.ls_indices()
        self.lost_sales = pulp.LpVariable.dicts("lost sales",
                                                self.ls_idx,
                                                lowBound=0)
        self.bin_idx = self.pre.sub.bin_indices()
        self.bin_prod = pulp.LpVariable.dicts("ProdSetup",
                                             self.bin_idx,
                                             cat="Binary")

    def define_objective(self):
        costs = self.pre.initial.config_dict["model"]

        hc, default_hc = self.pre.holding_costs()
        holding_costs = LpAffineExpression(((self.inv_level[i], hc.get((i[0], i[1]), default_hc[i[0]]))
                                            for i in self.inv_idx))
        ftl = self.pre.ftl_matrix()
        trans_costs = LpAffineExpression(((self.FTL[(o, d, t)], ftl[d].get(o, int(costs["ftl_def"])))
                                          for o, d, t in self.ftl_idx))
        p_c = self.pre.prod_costs()
        prod_costs = LpAffineExpression(((self.prod[(i, f, t)], p_c[i, f])
                                         for i, f, t in self.prod_idx))
        shortage_costs = LpAffineExpression((self.lost_sales[i], int(costs["short_costs"]))
                                            for i in self.ls_idx)
        self.inv_model += holding_costs + trans_costs + prod_costs + shortage_costs

    def define_constraints(self):
        self.i_d = self.pre.sub.define_indices()
        self.sku_LOC = self.pre.sub.SKU_loc_assignment()
        self.last_t, self.time_id = self.pre.sub.find_last_t(), self.i_d["time_id"]
        self.prevt = {t :self.time_id[ind-1]  for ind, t in enumerate(self.time_id)} 
        self.demand = self.pre.data.to_dict()
        self.lt = self.pre.lead_time()

        self.cons_dict = {}
        self.cons_dict.update(self.holding_capacity_cons())
        self.cons_dict.update(self.prod_capacity_cons())
        self.cons_dict.update(self.ftl_cons())
        self.cons_dict.update(self.initial_shipment())
        self.cons_dict.update(self.intial_inv())
        self.cons_dict.update(self.st_echelon_int_inv())
        self.cons_dict.update(self.f2f_inv())
        self.cons_dict.update(self.nd_echelon_int_inv())
        self.cons_dict.update(self.rd_echelon_int_inv())
        self.cons_dict.update(self.st_echelon_ext_inv())
        self.cons_dict.update(self.nd_echelon_ext_inv())
        self.cons_dict.update(self.rd_echelon_ext_inv())
        self.cons_dict.update(self.min_batch_size())
        self.cons_dict.update(self.ss_req())
        return self.cons_dict

    def holding_capacity_cons(self):
        cap = self.pre.holding_capacity()
        last_ss = self.pre.sum_ss()
        cons = {f"{w,t}HoldCap": LpAffineExpression(((self.inv_level[(i, w, t)], 1)
                                                     for i in self.sku_LOC.get(w, []) if self.last_t[i] > self.time_id.index(t))) <= cap[w] - last_ss.get((w, t), 0)
                for x, w, t in self.inv_idx}
        return cons

    def prod_capacity_cons(self):
        isf, _ = self.pre.sub.production()
        cap = self.pre.prod_capacity()
        cons = {f"{f,t}ProdCap": LpAffineExpression(((self.prod[(i, f, t)], 1)
                                                     for i in isf.get(f, []) if self.last_t[i] > self.time_id.index(t))) <= cap[f]
                for x, f, t in self.prod_idx}
        return cons

    def ftl_cons(self):
        sum_idx = self.pre.sub.sum_indices()
        cons = {f"{o,d,t}FTL":LpAffineExpression(((self.shipment[(o,d,i,t)] ,1)
                                                                            for i in sum_idx.get((o,d,t), []))) == 33 * self.FTL[(o,d,t)]
                                                            for o,d,x,t in self.ship_idx}
        return cons

    def initial_shipment(self):
        dep = self.pre.sub.departure_allocation()
        cons = {f"{f, t}initial_shipments": LpAffineExpression([*((self.shipment[(f,d,i,t)] , -1) for d in dep[f] for i in self.sku_LOC.get(f, []) if i in self.sku_LOC.get(d, []))])
                                                                        == 0
                                                                        for f in self.i_d["fact_id"] + self.i_d["cw_id"]
                                                                        for t in self.time_id[0]}
        return cons

    def intial_inv(self):
        self.cw_ss, self.rw_ss = self.pre.compute_ss()
        self.cw_ss_d, self.rw_ss_d = self.cw_ss["Safety_Stock"].fillna(0).to_dict(), self.rw_ss["Safety_Stock"].fillna(0).to_dict()

        cons = {f"{w,'0', i}initial": LpAffineExpression([(self.inv_level[(i, w, "0")], 1)])
                                                         ==  self.cw_ss_d.get((w, i), self.rw_ss_d.get((w, i), 0)) + self.demand.get((t,i,w), 0) 
                                                         for i,w, t in self.inv_idx if t == self.time_id[1]}
        try:
            cons.update({f"{w,str(self.time_id[1]), i}initial":LpAffineExpression([(self.inv_level[(i, w, self.time_id[1])], 1)])
                                                                      ==  self.cw_ss_d.get((w, i), self.rw_ss_d.get((w, i), 0)) + self.demand.get((t,i,w), 0) 
                                                                            for i,w, t in self.inv_idx if t == self.time_id[2] if w in self.i_d["rw_id"]})
        except IndexError:
            pass
        return cons

    def st_echelon_int_inv(self):
        dep = self.pre.sub.departure_allocation()
        isf, _ = self.pre.sub.production()
        cons = {f"{f, t, i}1stech_InvBal": LpAffineExpression(((self.prod[(i, f, self.lt[i, t])], 1),
                                                                        *((self.shipment[(f,w,i,t)] , -1) for w in dep[f] if i in self.sku_LOC.get(w, []))))
                                                                        == self.inv_level[(i,f,t)] + self.demand.get((t,i,f), 0) - self.lost_sales.get((t,i,f), 0)- self.inv_level.get((i,f,self.prevt[t]), 0)
                                                                        for f in self.i_d["fact_id"]
                                                                        for i in isf.get(f, []) 
                                                                        for t in self.time_id[1: self.last_t[i]]}
        return cons

    def f2f_inv(self):
        f2f = self.pre.sub.f2f_skus()
        _, sku_plan = self.pre.sub.production()
        cons = {f"{f,t, i}1stech_InvBal": LpAffineExpression([*((self.shipment[(x, f, i, self.prevt[t])], 1) for x in sku_plan.get(i, []))])
                                                             == self.inv_level[(i,f,t)] + self.demand.get((t,i,f), 0) - self.lost_sales.get((t,i,f), 0) - self.inv_level.get((i, f, self.prevt[t]), 0)
                                                            for f in self.i_d["fact_id"]
                                                            for i in f2f.get(f, []) 
                                                            for t in self.time_id[1:self.last_t[i]]}
        return cons

    def nd_echelon_int_inv(self):
        int_sku, _ = self.pre.sub.cw_int_ext_skus()
        _, cw_to_rw,_ = self.pre.sub.arrival_allocation()
        _, sku_plan = self.pre.sub.production()

        cons = {f"{d,t, i}2ndech_InvBal":LpAffineExpression((*((self.shipment[(f,d,i,self.prevt[t])], 1) for f in sku_plan[i]),
                                                                        *((self.shipment[(d,w,i,t)], -1) for w in self.i_d["rw_id"] if d in cw_to_rw[w] and i in self.sku_LOC.get(w, []))))
                                                                        == self.inv_level[(i,d,t)] + self.demand.get((t,i,d), 0) - self.lost_sales.get((t,i,d), 0)- self.inv_level.get((i,d,self.prevt[t]), 0)
                                                                        for d in self.i_d["cw_id"]              
                                                                        for i in int_sku.get(d,[])
                                                                        for t in self.time_id[1:self.last_t[i]]}
        return cons

    def rd_echelon_int_inv(self):
        isf, _ = self.pre.sub.production()
        factory_to_rw, cw_to_rw, _ = self.pre.sub.arrival_allocation()
        int_sku, _ = self.pre.sub.rw_int_ext_skus()
        last_t_ = self.pre.sub.minimize_constraint()

        try:
            cons = {f"{w, t, i}3rdech_InvBal":LpAffineExpression((*((self.shipment[(f,w,i, self.prevt[t])], 1) for f in factory_to_rw[w] if i in isf.get(f, [])),
                                                                            *((self.shipment[(d,w,i,self.prevt[t])], 1) for d in cw_to_rw[w])))  
                                                                             == self.inv_level[(i,w,t)] + self.demand.get((t,i,w), 0)  - self.lost_sales.get((t,i,w), 0) - self.inv_level.get((i,w,self.prevt[t]), 0)
                                                                                for w in self.i_d["rw_id"]
                                                                                for i in int_sku.get(w, [])
                                                                                for t in self.time_id[2:last_t_[(w, i)]]}
        except KeyError as e:
            print(e.args)
        return cons

    def st_echelon_ext_inv(self):
        ext_sku = self.pre.sub.fact_ext_skus()
        supplier = self.pre.sub.supplier()
        cons = {f"{f,t,i}1stech_ext_sku_InvBal": LpAffineExpression([*((self.shipment[(e,f,i,self.prevt[t])], 1) for e in supplier[i])])
                                                                        == self.demand.get((t,i,f), 0) - self.lost_sales.get((t,i,f), 0) + self.inv_level[(i,f,t)] - self.inv_level.get((i,f,self.prevt[t]), 0)
                                                                        for f in self.i_d["fact_id"]
                                                                        for i in ext_sku.get(f, [])
                                                                        for t in self.time_id[1:self.last_t[i]]}
        cons = {}
        # try:
        #     for t in self.time_id[1:self.last_t[i]]:
        #         for i in ext_sku:
        #             for f in self.i_d["fact_id"]:
        #                 cons["{f,t,i}1stech_ext_sku_InvBal"] = LpAffineExpression([*((self.shipment[(e,f,i,self.prevt[t])], 1) for e in supplier[i])]) == self.demand.get((t,i,f), 0) - self.lost_sales.get((t,i,f), 0) + self.inv_level[(i,f,t)] - self.inv_level.get((i,f,self.prevt[t]), 0)
        # except KeyError:
        #     print(f, t, i)


        return cons

    def nd_echelon_ext_inv(self):
        supplier = self.pre.sub.supplier()
        _, ext_sku = self.pre.sub.cw_int_ext_skus()
        _, cw_to_rw, _ = self.pre.sub.arrival_allocation()
        cons = {f"{d,t,i}2ndech_ext_sku_InvBal": LpAffineExpression((*((self.shipment[(e,d,i, self.prevt[t])], 1) for e in supplier[i]),
                                                                    *((self.shipment[(d,w,i,t)], -1) for w in self.i_d["rw_id"] if d in cw_to_rw[w] and i in self.sku_LOC.get(w, []))))
                                                                        == self.demand.get((t,i,d), 0) - self.lost_sales.get((t,i,d), 0) + self.inv_level[(i,d,t)] - self.inv_level.get((i,d,self.prevt[t]), 0)
                                                                        for d in self.i_d["cw_id"]
                                                                        for i in ext_sku.get(d,[])
                                                                        for t in self.time_id[1:self.last_t[i]]}
        return cons

    def rd_echelon_ext_inv(self):
        supplier = self.pre.sub.supplier()
        direct_sh_extsku = self.pre.sub.direct_sh_cons()
        _, ext_sku = self.pre.sub.rw_int_ext_skus()
        last_t_ = self.pre.sub.minimize_constraint()
        cons = {f"{w,t,i}3rdech_ext_sku_InvBal": LpAffineExpression([*((self.shipment[(e,w,i,self.prevt[t])], 1) for e in supplier[i])])
                                                                    == self.demand.get((t,i,w), 0) - self.lost_sales.get((t,i,w), 0) + self.inv_level[(i,w,t)] - self.inv_level.get((i,w,self.prevt[t]), 0)
                                                                    for w in self.i_d["rw_id"] if direct_sh_extsku[w] == 1
                                                                    for i in ext_sku.get(w, [])
                                                                    for t in self.time_id[2:last_t_[(w, i)]]}
        cons.update({f"{w,t,i}3rdech_ext_sku_InvBal": LpAffineExpression([*((self.shipment[(d,w,i,self.prevt[t])], 1) for d in cw_to_rw[w])])
                                                                    == self.demand.get((t,i,w), 0) - self.lost_sales.get((t,i,w), 0) + self.inv_level[(i,w,t)] - self.inv_level.get((i,w,self.prevt[t]), 0)
                                                                    for w in self.i_d["rw_id"] if direct_sh_extsku[w] == 0
                                                                    for i in ext_sku.get(w, [])
                                                                    for t in self.time_id[2:last_t_[(w, i)]]})
        return cons

    def ss_req(self):
        self.cw_ss, self.rw_ss = self.pre.compute_ss()
        self.cw_ss_d, self.rw_ss_d = self.cw_ss["Safety_Stock"].fillna(0).to_dict(), self.rw_ss["Safety_Stock"].fillna(0).to_dict()

        p = [k for k,v in self.cw_ss_d.items() if v > 0]

        cons = {f"{d, t, i}2ndech_ssreq":LpAffineExpression([(self.inv_level[(i,d,t)], 1)]) 
                                                                            >= self.cw_ss_d.get((d, i), 0) - self.lost_sales.get((t,i,d), 0)
                                                                            for d, i in p
                                                                            for t in self.time_id[:self.last_t[i]]}
        p = [k for k,v in self.rw_ss_d.items() if v > 0]

        cons.update({f"{w, t, i}3rdech_ssreq":LpAffineExpression([(self.inv_level[(i,w,t)], 1)]) 
                                                                        >= self.rw_ss_d.get((w,i), 0) - self.lost_sales.get((t,i,w), 0)
                                                                        for w,i in p
                                                                        for t in self.time_id[:self.last_t[i]]})
        return cons

    def min_batch_size(self):
        intsku_f, _ = self.pre.sub.production()
        mbs = self.pre.sub.minimize_bin()
        big_M, big_M_alt = self.pre.sub.compute_M(self.data, mbs)
        cons = {f"{i, t}Min_batch_size": LpAffineExpression([(self.prod.get((i,f,t),0), 1)])
                                                                        >= self.bin_prod[(i,t)] * mbs[i]
                                                                        for i in list(big_M)
                                                                        for f in self.i_d["fact_id"] if i in intsku_f.get(f, [])
                                                                        for t in self.time_id[:self.last_t[i]]}
        cons.update({f"{i, t}Max_batch_size": LpAffineExpression([(self.prod.get((i,f,t),0), 1)]) 
                                                                         <= self.bin_prod[(i,t)] * big_M[i]
                                                                        for i in list(big_M)
                                                                        for f in self.i_d["fact_id"] if i in intsku_f.get(f, [])
                                                                        for t in self.time_id[:self.last_t[i]]})
        cons.update({f"{i, t}batch_size": LpAffineExpression([(self.prod.get((i,f,t),0), 1)]) 
                                                                         == self.bin_prod[(i,t)] * big_M_alt[i]
                                                                        for i in list(big_M_alt)
                                                                        for f in self.i_d["fact_id"] if i in intsku_f.get(f, [])
                                                                        for t in self.time_id[:self.last_t[i]]})
        return cons

    def build_biobj(self):
        self.cons_dict = self.build_model()
        self.epsilon = [0.51]
        self.service_level = pulp.LpVariable("service level")
        self.cons_dict.update(self.biobj_cons())


    def biobj_cons(self):
        cons = {"service_level_measure": LpAffineExpression((self.lost_sales[(i, w, t)], 1)
                                                                            for i,w,t in self.ls_idx) <= (1- self.service_level) * self.total_demand}
        cons.update({"epsilon": self.service_level >= 0})
        return cons

    def update_epsilon(self, model):
        while self.epsilon[-1] < 1.05:
            variables, model = LpProblem.from_json(folder + model_file + ".json")
            with open(folder + model_file + "constraints.json") as f:
                cons = ujson.load(f)
            self.call_cplex(model)
            print(self.inv_model.status)
            bi_obj_results[self.epsilon[-1]] = value(model.objective)
            self.epsilon.append(epsilon[-1] + 0.05)
            z_score = NormalDist().inv_cdf(min(epsilon[-1], 0.99999))
        print(bi_obj_results)


    





