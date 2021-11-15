import pandas as pd
from statistics import NormalDist



class SafetyStock():
    """docstring for SafetyStock"""

    def __init__(self, data, loc_data, central_w, regional_w, config):
        super().__init__()
        self.data = data
        self.loc_data = loc_data
        self.central_w = central_w
        self.regional_w = regional_w
        self.config = config
        self.service_level = NormalDist().inv_cdf(float(self.config["model"]["service_level"]))

        self.op = {"pallets": ["mean", "std"],
                   "lead_time": "first",
                   "std_lead_time": "first"}

    def ss_allocation(self):
        cw_ss = self.compute_cw_ss()
        rw_ss = self.compute_rw_ss()
        return cw_ss, rw_ss

    def central_allocation(self):
        cent_wh_alloc = {}
        # store mean and std of demand aggregated per central warehouse and respective reg. warehouse
        cw_ss = pd.DataFrame()
        rw_ss = pd.DataFrame()
        for c_wh in self.central_w:
            # extracting whih central warehosue is responsible for which regional warehouse
            cent_wh_alloc[c_wh] = [
                c_wh] + self.loc_data.loc[self.loc_data["resp. central WH"] == c_wh].index.tolist()
            # computing demand mean and std accroding to the demand at that cw and the rw it oversees
            stats = self.data[self.data["location"].isin(cent_wh_alloc[c_wh])].groupby(
                "SKU_id", as_index=False).agg(self.op)
            stats["location"] = c_wh
            cw_ss = cw_ss.append(stats)
        return cw_ss

    def compute_cw_ss(self):
        cw_ss = self.central_allocation()
        cw_ss.set_index(["location", "SKU_id"], inplace=True)
        cw_ss.columns = ["Pallets_mean", "Pallets_std",
                         "lead_time_mean", "lead_time_std"]  # renaming the columns
        cw_ss["Safety_Stock"] = self.service_level * (cw_ss["lead_time_mean"] * cw_ss["Pallets_std"]**2
                                                      + cw_ss["Pallets_mean"]**2 * cw_ss["lead_time_std"]**2)**0.5
        return cw_ss

    def regional_allocation(self):
        rw_ss = pd.DataFrame()
        for r_wh in self.regional_w:
            stats = self.data[self.data["location"] == r_wh].groupby(
                "SKU_id", as_index=False).agg(self.op)
            stats["location"] = r_wh
            rw_ss = rw_ss.append(stats)
        return rw_ss

    def compute_rw_ss(self):
        rw_ss = self.regional_allocation()
        rw_ss.set_index(["location", "SKU_id"], inplace=True)
        rw_ss.columns = ["Pallets_mean", "Pallets_std", "lead_time_mean", "lead_time_std"] # renaming the columns
        rw_ss["Safety_Stock"] = self.service_level * (rw_ss["lead_time_mean"] * rw_ss["Pallets_std"]**2)**0.5
        return rw_ss

