from Preprocess import Initialisation
from pulp import *
import cplex


class CplexSolver():
    """docstring for Cplex"""

    def __init__(self, model, ini):
        self.model = model
        self.ini = ini
        self.config_dict = self.ini.config_dict

    def call_cplex(self):
        param = self.config_dict["cplex"]
        solver = CPLEX_PY()

        solver.buildSolverModel(self.model)
        try:
            solver.solverModel.parameters.emphasis.memory.set(int(param["memory_emphasis"]))
            solver.solverModel.parameters.workmem.set(int(param["working_memory"]))
            solver.solverModel.parameters.mip.strategy.file.set(int(param["node_file"]))
            solver.solverModel.parameters.mip.cuts.cliques.set(int(param["cuts_clique"]))
            solver.solverModel.parameters.mip.cuts.covers.set(int(param["cuts_covers"]))
            solver.solverModel.parameters.mip.cuts.flowcovers.set(int(param["cuts_flowcovers"]))
            solver.solverModel.parameters.mip.cuts.gomory.set(int(param["cuts_gomory"]))
            solver.solverModel.parameters.mip.cuts.gubcovers.set(int(param["cuts_gubcovers"]))
            solver.solverModel.parameters.mip.cuts.implied.set(int(param["cuts_implied"]))
            solver.solverModel.parameters.mip.cuts.mircut.set(int(param["cuts_mircut"]))
            solver.solverModel.parameters.mip.cuts.pathcut.set(int(param["cuts_path"]))
            solver.solverModel.parameters.mip.limits.cutsfactor.set(int(param["cuts_factor"]))
            solver.solverModel.parameters.mip.strategy.branch.set(int(param["branch_strategy"]))
            solver.solverModel.parameters.mip.strategy.probe.set(int(param["strategy_probe"]))
            solver.solverModel.parameters.mip.tolerances.mipgap.set(float(param["mipgap"]))
            solver.callSolver(self.model)
        except cplex.exceptions.errors.CplexSolverError:
            print("One of the Cplex parameters specified is invalid. ")
        status = solver.findSolutionValues(self.model)
        solver.solverModel.parameters.conflict.display.set(2)  # if model is unsolvable will display the problematic constraint(s)

    def call_cplex_tuning_tool(self):
        param = self.config_dict["cplex"]

        solver = CPLEX_PY()
        solver.buildSolverModel(self.model)
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


class GurobiSolver():
    """docstring for GurobiSolver"""

    def __init__(self, model):
        super(GurobiSolver, self).__init__()
        self.model = model

    def call_gurobi(self):
        solver = GUROBI(epgap=0.05, cuts=2, presolve=1)
        solver.buildSolverModel(self.model)
        solver.callSolver(self.model)
        status = solver.findSolutionValues(self.model)
        
