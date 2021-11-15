from Preprocess import Preprocess, Initialisation
from Model import InventoryModel
from Solvers import CplexSolver, GurobiSolver
from Postprocess import ExportVariables
from Recycle import Recycle


def main():
    ini = Initialisation()
    rec = Recycle()
    new_model = rec.recycle_model()
    if new_model is True:
        print("True")
        pre = Preprocess()
        model = InventoryModel(pre)
        inv_model = model.build_model()
        constraints, indices = model.cons_dict, model.i_d

    if ini.biobj:
        pre = Preprocess()
        model = InventoryModel()
        biobj = model.build_biobj()

    if new_model is not True:
        constraints, variables, inv_model = new_model
        indices = rec.i_d

    if ini.config_dict["cplex"]:
        cplex = CplexSolver(inv_model, ini)
        exp = ExportVariables(model, indices, ini)
        cplex.call_cplex()
        rec.save_model_json(inv_model, constraints, ask=True)
        exp.export_solutions()




if __name__ == '__main__':
    main()
