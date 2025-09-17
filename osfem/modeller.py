"""
 Title:         Optimise
 Description:   Functions for quick optimisation
 Author:        Janzen Choi

"""

# Libraries
import time
import numpy as np
import matplotlib.pyplot as plt
from osfem.models.__model__ import get_model
from osfem.general import round_sf, get_file_path_exists, safe_mkdir
from osfem.plotter import create_1to1, plot_1to1, lobf_1to1, save_plot, CAL_COLOUR, VAL_COLOUR

# PYMOO libraries
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize

# Define available fields
FIELD_INFO_DICT = {
    "ttf": {"scale": 1/3600, "units": "h",     "limits": (0,10000), "label": r"$t_{f}$"},
    "stf": {"scale": 1,      "units": "mm/mm", "limits": (0,0.6),   "label": r"$\epsilon_{f}$"},
    "mcr": {"scale": 1,      "units": "1/h",   "limits": (0,8e-8),  "label": r"$\dot{\epsilon}_{m}$"},
}

# Modeller class
class Modeller:

    def __init__(self, model_name:str):
        """
        Initialises the model
        """

        # Initialise
        self.model_name = model_name
        self.model = get_model(model_name)
        self.field = model_name.split("_")[0]
        self.opt_params = None

        # Determine results path
        parent_path = "results"
        time_stamp = time.strftime("%y%m%d%H%M%S", time.localtime(time.time()))
        output_dir = f"{parent_path}/{time_stamp}_{self.model_name}"
        safe_mkdir(parent_path)
        safe_mkdir(output_dir)
        self.get_output_path = lambda x : f"{output_dir}/{x}"

    def optimise(self, data_list:list, num_gens:int=1000, population:int=100,
                 offspring:int=100, crossover:float=0.9, mutation:float=0.1) -> list:
        """
        Runs optimisation

        Parameters:
        * `data_list`:  List of data dictionaries
        * `num_gens`:   Number of generations
        * `population`: Population size
        * `offspring`:  Number of offspring per generation
        * `crossover`:  Crossover probability
        * `mutation`:   Mutation probability

        Returns the optimal parameters
        """

        # Define the objective function
        fit_list = [data[self.field] for data in data_list]
        def obj_func(params):
            prd_list = self.model.evaluate_data(data_list, params)
            error = [abs((f-p)/f) for f, p in zip(fit_list, prd_list)]
            return np.average(error)

        # Prepare parameter bounds
        param_names = self.model.get_param_names()
        param_dict = self.model.get_param_dict()
        l_bounds = tuple([param_dict[pn]["l_bound"] for pn in param_names])
        u_bounds = tuple([param_dict[pn]["u_bound"] for pn in param_names])

        # Define algorithm
        algorithm = NSGA2(
            pop_size     = population,
            n_offsprings = offspring,
            crossover    = SBX(prob=crossover, prob_var=1.0),
            mutation     = PolynomialMutation(prob=mutation),
            eliminate_duplicates = True
        )

        # Define problem
        class Problem(ElementwiseProblem):
            def __init__(self):
                super().__init__(n_var=len(l_bounds), n_obj=1, xl=l_bounds, xu=u_bounds)
            def _evaluate(self, params, out, *args, **kwargs) -> None:
                out["F"] = obj_func(params)
        problem = Problem()
        results = minimize(problem, algorithm, ("n_gen", num_gens), verbose=False, seed=None)

        # Identify optimal parameters
        opt_array = results.X
        opt_array = opt_array[0] if opt_array.ndim == 2 else opt_array
        self.opt_params = list(opt_array)

        # Return
        return round_sf(self.opt_params, 5)

    def plot_1to1(self, cal_data_list:list, val_data_list:list, params:list=None) -> None:
        """
        Creates a 1-to-1 plot

        Parameters:
        * `cal_data_list`: List of calibration datasets
        * `val_data_list`: List of validation datasets
        * `params`:        List of parameters
        * `limits`:        Define limits for the plot
        * `exp`:           The exponent to scale the ticks
        """
    
        # Get parameters
        params = self.opt_params if params == None else params
        
        # Get all calibration and validation data
        cal_fit_list, cal_prd_list = self.evaluate(cal_data_list, params)
        val_fit_list, val_prd_list = self.evaluate(val_data_list, params)

        # Initialise plot
        field_info = FIELD_INFO_DICT[self.field]
        limits = field_info["limits"]
        create_1to1(field_info["label"], field_info["units"], limits)

        # Plot data for each temperature
        all_temps = sorted(list(set([d["temperature"] for d in cal_data_list+val_data_list])))
        for temp, marker in zip(all_temps, ["^", "s", "*"]):
            cal_data_sublist = [cd for cd in cal_data_list if cd["temperature"] == temp]
            val_data_sublist = [vd for vd in val_data_list if vd["temperature"] == temp]
            cal_fit_sublist, cal_prd_sublist = self.evaluate(cal_data_sublist, params)
            val_fit_sublist, val_prd_sublist = self.evaluate(val_data_sublist, params)
            plot_1to1(cal_fit_sublist, cal_prd_sublist, CAL_COLOUR, marker)
            plot_1to1(val_fit_sublist, val_prd_sublist, VAL_COLOUR, marker)
        
        # Plot LOBFs
        lobf_1to1(cal_fit_list, cal_prd_list, CAL_COLOUR, limits)
        lobf_1to1(val_fit_list, val_prd_list, VAL_COLOUR, limits)

        # Add legend for temperatures
        bbox_pos = (0.0, 0.835)
        t800  = plt.scatter([], [], color="none", edgecolor="black", linewidth=1, label="800°C",  marker="^", s=10**2)
        t900  = plt.scatter([], [], color="none", edgecolor="black", linewidth=1, label="900°C",  marker="s", s=10**2)
        t1000 = plt.scatter([], [], color="none", edgecolor="black", linewidth=1, label="1000°C", marker="*", s=10**2)
        legend = plt.legend(handles=[t800, t900, t1000], framealpha=1, edgecolor="black", fancybox=True, facecolor="white", fontsize=12, loc="upper left", bbox_to_anchor=bbox_pos)
        plt.gca().add_artist(legend)
        
        # Save plot
        output_path = self.get_output_path("1to1")
        output_path = get_file_path_exists(output_path, "png")
        save_plot(output_path)

    def get_are(self, data_list:list, params:list=None) -> float:
        """
        Calculates the average relative errors

        Parameters:
        * `data_list`: List of datasets
        * `params`:    List of parameters
        
        Returns the average relative error
        """
        params = self.opt_params if params == None else params
        fit_list, prd_list = self.evaluate(data_list, params)
        are = np.average([abs((f-p)/f) for f, p in zip(fit_list, prd_list)])
        return f"{round_sf(are*100, 5)}%"

    def evaluate(self, data_list:list, params:list) -> tuple:
        """
        Evaluates the model and scales the outputs

        Parameters:
        * `data_list`: List of data dictionaries
        * `params`:    List of parameters

        Returns the fitting and predicted lists
        """
                
        # Get calibration and validation data
        fit_list = [data[self.field] for data in data_list]
        prd_list = self.model.evaluate_data(data_list, params)
        
        # Scale data
        field_info = FIELD_INFO_DICT[self.field]
        scale = lambda x_list : [x*field_info["scale"] for x in x_list]
        fit_list = scale(fit_list)
        prd_list = scale(prd_list)
        
        # Return
        return fit_list, prd_list

    def get_info(self, field:str) -> list:
        """
        Quickly gets information from the model's parameters

        Parameters:
        * `field`: The field of the parameter

        Returns a list of the information
        """
        param_dict = self.model.get_param_dict()
        param_names = self.model.get_param_names()
        info_list = [param_dict[pn][field] for pn in param_names]
        return info_list
