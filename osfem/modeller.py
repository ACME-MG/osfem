"""
 Title:         Optimise
 Description:   Functions for quick optimisation
 Author:        Janzen Choi

"""

# Libraries
import inspect
import numpy as np
import osfem.models as models
from osfem.general import round_sf
from osfem.plotter import create_1to1_plot, save_plot
from scipy.optimize import minimize

# Define available fields
FIELD_INFO_DICT = {
    "ttf": {"scale": 1/3600, "units": "h",     "label": r"$t_{f}$"},
    "stf": {"scale": 1,      "units": "mm/mm", "label": r"$\epsilon_{f}$"},
    "mcr": {"scale": 1,      "units": "1/h",   "label": r"$\dot{\epsilon}_{min}$"},
}

# Model class
class Model:

    def __init__(self, model_name:str):
        """
        Initialises the model
        """
        self.model_name = model_name
        self.model = get_model(model_name)
        self.field = model_name.split("_")[0]
        self.opt_params = None

    def optimise(self, data_list:list, init_params:list=None):
        """
        Runs optimisation

        Parameters:
        * `data_list`:   List of data dictionaries
        * `init_params`: Initial values for the parameters
        """

        # Define the objective function
        def obj_func(params):
            fit_list = [data[self.field] for data in data_list]
            prd_list = [self.model(data, *params) for data in data_list]
            sqr_err  = [(f-p)**2 for f, p in zip(fit_list, prd_list)]
            return np.average(sqr_err)

        # Define initial values if undefined
        if init_params == None:
            sig = inspect.signature(self.model)
            arguments = list(sig.parameters.keys())
            num_params = len(arguments)-1
            init_params = [1.0]*num_params

        # Optimise and check success
        results = minimize(obj_func, init_params, method="L-BFGS-B")
        if not results.success:
            print("Error! Optimisation unsuccessful!")

        # Save optimised parameters and return
        self.opt_params = list(results.x)
        return round_sf(self.opt_params, 5)

    def plot_1to1(self, cal_data_list:list, val_data_list:list,
                  params:list=None, limits:tuple=None) -> None:
        """
        Creates a 1-to-1 plot

        Parameters:
        * `cal_data_list`: List of calibration datasets
        * `val_data_list`: List of validation datasets
        * `params`:        List of parameters
        * `limits`:        Define limits for the plot
        """
        
        # Define parameters
        params = self.opt_params if params == None else params
        
        # Get calibration and validation data
        cal_fit_list, cal_prd_list = self.evaluate_2(cal_data_list, params)
        val_fit_list, val_prd_list = self.evaluate_2(val_data_list, params)

        # Create the 1to1 plot
        field_info = FIELD_INFO_DICT[self.field]
        create_1to1_plot(cal_fit_list, val_fit_list, cal_prd_list, val_prd_list,
                         field_info["label"], field_info["units"], limits)
        save_plot(f"results/{self.model_name}.png")
    
    def get_are(self, data_list:list, params:list=None) -> float:
        """
        Calculates the average relative errors

        Parameters:
        * `data_list`: List of datasets
        * `params`:    List of parameters
        
        Returns the average relative error
        """
        params = self.opt_params if params == None else params
        fit_list, prd_list = self.evaluate_2(data_list, params)
        are = np.average([abs((f-p)/f) for f, p in zip(fit_list, prd_list)])
        return f"{round_sf(are, 5)*100}%"

    def evaluate_2(self, data_list:list, params:list) -> tuple:
        """
        Evaluates the model and scales the outputs

        Parameters:
        * `data_list`: List of data dictionaries
        * `params`:    List of parameters

        Returns the fitting and predicted lists
        """
                
        # Get calibration and validation data
        fit_list = [data[self.field] for data in data_list]
        prd_list = self.evaluate(data_list, params)
        
        # Scale data
        field_info = FIELD_INFO_DICT[self.field]
        scale = lambda x_list : [x*field_info["scale"] for x in x_list]
        fit_list = scale(fit_list)
        prd_list = scale(prd_list)
        
        # Return
        return fit_list, prd_list

    def evaluate(self, data_list:list, params:list) -> list:
        """
        Evaluates the model

        Parameters:
        * `data_list`: List of data dictionaries
        * `params`:    List of parameters

        Returns the model's outputs
        """
        if params == None:
            raise ValueError("Parameters have not been defined or optimised!")
        prd_list = [self.model(data, *params) for data in data_list]
        return prd_list
    
def get_model(model_name:str):
    """
    Gets the model

    Parameters:
    * `model_name`: The name of the model

    Returns the model
    """

    # Check if model exists
    model_names = [name for name, _ in inspect.getmembers(models, inspect.isfunction)]
    if not model_name in model_names:
        raise ValueError(f"No model named '{model_name}'!")

    # Return model
    model = getattr(models, model_name, None)
    return model
