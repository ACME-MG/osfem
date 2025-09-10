"""
 Title:         TTF LLM
 Description:   Logarithmic Larson-Miller
 Author:        Janzen Choi

"""

# Libraries
import numpy as np
from osfem.models.__model__ import __Model__

# Model class
class Model(__Model__):

    def initialise(self):
        """
        Runs at the start, once
        """
        self.add_param("a", r"$a$", 0, 1e4)
        self.add_param("b", r"$b$", 0, 1e3)
        self.add_param("c", r"$c$", 0, 1e5)
        self.add_param("C", r"$C$", 0, 1e1)
    
    def evaluate(self, a, b, c, C) -> float:
        """
        Evaluates the model

        Parameters:
        * `...`: Parameters
        
        Returns the response
        """
        s = self.get_field("stress")
        t = self.get_field("temperature")
        p = -a*np.log(s) - b*s + c
        ttf = np.exp(p/t-C)
        return ttf
