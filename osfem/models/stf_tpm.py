"""
 Title:         STF TPM
 Description:   Theta Projection Model 
 Author:        Janzen Choi

"""

# Libraries
from osfem.models.__model__ import __Model__

# Model class
class Model(__Model__):

    def initialise(self):
        """
        Runs at the start, once
        """
        self.add_param("a", r"$a$", -2, 0, lambda x : -10**x, (-1,0))
        self.add_param("b", r"$b$", -3, 0, lambda x : 10**x,  (0,0.1))
        self.add_param("c", r"$c$", -5, 0, lambda x : 10**x,  (0,0.001))
        self.add_param("d", r"$d$", -6, 0, lambda x : -10**x, (-0.0001,0))
    
    def evaluate(self, a, b, c, d) -> float:
        """
        Evaluates the model

        Parameters:
        * `...`: Parameters
        
        Returns the response
        """
        s = self.get_field("stress")
        t = self.get_field("temperature")
        a = -10**a
        b = 10**b
        c = 10**c
        d = -10**d
        stf = a + b*s + c*t + d*s*t
        return stf
