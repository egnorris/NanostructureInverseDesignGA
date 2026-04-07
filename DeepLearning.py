import Support
import numpy as np
global defaultKwargs
global keywords
defaultKwargs = Support.defaultKwargs
keywords = Support.keywords

class DeepLearning():
    def __init__(self, **kwargs):
        self.l = Support.getkwarg(kwargs, defaultKwargs["l"], keywords["l"])
        self.m = Support.getkwarg(kwargs, defaultKwargs["m"], keywords["m"])
        self.f = Support.getkwarg(kwargs, defaultKwargs["f"], keywords["f"])

        """
            Input Error Handling
        """
        #check that degree, order, and field are input as lists
        if type(self.l) != type([]):
            raise Exception(f'degree, l must be a list; {type(self.l)} != {type([])}')
        elif type(self.m) != type([]):
            raise Exception(f'order, m must be a list; {type(self.m)} != {type([])}')
        elif type(self.f) != type([]):
            raise Exception(f'field, f must be a list; {type(self.f)} != {type([])}')
        
        #check that at least one and not more than two fields are selected
        if (len(self.f) < 1) or (len(self.f) > 2):
            raise Exception(f'{len(self.f)} fields selected; choose at least one and no more than two')
        #check that the length of degree and order match
        if len(self.l) != len(self.m):
            raise Exception(f'length of degree, l and order, m must match; {len(self.l)} != {len(self.m)}')

        #allow alternative names for fields from list
        electricFieldNames = ["E", "e", "elec", "electric", "Elec", "Electric"]
        magneticFieldNames = ["H", "M", "h", "m", "mag", "magnetic", "Mag", "Magnetic"]
        for i in range(len(self.f)):
            if self.f[i] in electricFieldNames:
                self.f[i] = "E"
            elif self.f[i] in magneticFieldNames:
                self.f[i] = "H"
        if (self.f[i] != "E") and (self.f[i] != "H"):
            raise Exception(f'f[{i}]={self.f[i]} is invalid; Allowed Field types are "E" and "H"')

        #check that the elements of the degree and order lists are allowed
        for i in range(len(self.l)):
            
            #elements of degree and order lists must be integers
            if type(self.l[i]) != type(1):
                raise Exception(f'degree, l must be a list of integers; l[{i}]: {type(self.l[i])} != {type(1)}')
            elif type(self.m[i]) != type(1):
                raise Exception(f'order, m must be a list of integers; m[{i}]: {type(self.m[i])} != {type(1)}')

            #0 < l < 3 and  0 < m < l+1 
            if (self.l[i] < 1) or (self.l[i] > 2):
                raise Exception(f'order, (l,m) ({self.l[i]}, {self.m[i]}) is invalid,   0 < l < 3')
            elif (self.m[i] > self.l[i]):
                raise Exception(f'order, (l,m) ({self.l[i]}, {self.m[i]}) is invalid,  0 < m <= l')
        

            
    
            



        

        




