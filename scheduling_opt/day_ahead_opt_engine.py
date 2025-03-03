import gurobipy as gp
from gurobipy import GRB
import numpy as np

class BoilerScheduler:
    
    def __init__(self,
                 boiler_model_params:dict,
                 max_power:float, 
                 min_power:float, 
                 max_storage:float, 
                 min_storage:float,
                 horizon:int,
                 lambda_reg:float):
        
        self.max_power = max_power
        self.min_power = min_power
        self.max_storage = max_storage
        self.min_storage = min_storage
        self.lambda_reg = lambda_reg
        self.T = horizon 
        
        self.b1 = boiler_model_params.get('b1')  
        self.b2 = boiler_model_params.get('b2')
        self.a1 = boiler_model_params.get('a1')
        self.a2 = boiler_model_params.get('a2')
        
        self.opt_power_schedule = None
        self.opt_heat_storage_schedule = None
        
    def step(self, 
             horizon:int, 
             heat_demand:np.array,
             da_price:np.array,
             initial_storage_t0=1.8,
             initial_storage_t1=1.78,
             initial_power=0,
             verbose:bool=False,
             **kwargs):

        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        
        # Setup dimensions
        self.T = horizon
        n_hours = len(da_price)
        
        # -------------------- MODEL --------------------
        model = gp.Model("BoilerScheduler")
        
        if not verbose:
            model.setParam('OutputFlag', 0)
        
        # -------------------- VARIABLES --------------------
        power_boiler = model.addVars(self.T, lb=self.min_power, ub=self.max_power, name="power_boiler")
        energy_da = model.addVars(n_hours, name="energy_day_ahead")
        heat_storage = model.addVars(self.T, lb=0, name="heat_storage")
        
        power_diff = model.addVars(self.T-1, lb=-GRB.INFINITY, name="power_diff")
        abs_diff = model.addVars(self.T-1, lb=0, name="abs_diff")
        
        # -------------------- INITIAL CONDITIONS --------------------
        model.addConstr(heat_storage[0] == initial_storage_t0)
        model.addConstr(heat_storage[1] == initial_storage_t1)
        model.addConstr(power_boiler[0] == initial_power)
        
        # -------------------- DYNAMIC CONSTRAINTS --------------------
        for t in range(2, self.T):
            model.addConstr(
                heat_storage[t] == self.a1 * heat_storage[t-1] + 
                self.a2 * heat_storage[t-2] + 
                self.b1 * power_boiler[t] + 
                self.b1 * power_boiler[t-1]
            )
            model.addConstr(heat_storage[t] >= heat_demand[t])
        
        # -------------------- POWER DIFFERENCE CONSTRAINTS --------------------
        for t in range(1, self.T):
            model.addConstr(power_diff[t-1] == power_boiler[t] - power_boiler[t-1])
            model.addGenConstrAbs(abs_diff[t-1], power_diff[t-1], f"abs_diff_{t}")
        
        # -------------------- OBJECTIVE FUNCTION --------------------
        power_cost = gp.quicksum(power_boiler[t] for t in range(self.T))
        smoothness_penalty = abs_diff.sum()
        model.setObjective(power_cost + self.lambda_reg * smoothness_penalty, GRB.MINIMIZE)
        
        # -------------------- OPTIMIZE --------------------
        model.optimize()
        
        # -------------------- RESULTS --------------------
        self.opt_power_schedule = np.array([v.X for v in power_boiler.values()])
        self.opt_heat_storage_schedule = np.array([v.X for v in heat_storage.values()])
        
        return self.opt_power_schedule, self.opt_heat_storage_schedule

