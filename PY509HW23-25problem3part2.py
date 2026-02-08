# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 09:18:48 2024

@author: MadMa
"""

# PY 509 HW23-25 - tau for R(t) = 0 (taken to be close to 0)

import numpy as np
import warnings
from scipy.integrate import solve_ivp

# Constants
G = 6.674e-11 # [m^3/kg/s^2]
c = 3e8 # [m/s]
M_sun = 1.989e30 # [kg]
R_sun = 6.957e8   # [m]

# Suppress the warning for negative square root
warnings.filterwarnings("ignore")

# Initial conditions
R0 = R_sun
t0 = 0.0
y0 = [t0, R0]

# Define the ODE system for collapse
def oppenheimer_snyder(t, y):
    t, R = y
    dRdt = -np.sqrt((8*np.pi*G/3)*c**2*R**3)
    return [1, dRdt]

# Function to stop integration when R(t) is close to zero
def event_horizon(t, y):
    return y[1] - 1e-10

# Integrate the ODE system until R(t) is close to zero
solution = solve_ivp(oppenheimer_snyder, [0, 1e10], y0, events=event_horizon)

# Proper time is the first element of the solution
proper_time = solution.t_events[0][0]

print("Proper time when R(t) is close to zero:", proper_time, "seconds")

