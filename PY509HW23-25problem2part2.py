# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 09:31:52 2024

@author: MadMa
"""

# PY 509 HW23-25 - problem 2 compute r(t)

# compute r(t)
import numpy as np

# fct to define dr/dtau
def dr_dtau(r, E, M, c):
    return np.sqrt(E**2/c**2 - (1 - 2*M/r))

# Function to perform one step of the fourth-order Runge-Kutta method
def runge_kutta_step(r, E, M, c, h):
    k1 = h * dr_dtau(r, E, M, c)
    k2 = h * dr_dtau(r + 0.5 * k1, E, M, c)
    k3 = h * dr_dtau(r + 0.5 * k2, E, M, c)
    k4 = h * dr_dtau(r + k3, E, M, c)
    return r + (k1 + 2*k2 + 2*k3 + k4) / 6

# Function to solve for r as a function of tau using the fourth-order Runge-Kutta method
def solve_r_tau(E, M, c, r_initial, tau_final, num_steps):
    r_values = np.zeros(num_steps)
    tau_values = np.linspace(0, tau_final, num_steps)
    r_values[0] = r_initial
    h = tau_final / num_steps
    for i in range(1, num_steps):
        r_values[i] = runge_kutta_step(r_values[i-1], E, M, c, h)
    return r_values, tau_values

# Parameters
E = 1.0  # Energy per unit mass
M = 1.0  # Black hole mass (arbitrary units)
c = 1.0  # Speed of light (arbitrary units)
r_initial = 100.0  # Initial value of r
tau_final = 10.0  # Final value of proper time
num_steps = 1000  # Number of steps

# Solve for r as a function of tau
r_values, tau_values = solve_r_tau(E, M, c, r_initial, tau_final, num_steps)

# Print the results
for tau, r in zip(tau_values, r_values):
    print("Tau:", tau, "  r:", r)

import matplotlib.pyplot as plt
plt.plot(tau_values,r_values,'g')
plt.xlabel(r'$\tau$')
plt.ylabel(r'$r$')
plt.title(r'$r(\tau)$ vs $\tau$')
plt.grid(True)
plt.show()

# compute proper time
'''

import numpy as np
from mpmath import quad

# Define constants
G = 6.67430e-11  # [m^3 kg^-1 s^-2]
c = 299792458    # [m/s]
#c = 3e8
#M = 1e6 * 1.988e30  # black hole mass [kg]
#M = 1e8 * 1.988e30  # black hole mass [kg]
M = 1e10*2e30
# Function for the integrand
def integrand(r):
    return 1 / np.sqrt((1 - 2 * G * M / (c**2 * r)))

# Perform the integration using mpmath
tau, _ = quad(integrand, [np.inf, 2 * G * M / c**2], error=True)

# Convert the result to seconds
tau_seconds = float(tau) * (G * M / c**3)

print("Proper time to pass through the horizon:", tau_seconds, "seconds")
'''
