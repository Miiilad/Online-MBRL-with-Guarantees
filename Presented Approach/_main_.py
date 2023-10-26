from __future__ import division
import numpy as np
# from vpython import *

import matplotlib.pyplot as plt

from math import atan2
from scipy.integrate import solve_ivp
import os
from shutil import copy

from model import Pendulum
import SOL
# import graphics
import utils
import sys

import time

# Choose the Identification algorithm
select_ID_algorithm = {'SINDy': 1, 'RLS': 0, 'Least Squares': 0,
                       'Gradient Descent': 0}  # options={'SINDy','RLS'}
select_output_graphs = {'states': 1, 'value': 0, 'P': 0, 'error': 0, 'runtime': 0}
# To be used for labeling the output results' directory
version = 7
script_path = os.path.dirname(os.path.realpath(__file__))
output_dir_path = script_path + '/Results_Poly_{}_{}'.format(version, list(select_ID_algorithm.keys())[
    list(select_ID_algorithm.values()).index(1)])
try:
    os.makedirs(output_dir_path)
except FileExistsError:
    print('The directory already exists')
# save a copy of the script together with the results
copy(__file__, output_dir_path)
copy(script_path + '/SOL.py', output_dir_path)
copy(script_path + '/model.py', output_dir_path)
copy(script_path + '/graphics.py', output_dir_path)
copy(script_path + '/utils.py', output_dir_path)

# system specifications
Model = Pendulum()
n = Model.dim_n
m = Model.dim_m
Domain = Model.Domain

pi = np.pi
# choosing and initaializing the Librery of bases
chosen_bases = ['1','x', 'sinx']
CMAC_setting = {'receptor_type': 'Gaussian', 'num_of_receptors': [20, 20],
                'domain_of_interest': np.array([[-2 * pi, 2 * pi], [-2 * pi, 2 * pi]]), 'variance': [0.15, 0.15]}
measure_dim = Model.measure_dim
Lib = SOL.Library(chosen_bases, CMAC_setting, measure_dim, m)
p = Lib._Phi_dim
Theta_dim = (m + 1) * p
print('Set of Bases:', Lib._Phi_lbl)
# Building a Database
db_dim = 2000
Database = SOL.Database(db_dim=db_dim, Theta_dim=Theta_dim,
                        output_dir_path=output_dir_path, Lib=Lib, load=0, save=0)
X_buf = np.zeros((measure_dim, 5))
U_buf = np.zeros((m, 3))

# Define and initialize the Identification System Model
SysID_Weights = np.ones((measure_dim, Theta_dim)) * 0.001
SysID = SOL.SysID(select_ID_algorithm, Database, SysID_Weights, Lib)
SysID_Weights = SysID.Weights
# dx_dt= partial(double_pend_cart.dynamics, u=0)
num_of_episode = 100
# duration of simulation
t_end = 15
# sampling time
h = 0.01  # 0.01
# time grid points
t = np.arange(0, t_end + h, h)

# To characterise the performance measure
R = np.diag([2])
Q = np.diag([1, 1])
gamma = 0.6
Obj = SOL.Objective(Q, R, gamma)

# Tracking reference
x_ref = np.zeros((measure_dim))

# Define the controller
Controller = SOL.Control(h, Objective=Obj, Lib=Lib, P_init=np.zeros((p, p)))
u_lim = [200]

# Simulation results
Sim = SOL.SimResults(t, Lib, Database, SysID, Controller,
                     select=select_output_graphs, output_dir_path=output_dir_path)

for j in range(num_of_episode):
    # Initialize the system and controller
    u = np.zeros(m)
    x_init = Model.randomly_initialize()
    # if j == 0:
    #     Controller.P = np.zeros((p, p))
    Controller.P = np.zeros((p, p))
    x_dot_approx = np.zeros(measure_dim)
    x_dot_hat = np.zeros(measure_dim)
    error = np.zeros(len(t))
    runtime = np.zeros(2)
    for i in range(len(t) - 2):

        Model.integrate(u, [t[i], t[i + 1]])
        sample = Model.Read_sensor()
        x_s = sample - x_ref

        X_buf = np.roll(X_buf, -1)
        X_buf[:, -1] = x_s
        U_buf = np.roll(U_buf, -1)
        U_buf[:, -1] = u

        if not Model.In_domain(sample):
            print('OUT OF DOMAIN')
            break

        if i > 3:
            x_dot_approx = utils.x_dot_approx(X_buf, h)
            # x_dot_approx = Model.dynamics(0, X_buf[:, 2], U_buf[:, 0][0])
            x_dot_hat = SysID.evaluate(X_buf[:, 2], U_buf[:, 0])
            error[i] = np.linalg.norm(x_dot_approx - x_dot_hat)
            if 1 :
                Database.add(X_buf[:, 2], x_dot_approx, U_buf[:, 0])
                t_start = time.time()
                SysID_Weights = SysID.update(
                    X_buf[:, 2], x_dot_approx, U_buf[:, 0])
                runtime[0] = time.time()-t_start
                
        t_start = time.time()
        if j<10: Controller.integrate_P_dot(x_s, SysID_Weights, k=1, sparsify=False)
        u_0 = Controller.calculate(x_s, SysID_Weights, u_lim)
        u_0 += 0.01 * np.sin(50 * t[i])
        runtime[1] = time.time()-t_start

        Sim.record(i, sample, u_0, Controller.P,
                   Controller.value(x_s), error[i], x_ref, runtime)
        u = np.copy(u_0)
        

        if i % 150 == 0:
            print('u=', u)
            print('progress={:4.1f}%'.format(i / len(t) * 100))

    Database.DB_save()
    SysID.save()
    Sim.graph(j, i)
    Sim.printout(j)
    print("Initial condition: ", x_init, "\n")

    np.save(output_dir_path + '/P_verify.npy', Controller.P)
    np.save(output_dir_path + '/W_Wj.npy', SysID_Weights)

