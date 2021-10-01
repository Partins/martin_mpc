#!/usr/bin/env python3

from numpy.lib.nanfunctions import _divide_by_count
import opengen as og
import casadi.casadi as cs
import matplotlib.pyplot as plt
import numpy as np

def lin_dynamics_ct(x, u):
    g = 9.80
    Ax = 0.01
    Ay = 0.01
    Az = 0.01
    gravity = 9.80
    tau_roll = 0.05 # Time constant
    tau_pitch = 0.05 # Time constant
    K_roll = 0.15  # Roll angle gain
    K_pitch = 0.15 # Pitch angle gain

    dx      = x[3]
    dy      = x[4]
    dz      = x[5]
    ddx     = -Ax * x[3]    - g * x[7]
    ddy     = -Ay * x[4]    - g * x[6] 
    ddz     = -Az * x[5]    + u[2] 
    dphi    = -1/tau_roll   + K_roll/tau_roll * u[0]
    dtheta  = -1/tau_pitch  + K_pitch/tau_pitch * u[1]

    return [dx, dy, dz, ddx, ddy, ddz, dphi, dtheta]

def lin_dynamics_dt(x, u, Ts):
    dx = lin_dynamics_ct(x,u)
    return [x[i] + Ts * dx[i] for i in range(8)]

def stage_cost(x, u, Q, R):
    x_ref = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    u_ref = [0.0, 0.0, 9.8]
    cost = Q[0]*(x[0]-x_ref)**2
    for k in range(1,8):
        cost += Q[k] * (x[k]-x_ref[k])**2
    for i in range(3):
        cost += R[i]*(u[i]-u_ref[i])**2
    return cost
# System model
# ------------------------------------

# Build parametric optimizer
# ------------------------------------
RATE = 20
(nu, nx, N) = (3, 8, 20)
ts = 1/RATE

x_ref = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
u_hover = [0.0, 0.0, 9.8]
Q_state = [7, 7, 13.8, 1.0, 1.0, 1.0, 0.03, 0.03]
Q_hovering = [3.0, 3.0, 1.0]
Q_u = [1.0, 1.0, 1.0]

u = cs.SX.sym('u0', nu*N)
x0 = cs.SX.sym('x0', nx)

x = x0
cost = 0

# Sum all the states
for k in range(0, N*nu, nu):
        cost += stage_cost(x, u[k:k+3], Q_state, Q_hovering)
        x = lin_dynamics_dt(x, u[k:k+3], ts)

umin = [-3.13/12, -3.14/12, 5.0] * (nu*N)
umax = [3.13/12, 3.14/12, 15.0] * (nu*N)
bounds = og.constraints.Rectangle(umin, umax)

problem = og.builder.Problem(u, x0, cost).with_constraints(bounds)
build_config = og.config.BuildConfiguration()\
    .with_build_directory("python_test_build")\
    .with_build_mode("debug")\
    .with_tcp_interface_config()
meta = og.config.OptimizerMeta()\
    .with_optimizer_name("navigation")
solver_config = og.config.SolverConfiguration()\
    .with_tolerance(1e-5)
builder = og.builder.OpEnOptimizerBuilder(problem, 
                                          meta,
                                          build_config, 
                                          solver_config) \
    .with_verbosity_level(1)
builder.build()






