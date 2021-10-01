#!/usr/bin/env python3

from numpy.lib.nanfunctions import _divide_by_count
import opengen as og
import casadi.casadi as cs
import matplotlib.pyplot as plt
import numpy as np
import math


N = 40 # prediction horizon
dt = 1.0/20  # sampling time
nu = 3 # number of inputs
nu_hover = nu  # hovering condition

nx = 8 # number of states
nref = nx # reference
nQ_state = 8 # state error weight
nQ_hovering = 3 # hovering weight
nQ_u = 3   # rate weight
J = 0 # cost 

# creates a symbolic variable cs.SX.sym('x') while cs.SX.sym('x', n) creates a sym array on n elements 
u = cs.SX.sym("u", N*nu)  # decision variable u*
p = cs.SX.sym("p", nx + nref + nu_hover + nQ_state + nQ_hovering + nQ_u) # parametrized augmented state
X = p[0:nx] 
x_ref = p[nx:nx+nx] 
u_hover = p[nx+nx:nx+nx+nu] 
Q_state = p[nx+nx+nu:nx+nx+nu+nQ_state]  
Q_hovering = p[nx+nx+nu+nQ_state:nx+nx+nu+nQ_state+nQ_hovering] 
Q_u = p[nx+nx+nu+nQ_state+nQ_hovering:nx+nx+nu+nQ_state+nQ_hovering+nQ_u] 
print('X',X)
print('x_ref',x_ref)
print('u_hover',u_hover)
print('Q_state',Q_state)
print('Q_hovering',Q_hovering)
print('Q_u',Q_u)



# def lin_dynamics_ct(x, u):
    # very interesting selection how you came up with these values?
g = 9.81#9.80
Ax = 0.1#0.01
Ay = 0.1#0.01
Az = 0.2#0.01
# gravity = 9.80 ?????
tau_roll = 0.5#0.05 # Time constant
tau_pitch = 0.5#0.05 # Time constant
K_roll = 1#0.15  # Roll angle gain
K_pitch = 1#0.15 # Pitch angle gain

for i in range(N) :
    k = (i) * nu
    u0=u[k:k+nu] # desired input at the prediction step N
    # p_0=X[0:3] # position vector
    # v=X[3:6]   # velocity vector
    # roll=X[6]  # roll angle
    # pitch=X[7] # pitch angle
    # T = u0[0]   # mass normalized thrust
    # roll_g = u0[1]  # desired roll
    # pitch_g = u0[2] # desired pitch

    #dx      = x[3]
    #dy      = x[4]
    #dz      = x[5]
    ddx     = -Ax * X[3]    + g * X[7]
    ddy     = -Ay * X[4]    - g * X[6] 
    ddz     = -Az * X[5]    + u0[2] - g
    dphi    = -1/tau_roll*X[6]   + K_roll/tau_roll * u0[0]
    dtheta  = -1/tau_pitch*X[7]  + K_pitch/tau_pitch * u0[1]

#     return [dx, dy, dz, ddx, ddy, ddz, dphi, dtheta]

# def lin_dynamics_dt(x, u, Ts):
#     dx = lin_dynamics_ct(x,u)
#     return [x[i] + Ts * dx[i] for i in range(8)]
    X[0] += X[3]*dt
    X[1] += X[4]*dt
    X[2] += X[5]*dt
    X[3] += ddx*dt
    X[4] += ddy*dt
    X[5] += ddz*dt
    X[6] += dphi*dt
    X[7] += dtheta*dt


# def stage_cost(x, u, Q, R):
#     x_ref = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
#     u_ref = [0.0, 0.0, 9.8]
#     cost = Q[0]*(x[0]-x_ref)**2
#     for k in range(1,8):
#         cost += Q[k] * (x[k]-x_ref[k])**2
#     for i in range(3):
#         cost += R[i]*(u[i]-u_ref[i])**2
#     return cost

    J+= cs.sumsqr(Q_state*(X-x_ref))+cs.sumsqr(Q_hovering*(u0-u_hover))
    if i < N - 1:
        J+=cs.sumsqr(Q_u*(u0-u[(i+1)*nu:(i+1)*nu+nu]))
# # System model
# # ------------------------------------



# # Build parametric optimizer
# # ------------------------------------
# RATE = 20
# (nu, nx, N) = (3, 8, 20)
# ts = 1/RATE

# x_ref = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
# u_hover = [0.0, 0.0, 9.8]
# Q_state = [7, 7, 13.8, 1.0, 1.0, 1.0, 0.03, 0.03]
# Q_hovering = [3.0, 3.0, 1.0]
# Q_u = [1.0, 1.0, 1.0]

# u = cs.SX.sym('u0', nu*N)
# x0 = cs.SX.sym('x0', nx)

# x = x0
# cost = 0

# # Sum all the states
# for k in range(0, N*nu, nu):
#         cost += stage_cost(x, u[k:k+3], Q_state, Q_hovering)
#         x = lin_dynamics_dt(x, u[k:k+3], ts)

# umin = [-3.13/12, -3.14/12, 5.0] * (nu*N)
# umax = [3.13/12, 3.14/12, 15.0] * (nu*N)
u_min = [-math.pi/12,-math.pi/12,0]
u_max = [math.pi/12,math.pi/12,14.961+9.81]
umin=np.kron(np.ones((1,N)),u_min)
umin=umin.tolist()
umin=umin[0]
umax=np.kron(np.ones((1,N)),u_max)
umax=umax.tolist()
umax=umax[0]

bounds= og.constraints.Rectangle(umin,umax)
problem = og.builder.Problem(u, p, J) \
.with_constraints(bounds)

tcp_config = og.config.TcpServerConfiguration(bind_port=3303)        
          
meta = og.config.OptimizerMeta() \
    .with_version("0.0.0") \
    .with_authors(["A. Papadimitriou"]) \
    .with_licence("CC4.0-By") \
    .with_optimizer_name("optimizer01") 
build_config = og.config.BuildConfiguration() \
    .with_build_directory("python_build") \
    .with_build_mode("release") \
    .with_tcp_interface_config(tcp_config)\
    .with_build_c_bindings()  \
    .with_rebuild(True)
solver_config = og.config.SolverConfiguration() \
    .with_tolerance(1e-5) \
    .with_max_inner_iterations(155) \
    .with_penalty_weight_update_factor(5)\
    .with_max_outer_iterations(4)
    #.with_lfbgs_memory(15) \
builder = og.builder.OpEnOptimizerBuilder(problem,
    metadata=meta,
    build_configuration=build_config,
    solver_configuration=solver_config).with_verbosity_level(1)
#builder.enable_tcp_interface()
builder.build()


#mng = og.tcp.OptimizerTcpManager('python_build/optimizer01')
#mng.start()
#
#pong = mng.ping() # check if the server is alive
#print(pong)
#x_init= [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
#x_ref = [0.0,0.0,3.0,0.0,0.0,0.0,0.0,0.0]
### weights and hovering reference
#u_hover=[0, 0, 9.81]
#Q_state=[5.0, 5.0, 20.0, 5.0, 5.0, 5.0, 1.0, 1.0]
#Q_hovering=[10.0,10.0,10.0]
#Q_u=[20,20,20]
#
#
#x_0=x_init+ x_ref +u_hover+ Q_state+Q_hovering+Q_u#concatenate lists in Python
#solution = mng.call(x_0) # call the solver over TCP
#print(solution)
#print('u_0:')
#u0=solution[u'solution']
#print(u0[0:3])
##print('max_constraint_violation',solution[u'max_constraint_violation'])
#print('num_outer_iterations' ,solution[u'num_outer_iterations'])
#mng.kill()
