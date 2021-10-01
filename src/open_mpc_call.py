from numpy.lib.nanfunctions import _divide_by_count
import opengen as og
import casadi.casadi as cs
import matplotlib.pyplot as plt
import numpy as np

mng = og.tcp.OptimizerTcpManager('python_build/optimizer01')
mng.start()

pong = mng.ping() # check if the server is alive
print(pong)
x_init= [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
x_ref = [0.0,0.0,3.0,0.0,0.0,0.0,0.0,0.0]
## weights and hovering reference
u_hover=[0, 0, 9.81]
Q_state=[5.0, 5.0, 20.0, 5.0, 5.0, 5.0, 1.0, 1.0]
Q_hovering=[10.0,10.0,10.0]
Q_u=[20,20,20]


x_0=x_init+ x_ref +u_hover+ Q_state+Q_hovering+Q_u#concatenate lists in Python
solution = mng.call(x_0) # call the solver over TCP
print(solution)
print('u_0:')
u0=solution[u'solution']
print(u0[0:3])
#print('max_constraint_violation',solution[u'max_constraint_violation'])
print('num_outer_iterations' ,solution[u'num_outer_iterations'])
mng.kill()















