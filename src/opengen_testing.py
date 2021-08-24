#!/usr/bin/env python3
import casadi.casadi as cs
import opengen as og

u = cs.SX.sym("u", 5)
p = cs.SX.sym("p", 2)
phi = og.functions.rosenbrock(u,p)

ball = og.constraints.Ball2(None, 1.5)
rect = og.constraints.Rectangle(xmin=[-1, -2, -3], xmax=[0, 10, -1])

segment_ids = [1, 4]
bounds = og.constraints.CartesianProduct(segment_ids, [ball, rect])

problem = og.builder.Problem(u, p, phi).with_constraints(bounds)

print(problem)