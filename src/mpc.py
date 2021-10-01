#!/usr/bin/env python3

import rospy
import time
import math
import numpy as np
from scipy.signal import cont2discrete as c2d
from scipy.linalg import expm
import cvxpy as cp
import matplotlib.pyplot as plt 
from std_msgs.msg import Float32MultiArray, Int16
from rosflight_extras.msg import float_array

## Services
from gazebo_msgs.srv import GetModelState
from martin_mpc.srv import mpcsrv, mpcsrvResponse

#Params
import dynamic_reconfigure.client


class UAV_model:

    def __init__(self):
        self.time_pub = rospy.Publisher('mpc_time', Int16, queue_size=1)
        self.pub_uav_state_plotter        = rospy.Publisher('uav_state_plotter', float_array, queue_size=1)
        self.pub_uav_input_plotter        = rospy.Publisher('uav_input_plotter', float_array, queue_size=1)
        self.pub_uav_control_plotter      = rospy.Publisher('uav_control_plotter', float_array, queue_size=1)

        ## observer things
        self.prev_inputs = [0, 0, 0] # u[k-1]
        rospy.init_node('mpc', anonymous=True)
        self.mpc_srv = rospy.Service('MPC_calc', mpcsrv, self.mpc_calc)
        self.Ax = 0.01
        self.Ay = 0.01
        self.Az = 0.01
        self.gravity = 9.80
        self.tau_roll = 0.05 # Time constant
        self.tau_pitch = 0.05 # Time constant
        self.K_roll = 0.15  # Roll angle gain
        self.K_pitch = 0.15 # Pitch angle gain
        RATE = 20
        self.Q = np.diag([7, 7, 13.8, 1.0, 1.0, 1.0, 0.03, 0.03])
        #self.Q = np.diag([10, 10, 15.0, 3.0, 3.0, 3.0, 1.0, 1.0]) # Andreas
        self.R = 1*np.diag([3, 3, 1.0])
        self.Rd = 1*np.diag([3, 3, 1.0])
        #self.R = 1*np.diag([20.0, 20.0, 20.0]) # Andreas
        self.K = np.diag([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        self.d = np.zeros(shape=(3,1), dtype=float)
        self.obs_states = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] 

        # Max/min values
        self.umin = np.array([-3.14/1, -3.14/1, 4.0])
        self.umax = np.array([3.14/1, 3.14/1, 15])

        self.xmin = np.array([-np.inf, -np.inf, 0, -np.inf, -np.inf, -np.inf,
                 -3.14/12,-3.14/12])
        self.xmax = np.array([np.inf, np.inf, np.inf, np.inf, np.inf, np.inf,
                  3.14/12, 3.14/12])

        self.Ac = np.zeros(shape=(8, 8), dtype=float)
        self.Bc = np.zeros(shape=(8, 3), dtype=float)
        self.Bdc = np.zeros(shape=(8, 3), dtype=float)

        # Martin
        self.Ac[0,3] = 1
        self.Ac[1,4] = 1
        self.Ac[2,5] = 1
        self.Ac[3,3] = -self.Ax
        self.Ac[3,7] = -self.gravity
        self.Ac[4,4] = -self.Ay
        self.Ac[4,6] = -self.gravity
        self.Ac[5,5] = -self.Az
        self.Ac[6,6] = -1.0 / self.tau_roll
        self.Ac[7,7] = -1.0 / self.tau_pitch

        # Input transfer matrix
        self.Bc[5,2] = 1
        self.Bc[6,0] = self.K_roll / self.tau_roll
        self.Bc[7,1] = self.K_pitch / self.tau_pitch

        # Disturbance model
        self.Bdc[3,0] = 1.0
        self.Bdc[4,1] = 1.0
        self.Bdc[5,2] = 1.0

        self.Cc = np.array([1, 1, 1, 1, 1, 1, 1, 1])
        self.Dc = 0

        # Discretization
        self.Ad, self.Bd, self.Cd, self.Dd, dt = c2d((self.Ac, self.Bc, \
                                                     self.Cc, self.Dc), 1/RATE)

        # To discretize Bd
        tempA, self.Bdd, tempC, tempD, dt = c2d((self.Ac, self.Bdc, \
                                                     self.Cc, self.Dc), 1/RATE)
        rospy.logwarn(np.shape(np.zeros(shape=(8, 3))))
        tempmat1 = np.concatenate([self.Ad-np.eye(8), self.Bd], axis=1)
        #tempmat2 = np.concatenate([np.eye(8), np.zeros(shape=(8, 3))], axis=0)
        tempmat2 = np.concatenate( [np.eye(8), np.zeros([8,3])] , axis=1)
        
        LH = np.concatenate([tempmat1,tempmat2])

        self.LH_pinv = np.linalg.inv( np.transpose(LH)@LH ) @ np.transpose(LH)

        #rospy.logwarn(np.shape(self.LH_pinv))
        # Dynamic reconfiguration
        self.dyn_client = dynamic_reconfigure.client.Client("martin_mpc_param_node", timeout=30, config_callback=self.dyn_callback)
        
        rospy.logwarn("MPC READY")
        rospy.spin()
    
    def dyn_callback(self, config):
        self.Q  = np.diag([config.Q0, config.Q1, config.Q2, config.Q3, \
                           config.Q4, config.Q5, config.Q6, config.Q7])
        
        self.R  = np.diag([config.R1, config.R2, config.R3])

        self.Rd = np.diag([config.Rd1, config.Rd2, config.Rd3])

        self.K  = np.diag([config.K0, config.K1, config.K2, config.K3, \
                            config.K4, config.K5, config.K6, config.K7])
    
    def mpc_calc(self, req):

        
        #reference = np.array(req.reference) - (np.array(req.states) - np.array(self.obs_states))
        N = 20               # Prediction horizon 
        u = cp.Variable((3,N))  # Three inputs roll desired, pitch desired, thrust
        x = cp.Variable((8,N+1))# Eight states

        #print(req.reference)
        cost = 0 # Initialization
        #print(req.states)
        req_ref = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0] 
        req_input = [0.0, 0.0, 9.8]
        constraints = [x[:,0] == req.states]

        for k in range(0,N):                                                                                                        # Input rate cost
            cost += cp.quad_form(x[:,k]-req_ref, self.Q) + cp.quad_form(u[:,k]-req_input, self.R)#+ cp.quad_form(u[:,k] - self.prev_inputs, self.Rd)
            constraints += [x[:,k+1] == self.Ad@x[:,k] + self.Bd@u[:,k]] #  - self.Bdd@req.disturbance
            constraints += [self.umin <= u[:,k], u[:,k] <= self.umax]
    
        #for k in range(0,N):                                                                                                        # Input rate cost
        #    cost += cp.quad_form(req.reference - x[:,k], self.Q) + cp.quad_form(req.input_ref - u[:,k], self.R)#+ cp.quad_form(u[:,k] - self.prev_inputs, self.Rd)
        #    constraints += [x[:,k+1] == self.Ad@x[:,k] + self.Bd@u[:,k]] #  - self.Bdd@req.disturbance
        #    constraints += [self.umin <= u[:,k], u[:,k] <= self.umax]    
        prob = cp.Problem(cp.Minimize(cost), constraints)
        prob.solve(verbose=False)
        
        uav_plot_msg = float_array()
        print(x.value[2,:])
        #for i in range(len(x.value[2,:])):
        uav_plot_msg.header.stamp = rospy.Time.now()
        uav_plot_msg.data = x.value[2,:]
        self.pub_uav_state_plotter.publish(uav_plot_msg)
        
        uav_plot_msg.data = req.states
        self.pub_uav_input_plotter.publish(uav_plot_msg)
        uav_plot_msg.data = u.value[2,:]
        self.pub_uav_control_plotter.publish(uav_plot_msg) 

        self.prev_inputs = u.value[:,1]
        #self.obs_states = self.observer(u.value[:,0], req.states)

        return mpcsrvResponse(u.value[:,0])

    def observer(self, inputs, states):
        
        x_states = np.matmul(self.Ad, states) + np.matmul(self.Bd, inputs)
        
        return x_states

if __name__ == '__main__':
    multirotor = UAV_model()

