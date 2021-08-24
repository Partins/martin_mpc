#!/usr/bin/env python3

import rospy
import actionlib
import numpy as np

from martin_mpc.msg import *




class GENERATOR:
    
    _feedback   = martin_mpc.msg.generate_trajectoryFeedback()
    _result     = martin_mpc.msg.generate_trajectoryResult()

    def __init__(self):

        # Initial position and velocity. Updated after every trajectory 
        # generation where the last point of the previous trajectory is the 
        # starting point of the next trajectory. The assumption is that the 
        # UAV actually reaches the last point. This enables us to generate a 
        # new trajectory even before the UAV hasn't actually finished the 
        # previous one. 

        self.traj_pos0 = [0, 0, 0]
        self.traj_vel0 = [0, 0, 0]

        self._action_name = 'traj_gen'
        self.action_server =  actionlib.SimpleActionServer(self._action_name, \
                    martin_mpc.msg.generate_trajectoryAction, \
                    execute_cb=self.generate_trajectory, auto_start = False)
        self.action_server.start()
    # Function: generate_trajectory(self, goal_pos, goal_vel, rate, T)
    # A function that generates a trajectory (points) with a reference position
    # and velocity. The starting point is the last point of the previous 
    # trajectory. The amount of points is calculated based on the time for the
    # trajectory and the rate. The longer the time the more points will be
    # generated. 
    def generate_trajectory(self, goal):

        rospy.logwarn("Generating trajectory")
        T = goal.time
        n = T * goal.rate # Amount of points

        # From CORKE:  
        # AMAT*x = BMAT
        AMAT = np.array([[0,0,0,0,0,1], \
                         [T**5,T**4,T**3,T**2,T,1], \
                         [0,0,0,0,1,0], \
                         [5*T**4,4*T**3,3*T**2,2*T,1,0], \
                         [0,0,0,2,0,0], \
                         [20*T**3,12*T**2,6*T,2,0,0]])

        BMAT = np.array([ self.traj_pos0, goal.goal_pos, \
                          self.traj_vel0, [0,0,0], \
                          [0,0,0], [0,0,0] ])

        x = np.transpose(np.linalg.inv(AMAT) @ BMAT)
        traj_pos = np.array([self.traj_pos0]) # Array containing positions
        traj_vel = np.array([self.traj_vel0]) # Array containing velocities

        n = int(n)
        # Position
        for t in range(1,n+1):
            t = (t)*T/n
            t_mat = np.array([ [t**5], [t**4], [t**3], [t**2], [t], [1] ])
            traj_pos = np.append(traj_pos, np.transpose(x @ t_mat), axis=0)
        
        # Velocity    
        for t in range(1,n+1):
            t = (t)*T/n
            t_mat = np.array([ [5*t**4], [4*t**3], [3*t**2], [2*t], [1], [0] ])
            traj_vel = np.append(traj_vel, np.transpose(x @ t_mat), axis=0)

        self._result.trajectory_id = 1
        self._result.resulting_trajectory = [0.0, 0.0]
        
        self.action_server.set_succeeded(self._result)
        rospy.logwarn(type(t_mat))
        #return traj_pos, traj_vel
        

if __name__ == '__main__':
    try:
        rospy.init_node('trajectory_generator')
        server = GENERATOR()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass