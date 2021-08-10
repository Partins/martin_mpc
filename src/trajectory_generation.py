#!/usr/bin/env python3

import rospy


class GENERATOR:
    
    def __init__(self):

        # Initial position and velocity. Updated after every trajectory 
        # generation where the last point of the previous trajectory is the 
        # starting point of the next trajectory. The assumption is that the 
        # UAV actually reaches the last point. This enables us to generate a 
        # new trajectory even before the UAV hasn't actually finished the 
        # previous one. 

        self.traj_pos_0 = [0, 0, 0]
        self.traj_vel_0 = [0, 0, 0]

    def generate_trajectory(self, goal_pos, goal_vel, rate, T):

        n = T * rate # Amount of points
        # From CORKE:  
        # AMAT*x = BMAT
        AMAT = np.array([[0,0,0,0,0,1], \
                         [T**5,T**4,T**3,T**2,T,1], \
                         [0,0,0,0,1,0], \
                         [5*T**4,4*T**3,3*T**2,2*T,1,0], \
                         [0,0,0,2,0,0], \
                         [20*T**3,12*T**2,6*T,2,0,0]])

        BMAT = np.array([ self.traj_pos0, goal_pos, \
                          self.traj_vel0, [0,0,0], \
                          [0,0,0], [0,0,0] ])

        x = np.transpose(np.linalg.inv(AMAT) @ BMAT)
        test_pos = np.array([self.traj_pos0])
        test_vel = np.array([self.traj_vel0])

        
        # Position
        for t in range(1,n+1):
            t = (t)*T/n
            t_mat = np.array([ [t**5], [t**4], [t**3], [t**2], [t], [1] ])
            test_pos = np.append(test_pos, np.transpose(x @ t_mat), axis=0)
        
        # Velocity    
        for t in range(1,n+1):
            t = (t)*T/n
            t_mat = np.array([ [5*t**4], [4*t**3], [3*t**2], [2*t], [1], [0] ])
            test_vel = np.append(test_vel, np.transpose(x @ t_mat), axis=0)

        return test_pos, test_vel


if __name__ == '__main__':
    try:
        trajectory_generator = GENERATOR()
        trajectory_generator.run()
    except rospy.ROSInterruptException:
        pass