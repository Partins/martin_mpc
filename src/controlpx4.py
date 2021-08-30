#!/usr/bin/env python3

import rospy
import time
import math
import numpy as np
from scipy.signal import cont2discrete as c2d
import cvxpy as cp
import matplotlib.pyplot as plt 


from rosflight_msgs.msg import Command, RCRaw, Status
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path, Odometry
from rospy.impl.transport import INBOUND
from std_msgs.msg import Float64, String, Int64
from gazebo_msgs.msg import ModelStates, ModelState
from gazebo_msgs.srv import GetModelState, SetModelState
from apriltag_ros.msg import AprilTagDetectionArray
from rosflight_extras.srv import arm_uav
from martin_mpc.srv import mpcsrv, landsrv, gotosrv
from mavros_msgs.msg import AttitudeTarget

#Params
import dynamic_reconfigure.client

class UAV:

    def dyn_callback(self, config):
        self.tag_follow = config.tag_follow
        self.tot_error[0] = 0
        self.tot_error[1] = 0

    def __init__(self):

        rospy.init_node('UAV_control', anonymous=True)
        self.x_states_topic   = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        ## Subscribers
        self.sub_status         = rospy.Subscriber('status', Status, self.get_status)
        self.tag_pos            = rospy.Subscriber('tag_detections', AprilTagDetectionArray, self.get_tag)
        self.control_command    = rospy.Subscriber('control_command', String, self.user_command)
        self.state_update       = rospy.Subscriber('multirotor/truth/NED', Odometry, self.get_state)

        ## Publishers
        self.pub_command     = rospy.Publisher('mavros/setpoint_raw/attitude', AttitudeTarget, queue_size=1)
        #self.pub_raw        = rospy.Publisher('multirotor/RC', RCRaw, queue_size=1)
        #self.pub_plotter    = rospy.Publisher('plotter', float_array, queue_size=1)
        #self.path_pub       = rospy.Publisher('path', Path, queue_size=1)
        #self.pub_state      = rospy.Publisher('uav_state', Odometry, queue_size=1)
        
        
        ## Services
        self.mpc_calc           = rospy.ServiceProxy('MPC_calc', mpcsrv)


        self.land_srv           = rospy.Service('Land_UAV', landsrv, self.landsrv)
        self.takeoff_srv        = rospy.Service('Takeoff_UAV', landsrv, self.takeoffsrv)
        self.goto_srv           = rospy.Service('GOTO', gotosrv, self.gotosrv)
        
        ## Messages
        self.msg        = AttitudeTarget()
        #self.plot_msg   = float_array() # Cutom float array msg type

        ## ROS
        self.RATE = 20 # [Hz]
        self.rate = rospy.Rate(self.RATE)
        self.is_armed = False

        ## Inits
        self.tot_error  = [0, 0, 0]    
        self.prev_error = [0, 0, 0]     
        self.tag_x      = 0
        self.tag_y      = 0
        self.tag_z      = 0
        self.pos        = [0, 0, 0]
        self.yaw        = 0
        self.land       = 0
        self.x_states   = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        

        #state = ModelState()
        #state.model_name = 'multirotor'
        #state.reference_frame = 'ground_plane'
        #state.pose.position.x = 0.0
        #state.pose.position.y = 0.0
        #state.pose.position.z = 0.05
#
        ## yaw 90 deg: z=w=0.7068
        ## yaw 180deg: z=1
        #state.pose.orientation.x    = 0
        #state.pose.orientation.y    = 0
        #state.pose.orientation.z    = 0
        #state.pose.orientation.w    = 0
        #state.twist.linear.x        = 0
        #state.twist.linear.y        = 0
        #state.twist.linear.z        = 0
        #state.twist.angular.x       = 0
        #state.twist.angular.y       = 0
        #state.twist.angular.z       = 0

        # Dynamic reconfiguration
        self.dyn_client = dynamic_reconfigure.client.Client("martin_mpc_param_node", timeout=30, config_callback=self.dyn_callback)
        
        

    def run(self):

        # Main loop
        self.xr = [0.0 ,0.0, 2.0, 0.0 ,0.0 ,0.0 ,0.0, 0.0]  # Reference signal

        i = 0
        I = 0.02
        antiwind = 1

        # Generate a trajectory                                   
        self.traj_pos, self.traj_vel = self.generate_trajectory([0,0,0],[0,0,0],[0,0,0], [0,0,0], self.RATE,1)
        rospy.logwarn("Trajectory 1 generated")
        # Save last point to use for next trajectory generation
        
        self.msg.body_rate.z = 0
        n_loops = 0 # Counter of loops
        self.traj_index = 0
        #I = 0.1
        #P = 1
        #D = 0.01
####################################### main thing ############################################33
        self.tag_follow = False
        cntr = 0
        while not rospy.is_shutdown():
            #rospy.logwarn("RATE" + str(self.RATE))
            cntr += 1
            n_loops += 1
            self.traj_index += 1
            tmp_eul = self.get_current_state(self.tag_follow) # Controls absolute/relative positioning
            #rospy.logwarn([self.tag_y, self.tag_x])
            #rospy.logwarn(self.tag_follow)
            if self.traj_index >= len(self.traj_pos):
                self.traj_index -= 1
            # Setting the next point in the trajectory as reference 
            self.xr[0] = self.traj_pos[self.traj_index,0]  #
            self.xr[1] = self.traj_pos[self.traj_index,1]  # Position
            self.xr[2] = self.traj_pos[self.traj_index,2]  #
    
            self.xr[3] = self.traj_vel[self.traj_index,0]  #
            self.xr[4] = self.traj_vel[self.traj_index,1]  # Velocity
            self.xr[5] = self.traj_vel[self.traj_index,2]  #

                        # Antiwind of the error to the fiducial (maybe not needed?)
            #if abs(self.tot_error[0]) <= antiwind:
            #    self.tot_error[0] += -self.x_states[0]*I #+ self.x_states[0]*D
###
            #if abs(self.tot_error[1]) <= antiwind:
            #    self.tot_error[1] += self.x_states[1]*I #+ self.x_states[1]*D

            # Optimization (mpc.py)
            #tic = rospy.Time.now()
            resp1 = self.mpc_calc(self.xr, [0.0, 0.0, 9.8], self.x_states, self.tot_error)
            #toc = rospy.Time.now()
            #rospy.logwarn("Time to solve MPC:" + str(toc.nsecs-tic.nsecs))

            # Plotting
            #self.plot_msg.data = [self.xr[0], self.xr[1]]
            #self.plot_msg.header.stamp = rospy.Time.now()
            #self.pub_plotter.publish(self.plot_msg)

            ## UAV Command message
            self.msg.header.stamp = rospy.Time.now()
            #self.msg.mode = Command.MODE_ROLL_PITCH_YAWRATE_THROTTLE
            
            self.msg.body_rate.x =  math.cos(-tmp_eul[2]) * resp1.control_signals[0] + math.sin(-tmp_eul[2]) * resp1.control_signals[1]
            self.msg.body_rate.y = -math.sin(-tmp_eul[2]) * resp1.control_signals[0] + math.cos(-tmp_eul[2]) * resp1.control_signals[1]
            self.msg.body_rate.z = 0
            self.msg.thrust = resp1.control_signals[2]/9.8
            self.pub_command.publish(self.msg)
          
            #rospy.logwarn(cntr)
            self.rate.sleep()
            

########################################### MAIN END ###################################################################
    
    
    def landsrv(self, req):
        T = 60 # Time to land

        rospy.logwarn("Landing Service Initiated")
        
        tic = rospy.Time.now()
        self.traj_pos, self.traj_vel = self.generate_trajectory( \
                [self.x_states[0], self.x_states[1], self.x_states[2]], \
                [self.x_states[3], self.x_states[4], self.x_states[5]], \
                [self.x_states[0], self.x_states[1], 0], [0,0,0], self.RATE,T)
        
        toc = rospy.Time.now()
        rospy.logwarn("Landing trajectory generated with: " + str(self.RATE * T) + " points")
        self.traj_index = 0

        return True

    #def abort(self, req):
    #    T = 5 # Time to takeoff
    #    while rospy.
    #    rospy.logwarn("ABORTING")
    #    self.msg.header.stamp = rospy.Time.now()
    #    self.msg.mode = Command.MODE_ROLL_PITCH_YAWRATE_THROTTLE
    #        
    #    self.msg.x = 0
    #    self.msg.y = 0
    #    self.msg.F = 0
    #    self.pub_command.publish(self.msg)
#
    #    return True

    def takeoffsrv(self, req):
        T = 5 # Time to takeoff

        rospy.logwarn("Landing Service Initiated")
        
        tic = rospy.Time.now()
        self.traj_pos, self.traj_vel = self.generate_trajectory( \
                [self.x_states[0], self.x_states[1], self.x_states[2]], \
                [self.x_states[3], self.x_states[4], self.x_states[5]], \
                [0,0,2], [0,0,0], self.RATE,T)
        toc = rospy.Time.now()
        rospy.logwarn("Landing trajectory generated in "+ str(toc.nsecs-tic.nsecs) + "ns")
        self.traj_index = 0
        return True

    def gotosrv(self, msg):
        self.traj_pos, self.traj_vel = self.generate_trajectory( \
                [self.x_states[0], self.x_states[1], self.x_states[2]], \
                [self.x_states[3], self.x_states[4], self.x_states[5]], \
                [msg.x,msg.y,msg.z], [0,0,0], self.RATE, int(msg.T))
        self.traj_index = 0
        return True

    ## Get current state
    #   This function gets the state of the UAV without race conditions. 
    #   If the tag_tracking is set to True the position will be relative to the
    #   QR-code while setting it to False it gives the absolute position in 
    #   the world frame. 

    def get_current_state(self, tag_tracking=True):
        #self.states = self.get_model('multirotor', 'ground_plane')
        self.states = self.x_states_topic
        # Relative vs. absolute positioning
        if tag_tracking:
            self.x_states[0] = self.tag_y# - 0.05
            self.x_states[1] = self.tag_x# - 0.05
        else:
            self.x_states[0] = self.states[0]
            self.x_states[1] = self.states[1]
        self.x_states[2] = self.states[2]
        
        # Linear velocities
        self.x_states[3] = self.states[3]
        self.x_states[4] = self.states[4]
        self.x_states[5] = self.states[5]
       
        self.x_states[6] = self.states[6]   # Roll
        self.x_states[7] = self.states[7]   # Pitch

        return [0,0,self.yaw]
    def user_command(self, msg):
        if msg.data == "land":
            self.land = 1
        else:
            self.land = 0

    def quaternion_to_euler_angle_vectorized1(self, w, x, y, z):

        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = math.atan2(t0, t1)
     
        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = math.asin(t2)
     
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)

        return roll_x, pitch_y, yaw_z 

    ## Get State
    # Subscribes to the Odometry topic and converts to world frame. 

    def get_state(self, msg):
        
        self.x_states_topic[0] = msg.pose.pose.position.x
        self.x_states_topic[1] = -msg.pose.pose.position.y
        self.x_states_topic[2] = -msg.pose.pose.position.z
        
        # Linear velocities
        self.x_states_topic[3] = msg.twist.twist.linear.x
        self.x_states_topic[4] = -msg.twist.twist.linear.y
        self.x_states_topic[5] = -msg.twist.twist.linear.z

        tmp_eul = self.quaternion_to_euler_angle_vectorized1( \
                                            msg.pose.pose.orientation.w, \
                                            msg.pose.pose.orientation.x, \
                                            msg.pose.pose.orientation.y, \
                                            msg.pose.pose.orientation.z)
                                                        
        self.x_states_topic[6] = tmp_eul[0]   # Roll
        self.x_states_topic[7] = tmp_eul[1]   # Pitch
        self.yaw = tmp_eul[2]

        #odom_msg = Odometry()
        #odom_msg.header.stamp = rospy.Time.now()
#
        #odom_msg.pose.pose.position.x = self.x_states_topic[0]
        #odom_msg.pose.pose.position.y = self.x_states_topic[1]
        #odom_msg.pose.pose.position.z = self.x_states_topic[2]
#
        #odom_msg.twist.twist.linear.x = self.x_states_topic[3]
        #odom_msg.twist.twist.linear.y = self.x_states_topic[4]
        #odom_msg.twist.twist.linear.z = self.x_states_topic[5]
#
        #odom_msg.pose.pose.orientation.x = msg.pose.pose.orientation.x
        #odom_msg.pose.pose.orientation.y = msg.pose.pose.orientation.y
        #odom_msg.pose.pose.orientation.z = msg.pose.pose.orientation.z
        #odom_msg.pose.pose.orientation.w = msg.pose.pose.orientation.w
#
        #odom_msg.twist.twist.angular.x = msg.twist.twist.angular.x
        #odom_msg.twist.twist.angular.y = msg.twist.twist.angular.y
        #odom_msg.twist.twist.angular.z = msg.twist.twist.angular.z
        #self.pub_state.publish(odom_msg)



    def get_tag(self, msg):

        try:
            tag_pos = np.array([[msg.detections[0].pose.pose.pose.position.x],\
                                [msg.detections[0].pose.pose.pose.position.y],\
                                [msg.detections[0].pose.pose.pose.position.z]])

            #tag_pos = self.rotz(-3.14/2, tag_pos)   # Rotate around Z
            #tag_pos = self.roty(3.14, tag_pos)      # Rotate around y
            
            self.tag_x = tag_pos[0,0]
            self.tag_y = tag_pos[1,0]

        except:
            pass
    


    def rotx(self, angle, point):
        c = np.cos(angle)
        s = np.sin(angle)
        rotx = np.matrix([[1, 0, 0], \
                         [0, c, -s], \
                         [0, -s, c]])
        return np.matmul(rotx,point)

    def roty(self, angle, point):
        c = np.cos(angle)
        s = np.sin(angle)
        roty = np.matrix([[c, 0, s], \
                          [0, 1, 0], \
                          [-s, 0, c]])
        return np.matmul(roty,point)

    def rotz(self, angle, point):
        c = np.cos(angle)
        s = np.sin(angle)
        rotz = np.matrix([[c, -s, 0], \
                          [s, c, 0], \
                          [0, 0, 1]])
        return np.matmul(rotz,point)

    def get_status(self, msg):
        self.is_armed = msg.armed
        self.is_rc_override = msg.rc_override

    def generate_trajectory(self, start_pos, start_vel, goal_pos, goal_vel, rate, T):
        n = T * rate # Amount of points

        # From CORKE:  
        # AMAT*x = BMAT
        #rospy.logwarn(n)
        AMAT = np.array([[0,0,0,0,0,1], \
                         [T**5,T**4,T**3,T**2,T,1], \
                         [0,0,0,0,1,0], \
                         [5*T**4,4*T**3,3*T**2,2*T,1,0], \
                         [0,0,0,2,0,0], \
                         [20*T**3,12*T**2,6*T,2,0,0]])

        BMAT = np.array([ start_pos, goal_pos, \
                          start_vel, [0,0,0], \
                          [0,0,0], [0,0,0] ])

        x = np.transpose(np.linalg.inv(AMAT) @ BMAT)
        test_pos = np.array([start_pos])
        test_vel = np.array([start_vel])

        
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
       
       # For publishing the path as points for RVIZ. NOT WORKING RN
       # path_msg = Path()
       # pose = PoseStamped()
       # rospy.logwarn(test_pos[0,0])
       # pos = 0

       # for pos in range(0,10):
       #     pose.header.stamp = rospy.Time.now()
       #     pose.header.frame_id = "/my_cs"
       #     pose.header.seq = pos
       #     pose.pose.position.x = pos
       #     pose.pose.position.y = pos + 1
       #     pose.pose.position.z = pos + 2
       #     path_msg.poses.append(pose)
       #     path_msg.header.frame_id = "/my_cs"
       #     path_msg.header.stamp = rospy.Time.now()
       #     
       #     self.path_pub.publish(path_msg)

        return test_pos, test_vel

    # Subscriber callback. But I'm wondering about race conditions. Maybe good
    # to use ServiceProxy to get the states directly in the controller when 
    # needed. Will keep this here for now
    #def state_update(self, msg):
    #    
    #    self.x_states[0] = msg.pose[1].position.x
    #    self.x_states[1] = msg.pose[1].position.y
    #    self.x_states[2] = msg.pose[1].position.z
    #    self.x_states[3] = msg.twist[1].linear.x
    #    self.x_states[4] = msg.twist[1].linear.y
    #    self.x_states[5] = msg.twist[1].linear.z
#
    #    tmp_eul = self.quaternion_to_euler_angle_vectorized1( \
    #                                        msg.pose[1].orientation.w, \
    #                                        msg.pose[1].orientation.x, \
    #                                        msg.pose[1].orientation.y, \
    #                                        msg.pose[1].orientation.z)
    #    self.x_states[6] = tmp_eul[0]   # Roll
    #    self.x_states[7] = tmp_eul[1]   # Pitch                
    
if __name__ == '__main__':
    try:
        multirotor = UAV()
        multirotor.run()
    except rospy.ROSInterruptException:
        pass
        







