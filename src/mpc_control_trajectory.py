#!/usr/bin/env python3

## MPC controller with trajectory generation

import rospy
import time
import math
import numpy as np
from scipy.signal import cont2discrete as c2d
import cvxpy as cp
import matplotlib.pyplot as plt 


from rosflight_msgs.msg import Command, RCRaw, Status
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from rospy.impl.transport import INBOUND
from std_msgs.msg import Float64, String, Int64
from gazebo_msgs.msg import ModelStates, ModelState
from gazebo_msgs.srv import GetModelState, SetModelState
from apriltag_ros.msg import AprilTagDetectionArray
from rosflight_extras.srv import arm_uav
from martin_mpc.srv import mpcsrv
from rosflight_extras.msg import *


# PWM values if it was a real controller
STICK_LOW   = 1000
STICK_MID   = 1500
STICK_HIGH  = 2000
SWITCH_LOW  = 1000
SWITCH_HIGH = 2000

ROLL_CHN    = 0
PITCH_CHN   = 1
THR_CHN     = 2
YAW_CHN     = 3
AUX1        = 4 # RC_OVR
AUX2        = 5
AUX3        = 6
AUX4        = 7 # ARM

class UAV:

    def __init__(self):

        rospy.init_node('UAV_control', anonymous=True)

        ## Subscribers
        self.sub_status         = rospy.Subscriber('status', Status, self.get_status)
        self.tag_pos            = rospy.Subscriber('tag_detections', AprilTagDetectionArray, self.get_tag)
        self.control_command    = rospy.Subscriber('control_command', String, self.user_command)

        ## Publishers
        self.pub_command    = rospy.Publisher('command', Command, queue_size=1)
        self.pub_raw        = rospy.Publisher('multirotor/RC', RCRaw, queue_size=1)
        self.pub_plotter    = rospy.Publisher('plotter', float_array, queue_size=1)
        self.path_pub       = rospy.Publisher('path', Path, queue_size=1)
        
        ## Services
        self.set_state_service  = rospy.ServiceProxy('gazebo/set_model_state', SetModelState)
        self.get_model          = rospy.ServiceProxy('gazebo/get_model_state', GetModelState)
        self.arm_srv            = rospy.ServiceProxy('arm_UAV', arm_uav)
        self.pc_control         = rospy.ServiceProxy('pc_control', arm_uav)
        self.mpc_calc           = rospy.ServiceProxy('MPC_calc', mpcsrv)
        
        ## Messages
        self.msg_raw    = RCRaw()       # 1st Config in arm()
        self.msg        = Command()     # 1st Config in enable_computer_control()
        self.plot_msg   = float_array() # Cutom float array msg type

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
        self.traj_pos0  = [0, 0, 0]
        self.traj_vel0  = [0, 0, 0]
        self.x_states   = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        
        state = ModelState()
        state.model_name = 'multirotor'
        state.reference_frame = 'ground_plane'
        state.pose.position.x = 0.0
        state.pose.position.y = 0.0
        state.pose.position.z = 0.05

        # yaw 90 deg: z=w=0.7068
        # yaw 180deg: z=1
        state.pose.orientation.x    = 0
        state.pose.orientation.y    = 0
        state.pose.orientation.z    = 0
        state.pose.orientation.w    = 0
        state.twist.linear.x        = 0
        state.twist.linear.y        = 0
        state.twist.linear.z        = 0
        state.twist.angular.x       = 0
        state.twist.angular.y       = 0
        state.twist.angular.z       = 0

        rospy.wait_for_service('/gazebo/set_model_state')
        try:
            set_state = self.set_state_service
            result = set_state(state)
            assert result.success is True
        except rospy.ServiceException:
            print("/gazebo/get_model_state service call failed") 
        
        

    # Notes: Have references as topic and not service
    # Landing: Geometrical contraints for landing
    def run(self):
        rospy.logwarn('ARMING...')
        self.arm_srv.call(True)

        rospy.logwarn("Enabling computer control")
        self.pc_control.call(True)
        self.pub_command.publish(self.msg) # Spamming to avoid timeout

        # Main loop
        self.xr = [0.0 ,0.0, 2.0, 0.0 ,0.0 ,0.0 ,0.0, 0.0]  # Reference signal
        i = 0
        I = 0.02
        antiwind = 0.3
        T = 10
        traj_pos, traj_vel = self.generate_trajectory([0,0,2], [0,0,0], self.RATE,T)
        self.traj_pos0 = [traj_pos[self.RATE*T-1, 0], traj_pos[self.RATE*T-1, 1], traj_pos[self.RATE*T-1, 2]]
        self.traj_vel0 = [traj_vel[self.RATE*T-1, 0], traj_vel[self.RATE*T-1, 1], traj_vel[self.RATE*T-1, 2]]
        self.msg.z = 0
        
        n_loops = 0
        
        while not rospy.is_shutdown():
           
            n_loops += 1
            
            tmp_eul = self.state_update_service(False)

            if abs(self.tot_error[0]) <= antiwind:
                self.tot_error[0] += -self.x_states[0]*I
#
            if abs(self.tot_error[1]) <= antiwind:
                self.tot_error[1] += self.x_states[1]*I
            #if abs(self.tot_error[2]) <= 1:
            #    self.tot_error[2] += (self.xr[2]-self.x_states[2])*I

           # if abs(self.x_states[0]) < 0.1:
            #    self.tot_error[0] = 0
            #if abs(self.x_states[1]) < 0.1:
            #   self.tot_error[1] = 0
            
            
            #rospy.logwarn(self.x_states[0]-self.xr[0])
            self.xr[0] = traj_pos[i,0]
            self.xr[1] = traj_pos[i,1]
            self.xr[2] = traj_pos[i,2]
    
            self.xr[3] = traj_vel[i,0]
            self.xr[4] = traj_vel[i,1]
            self.xr[5] = traj_vel[i,2]

            self.plot_msg.data = [self.xr[0], self.xr[1]]
            # self.plot_msg.data = [  traj_pos[0,0], \
            #                    traj_pos[0,1], \
            #                    traj_pos[1,0], \
            #                    traj_pos[1,1], \
            #                    traj_pos[2,0], \
            #                    traj_pos[2,1], \
            #                    traj_pos[3,0], \
            #                    traj_pos[3,1], \
            #                    traj_pos[4,0], \
            #                    traj_pos[4,1], \
            #                    traj_pos[5,0], \
            #                    traj_pos[5,1], \
            #                    traj_pos[6,0], \
            #                    traj_pos[6,1], \
            #                    traj_pos[7,0], \
            #                    traj_pos[7,1], \
            #                    traj_pos[8,0], \
            #                    traj_pos[8,1], \
            #                    traj_pos[9,0], \
            #                    traj_pos[9,1], \
            #                    traj_pos[10,0], \
            #                    traj_pos[10,1], \
            #                    traj_pos[11,0], \
            #                    traj_pos[11,1], \
            #                    traj_pos[12,0], \
            #                    traj_pos[12,1], \
            #                    traj_pos[13,0], \
            #                    traj_pos[13,1], \
            #                    traj_pos[14,0], \
            #                    traj_pos[14,1], \
            #                    traj_pos[15,0], \
            #                    traj_pos[15,1], \
            #                    traj_pos[16,0], \
            #                    traj_pos[16,1], \
            #                    traj_pos[17,0], \
            #                    traj_pos[17,1], \
            #                    traj_pos[18,0], \
            #                    traj_pos[18,1], \
            #                    traj_pos[19,0], \
            #                    traj_pos[19,1], \
            #                    traj_pos[20,0], \
            #                    traj_pos[20,1]]

            #rospy.logwarn(np.shape(traj))
            resp1 = self.mpc_calc(self.xr, [0.0, 0.0, 9.8], self.x_states, self.tot_error)

            
            self.plot_msg.header.stamp = rospy.Time.now()
            self.pub_plotter.publish(self.plot_msg)

            ## UAV Command message
            self.msg.header.stamp = rospy.Time.now()
            self.msg.mode = Command.MODE_ROLL_PITCH_YAWRATE_THROTTLE
            self.msg.x =  math.cos(-tmp_eul[2]) * resp1.control_signals[0] + math.sin(-tmp_eul[2]) * resp1.control_signals[1]
            self.msg.y = -math.sin(-tmp_eul[2]) * resp1.control_signals[0] + math.cos(-tmp_eul[2]) * resp1.control_signals[1]
            self.msg.F = resp1.control_signals[2]/9.8
            self.pub_command.publish(self.msg)

            # Trajectory generation
            if n_loops == self.RATE*T:    
                traj_pos, traj_vel = self.generate_trajectory([2,2,2],[0,0,0], self.RATE, 3*T)
                self.traj_pos0 = [traj_pos[self.RATE*T-1, 0], traj_pos[self.RATE*T-1, 1], traj_pos[self.RATE*T-1, 2]]
                self.traj_vel0 = [traj_vel[self.RATE*T-1, 0], traj_vel[self.RATE*T-1, 1], traj_vel[self.RATE*T-1, 2]]
                n_loops = 0
                #for i in range(20*T):
                #    pose = PoseStamped()
                #    pose.pose.position.x = traj_pos[i,0]
                #    pose.pose.position.y = traj_pos[i,0]
                #    pose.pose.position.z = traj_pos[i,0]
#
                #    pose.pose.orientation.x = 0
                #    pose.pose.orientation.y = 0
                #    pose.pose.orientation.z = 0
                #    pose.pose.orientation.w = 0
                #    path_msg.poses.append(pose)
#
                #self.path_pub.publish(path_msg)
                #rospy.logwarn("NEW TRAJ")


                
            
            self.rate.sleep()

    def state_update_service(self, tag_tracking=True):
        self.states = self.get_model('multirotor', 'ground_plane')

        if tag_tracking:
            self.x_states[0] = -self.tag_x# - 0.05
            self.x_states[1] = -self.tag_y# - 0.05
        else:
            self.x_states[0] = self.states.pose.position.x
            self.x_states[1] = self.states.pose.position.y
        self.x_states[2] = self.states.pose.position.z
            
        self.x_states[3] = self.states.twist.linear.x
        self.x_states[4] = self.states.twist.linear.y
        self.x_states[5] = self.states.twist.linear.z
        y = self.states.pose.orientation.y
        x = self.states.pose.orientation.x
        z = self.states.pose.orientation.z
        w = self.states.pose.orientation.w
#           
        tmp_eul = self.quaternion_to_euler_angle_vectorized1( \
                                            self.states.pose.orientation.w, \
                                            self.states.pose.orientation.x, \
                                            self.states.pose.orientation.y, \
                                            self.states.pose.orientation.z)
                                                        
        self.x_states[6] = tmp_eul[0]   # Roll
        self.x_states[7] = tmp_eul[1]   # Pitch
        self.yaw = tmp_eul[2]
        return tmp_eul

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

    def get_tag(self, msg):
        #rospy.logwarn(type(msg.detections[0].pose.pose.pose.position.x))
        try:
            
            tag_pos = np.array([[msg.detections[0].pose.pose.pose.position.x],\
                                [msg.detections[0].pose.pose.pose.position.y],\
                                [msg.detections[0].pose.pose.pose.position.z]])

            tag_pos = self.rotz(-3.14/2, tag_pos)   # Rotate around Z
            tag_pos = self.roty(3.14, tag_pos)      # Rotate around y
            
            self.tag_x = tag_pos[0,0]
            self.tag_y = tag_pos[1,0]
            #rospy.logwarn("TAG")
            #rospy.logwarn(self.tag_x)
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

    def generate_trajectory(self, goal_pos, goal_vel, rate, T):
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
        
        #BMAT = np.array([ self.x_states[0:3], goal_pos, \
        #                  self.x_states[3:6], [0,0,0], \
        #                  [0,0,0], [0,0,0] ])
        BMAT = np.array([ self.traj_pos0, goal_pos, \
                          self.traj_vel0, [0,0,0], \
                          [0,0,0], [0,0,0] ])

        x = np.transpose(np.linalg.inv(AMAT) @ BMAT)
        test_pos = np.array([self.traj_pos0])
        test_vel = np.array([self.traj_vel0])
        #rospy.logwarn("STATES")
        #rospy.logwarn(self.x_states[3:6])
        #rospy.logwarn("INDEX0")
        #rospy.logwarn(test)

        #rospy.logwarn(test)
        
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
       
        path_msg = Path()
        pose = PoseStamped()
        rospy.logwarn(test_pos[0,0])
        pos = 0
        for pos in range(0,10):
            pose.header.stamp = rospy.Time.now()
            pose.header.frame_id = "/my_cs"
            pose.header.seq = pos
            pose.pose.position.x = pos
            pose.pose.position.y = pos + 1
            pose.pose.position.z = pos + 2
            path_msg.poses.append(pose)
            path_msg.header.frame_id = "/my_cs"
            path_msg.header.stamp = rospy.Time.now()
            
            self.path_pub.publish(path_msg)
    
            

            
        

        


        return test_pos, test_vel

        #


        #BMAT = self.x_states[0:3]
        #rospy.logwarn(x)
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
        







