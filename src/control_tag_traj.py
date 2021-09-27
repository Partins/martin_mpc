#!/usr/bin/env python3

import rospy
import time
import math
import numpy as np
from scipy.signal import cont2discrete as c2d
import cvxpy as cp
import opengen as og
import casadi.casadi as cs
#import matplotlib.pyplot as plt 


from rosflight_msgs.msg import Command, RCRaw, Status
from geometry_msgs.msg import PoseStamped, Pose, Point, Vector3
from nav_msgs.msg import Path, Odometry
from rospy.impl.transport import INBOUND
from std_msgs.msg import Float64, String, Int64
from gazebo_msgs.msg import ModelStates, ModelState
from gazebo_msgs.srv import GetModelState, SetModelState
from apriltag_ros.msg import AprilTagDetectionArray
from rosflight_extras.srv import arm_uav
from rosflight_extras.msg import float_array
from martin_mpc.srv import mpcsrv, landsrv, gotosrv
#from mav_msgs.msg import RollPitchYawrateThrust


#Params
import dynamic_reconfigure.client

class UAV:

    def dyn_callback(self, config):
        self.tag_follow = config.tag_follow
        self.tot_error[0] = 0
        self.tot_error[1] = 0
        self.Q_state  = [config.Q0, config.Q1, config.Q2, config.Q3, \
                           config.Q4, config.Q5, config.Q6, config.Q7]
        
        self.Q_hovering  = [config.R1, config.R2, config.R3]

        self.Q_u = [config.Rd1, config.Rd2, config.Rd3]

        self.K  = [config.K0, config.K1, config.K2, config.K3, \
                            config.K4, config.K5, config.K6, config.K7]
    

    def __init__(self):

        self.gazebo = True
        rospy.init_node('UAV_control', anonymous=True)
        self.x_states_topic   = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        ## Subscribers
        self.tag_pos            = rospy.Subscriber('tag_detections', AprilTagDetectionArray, self.get_tag)
        self.control_command    = rospy.Subscriber('control_command', String, self.user_command)

        if self.gazebo:
            self.pub_command    = rospy.Publisher('command', Command, queue_size=1)
            self.state_update   = rospy.Subscriber('multirotor/truth/NWU', Odometry, self.get_state)
        else:
            self.state_update   = rospy.Subscriber('vicon/shafter2/shafter2/odom', Odometry, self.get_state)
            self.pub_command    = rospy.Publisher('command/RollPitchYawrateThrust', RollPitchYawrateThrust, queue_size=1)    


        ## Publishers
        self.path_pub           = rospy.Publisher('path', Path, queue_size=1)
        self.uav_path_pub       = rospy.Publisher('uav_path', Path, queue_size=1)
        #self.pub_plotter        = rospy.Publisher('plotter', float_array, queue_size=1)

        ## Services
        if self.gazebo:
            self.arm_srv            = rospy.ServiceProxy('arm_UAV', arm_uav)
            self.pc_control         = rospy.ServiceProxy('pc_control', arm_uav)
            self.set_state_service  = rospy.ServiceProxy('gazebo/set_model_state', SetModelState)
        
        self.mpc_calc           = rospy.ServiceProxy('MPC_calc', mpcsrv)
        self.tag_land_srv       = rospy.Service('Tag_land_UAV', landsrv, self.tag_landsrv)
        self.tag_follow_srv     = rospy.Service('Tag_follow', landsrv, self.follow_tag)
        self.tag_align_srv       = rospy.Service('Tag_align_UAV', landsrv, self.tag_alignsrv)
        self.land_srv           = rospy.Service('Land_UAV', landsrv, self.landsrv)
        self.takeoff_srv        = rospy.Service('Takeoff_UAV', landsrv, self.takeoffsrv)
        self.goto_srv           = rospy.Service('GOTO', gotosrv, self.gotosrv)
        self.abort_srv           = rospy.Service('abort', landsrv, self.abort)
        
        ## Messages
        if self.gazebo:
            self.msg        = Command()     # 1st Config in enable_computer_control()
        else:
            self.msg        = RollPitchYawrateThrust()
        #self.plot_msg   = float_array() # Cutom float array msg type

        ## ROS
        self.RATE = 20 # [Hz]
        self.rate = rospy.Rate(self.RATE)
        self.is_armed = False

        ## Inits
        self.tot_error  = [0.0, 0.0, 0.0]    
        self.prev_error = [0, 0, 0]
        self.error_test = [0.0, 0.0, 0.0]    
        self.tag_x      = 0
        self.tag_y      = 0
        self.tag_z      = 0
        self.pos        = [0, 0, 0]
        self.yaw        = 0
        self.yaw_setpoint = 0
        self.land       = 0
        self.xr         = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] 
        self.x_states   = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] 
        self.aborting = False
        self.tag_landing = False
        self.tag_landing_cntr = 0
        self.ref_inputs = [0, 0, 4.9]
        self.point_xr = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] 
        
        self.thrust = Vector3()
        self.thrust.x = 0
        self.thrust.y = 0
        self.thrust.z = 0

        self.path_msg = Path() # Trajectory
        self.uav_path_msg = Path() # UAV
        self.P = 0
        self.I = 0
        self.D = 0

        self.Q_state=[1.0, 1.0, 1.0, 1, 1, 1, 1, 1]
        self.Q_hovering=[1.0,1.0,1]
        self.Q_u=[1,1,1]
        


        if self.gazebo:
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

        # Dynamic reconfiguration
        self.dyn_client = dynamic_reconfigure.client.Client("martin_mpc_param_node", timeout=30, config_callback=self.dyn_callback)
        while self.x_states[0] == 0.0:
            rospy.logwarn("Waiting for state update")
            self.get_current_state(False)
            self.zero_pos = [self.x_states[0], self.x_states[1], self.x_states[2]]
            rospy.logwarn(self.x_states)
        self.mng = og.tcp.OptimizerTcpManager('python_build/optimizer01')
        self.mng.start()
        while not self.mng.ping():
            rospy.logwarn("Waiting for solver")
            print('.')
            time.sleep(1)

        
        

    def run(self):
        
             # check if the server is alive
        if self.gazebo:
            rospy.logwarn('ARMING...')
            self.arm_srv.call(True)

            rospy.logwarn("Enabling computer control")
            self.pc_control.call(True)
            self.pub_command.publish(self.msg) # Spamming to avoid timeout

        self.zero_pos = [self.x_states[0], self.x_states[2], self.x_states[2]]

        # Generate "fake" trajectory to stay in place when starting. This is 
        # mainly to have to code working further down
        self.traj_pos, self.traj_vel = self.generate_trajectory([self.zero_pos[0], self.zero_pos[1], self.zero_pos[2]], \
                                                                [0,0,0], \
                                                                [self.zero_pos[0], self.zero_pos[1], self.zero_pos[2]], \
                                                                [0,0,0], 1,1)
        print(self.traj_vel)

        

####################################### main thing ############################################
        
        # Counters
        self.traj_index = 0
        cntr = 0

        if self.gazebo:
            self.msg.F = 0
        else: 
            self.msg.thrust = 0  

        while not rospy.is_shutdown():

            if not self.aborting:
                cntr += 1
                self.traj_index += 1


                if self.tag_landing == True: # if tag-landing then generate a new one
                    self.tag_landing_cntr +=1
                    self.tag_landsrv([''])

                if self.tag_follow:
                    self.follow_tag([''])

                x_0=self.x_states+ self.point_xr + self.ref_inputs + self.Q_state+ self.Q_hovering + self.Q_u#concatenate lists in Python
                print("POINT XR")
                print(self.point_xr)
                solution = self.mng.call(x_0)
                u0=solution[u'solution']
                resp1 = u0[0:3]
                yaw_error = self.yaw_setpoint - self.yaw

                if self.gazebo:
                    self.msg.header.stamp = rospy.Time.now()
                    self.msg.mode = Command.MODE_ROLL_PITCH_YAWRATE_THROTTLE
                    self.msg.x =  (math.cos(-self.yaw) * resp1[0] + math.sin(-self.yaw) * resp1[1])
                    self.msg.y = -(-math.sin(-self.yaw) * resp1[0] + math.cos(-self.yaw) * resp1[1])
                    self.msg.z = -yaw_error
                    print(resp1[2])
                    self.msg.F = resp1[2]/14.961
                    self.pub_command.publish(self.msg)
                    

                else:
                    self.msg.header.stamp = rospy.Time.now()
                    self.msg.roll =  math.cos(-self.yaw) * resp1.control_signals[0] + math.sin(-self.yaw) * resp1.control_signals[1]
                    self.msg.pitch = -(-math.sin(-self.yaw) * resp1.control_signals[0] + math.cos(-self.yaw) * resp1.control_signals[1])
                    self.msg.yaw_rate = -yaw_error
                    self.thrust.z = resp1.control_signals[2]/14.961
                    self.msg.thrust = self.thrust
                    self.pub_command.publish(self.msg)

                #rospy.logwarn(cntr)
                self.rate.sleep()
            

########################################### MAIN END ###################################################################
    
    
    def landsrv(self, req):
        T = 3 # Time to land

        rospy.logwarn("Landing Service Initiated")
        
        tic = rospy.Time.now()
        self.traj_pos, self.traj_vel = self.generate_trajectory( \
                [self.x_states[0], self.x_states[1], self.x_states[2]], \
                [self.x_states[3], self.x_states[4], self.x_states[5]], \
                [self.x_states[0], self.x_states[1], 0.0], [0,0,0], self.RATE,T)
     
        toc = rospy.Time.now()
        rospy.logwarn("Landing trajectory generated with: " + str(self.RATE * T) + " points")
        self.traj_index = 0

        self.path_msg.header.frame_id = "map"
        self.path_msg.header.stamp = rospy.Time.now()
        for i in range(T*self.RATE):
            pose = PoseStamped()
            pose.pose.position.x = self.traj_pos[i,0]
            pose.pose.position.y = self.traj_pos[i,1]
            pose.pose.position.z = self.traj_pos[i,2]
            self.path_msg.poses.append(pose)
        self.path_pub.publish(self.path_msg)
        self.point_xr = [self.x_states[0], self.x_states[1], 0, 0.0, 0.0, 0.0, 0.0, 0.0] 
        self.ref_inputs = [0.0, 0.0, 5.0]
        return True
    def follow_tag(self, req):
        self.tag_follow = True
        rospy.logwarn("Following tag")
        #self.point_xr = [self.x_states[0]-self.tag_y, self.x_states[1]-self.tag_x, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0] 
        self.point_xr = [self.x_states[0]-self.tag_y, self.x_states[1]-self.tag_x, 1.0, -self.tag_y, -self.tag_x, 0.0, 0.0, 0.0]

        
    def tag_alignsrv(self, req):
        T =3# - self.tag_landing_cntr # Time to tag

        rospy.logwarn("Tag Landing Service Initiated")
        self.point_xr = [self.x_states[0]-self.tag_y, self.x_states[1]-self.tag_x, self.x_states[2], 0.0, 0.0, -0.5, 0.0, 0.0] 
        #rospy.logwarn('Zerror')
        #rospy.logwarn(zerr)
        self.traj_pos, self.traj_vel = self.generate_trajectory( \
                [self.x_states[0], self.x_states[1], self.x_states[2]], \
                [self.x_states[3], self.x_states[4], self.x_states[5]], \
                [self.x_states[0]-self.tag_y, self.x_states[1]-self.tag_x, self.x_states[2]], [0,0,0], self.RATE,T)

        rospy.logwarn("Landing trajectory generated with: " + str(self.RATE * T) + " points")
        self.traj_index = 0

        self.path_msg.header.frame_id = "map"
        self.path_msg.header.stamp = rospy.Time.now()
        for i in range(T*self.RATE):
            pose = PoseStamped()
            pose.pose.position.x = self.traj_pos[i,0]
            pose.pose.position.y = self.traj_pos[i,1]
            pose.pose.position.z = self.traj_pos[i,2]
            self.path_msg.poses.append(pose)
        self.path_pub.publish(self.path_msg)
        self.point_xr = [self.x_states[0]-self.tag_y, self.x_states[1]-self.tag_x, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0] 

        return True
    def tag_landsrv(self, req):
        #T =10# - self.tag_landing_cntr # Time to tag
        self.tag_landing = True
        rospy.logwarn("Tag Landing Service Initiated")
        
        tic = rospy.Time.now()
        #generate_trajectory(self, start_pos, start_vel, goal_pos, goal_vel, rate, T):
        if self.x_states[2]-0.2 <= 0:
            zerr = 0
        else:
            zerr = self.x_states[2]-0.2
        print(zerr)
        self.point_xr = [self.x_states[0]-self.tag_y, self.x_states[1]-self.tag_x, zerr, 0.0, 0.0, 0.0, 0.0, 0.0] 
        #rospy.logwarn('Zerror')
        #rospy.logwarn(zerr)
        self.traj_pos, self.traj_vel = self.generate_trajectory( \
                [self.x_states[0], self.x_states[1], self.x_states[2]], \
                [self.x_states[3], self.x_states[4], self.x_states[5]], \
                [self.x_states[0]-self.tag_y, self.x_states[1]-self.tag_x, 0.0], [0,0,0], self.RATE,T)

        rospy.logwarn("Landing trajectory generated with: " + str(self.RATE * T) + " points")
        self.traj_index = 0

        self.path_msg.header.frame_id = "map"
        self.path_msg.header.stamp = rospy.Time.now()
        for i in range(T*self.RATE):
            pose = PoseStamped()
            pose.pose.position.x = self.traj_pos[i,0]
            pose.pose.position.y = self.traj_pos[i,1]
            pose.pose.position.z = self.traj_pos[i,2]
            self.path_msg.poses.append(pose)
        self.path_pub.publish(self.path_msg)

        #if self.x_states[2] < 0.1:
        #    self.tag_landing = False

        return True

    def abort(self, req):
        self.aborting = True
        self.tot_error[0] = 0
        self.tot_error[1] = 0
        rospy.logwarn("ABORTING")
        
        self.traj_pos, self.traj_vel = self.generate_trajectory(  \
            [self.x_states[0],self.x_states[1],self.x_states[2]], \
            [self.x_states[3],self.x_states[4],self.x_states[5]], \
            [self.x_states[0],self.x_states[1],self.x_states[2]], \
            [0,0,0], self.RATE,10)

        while self.msg.F > 0.1:
            rospy.logwarn("Abortin...")
            self.msg.header.stamp = rospy.Time.now()
            self.msg.mode = Command.MODE_ROLL_PITCH_YAWRATE_THROTTLE
            self.msg.x = 0
            self.msg.y = 0
            self.msg.F = self.msg.F - 0.01
            self.pub_command.publish(self.msg)
            self.rate.sleep()
#
        # Generate a trajectory to stay in place after abort
        tmp_eul = self.get_current_state(self.tag_follow)
        self.traj_index = 0
        self.traj_pos, self.traj_vel = self.generate_trajectory(  \
            [self.x_states[0],self.x_states[1],self.x_states[2]], \
            [self.x_states[3],self.x_states[4],self.x_states[5]], \
            [self.x_states[0],self.x_states[1],self.x_states[2]], \
            [0,0,0], self.RATE,10)
        self.ref_inputs = [0, 0, 0]
        self.aborting = False
        return True

    def takeoffsrv(self, req):
        T = 3 # Time to takeoff

        rospy.logwarn("Takeoff Service Initiated")
        self.tot_error[0] = 0
        self.tot_error[1] = 0
        tic = rospy.Time.now()
        self.traj_pos, self.traj_vel = self.generate_trajectory( \
                [self.x_states[0], self.x_states[1], self.x_states[2]], \
                [self.x_states[3], self.x_states[4], self.x_states[5]], \
                [self.x_states[0], self.x_states[1], 1.5], [0,0,0], self.RATE,T)
        toc = rospy.Time.now()
        rospy.logwarn(self.traj_pos)
        
        self.traj_index = 0

        self.path_msg.header.frame_id = "map"
        self.path_msg.header.stamp = rospy.Time.now()
        for i in range(T*self.RATE):
            pose = PoseStamped()
            pose.pose.position.x = self.traj_pos[i,0]
            pose.pose.position.y = self.traj_pos[i,1]
            pose.pose.position.z = self.traj_pos[i,2]
            self.path_msg.poses.append(pose)
        self.path_pub.publish(self.path_msg)
        self.ref_inputs = [0,0,9.81]
        self.point_xr = [0.0, 0.0, 1.50, 0.0, 0.0, 0.0, 0.0, 0.0] 
        
        return True

    def gotosrv(self, msg):
        if msg.z == 0.0:
            goal_height = self.point_xr[2]
        else:
            goal_height = msg.z
        if msg.T == 0.0:
            msg.T = 1.0
        self.traj_pos, self.traj_vel = self.generate_trajectory( \
                [self.x_states[0], self.x_states[1], self.x_states[2]], \
                [self.x_states[3], self.x_states[4], self.x_states[5]], \
                [self.zero_pos[0]+msg.x,self.zero_pos[1]+msg.y, goal_height], [0,0,0], self.RATE, int(msg.T))
        self.traj_index = 0

        self.path_msg.header.frame_id = "map"
        self.path_msg.header.stamp = rospy.Time.now()
        for i in range(int(msg.T)*self.RATE):
            pose = PoseStamped()
            pose.pose.position.x = self.traj_pos[i,0]
            pose.pose.position.y = self.traj_pos[i,1]
            pose.pose.position.z = self.traj_pos[i,2]
            self.path_msg.poses.append(pose)
        self.path_pub.publish(self.path_msg)
        self.point_xr = [self.zero_pos[0]+msg.x,self.zero_pos[1]+msg.y, goal_height, 0.0, 0.0, 0.0, 0.0, 0.0]
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
        
        self.uav_path_msg.header.frame_id = "map"
        self.uav_path_msg.header.stamp = rospy.Time.now()
        pose = PoseStamped()
        pose.pose.position.x = self.x_states[0]
        pose.pose.position.y = self.x_states[1]
        pose.pose.position.z = self.x_states[2]
        self.uav_path_msg.poses.append(pose)
        self.uav_path_pub.publish(self.uav_path_msg)
        #rospy.logwarn("GET CURRENT STATE")
        #print(self.x_states)
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
        if self.gazebo:
            mult = 1
        else:
            mult = 1
        self.x_states_topic[0] = msg.pose.pose.position.x
        self.x_states_topic[1] = mult*msg.pose.pose.position.y
        self.x_states_topic[2] = mult*msg.pose.pose.position.z
        
        # Linear velocities
        self.x_states_topic[3] = msg.twist.twist.linear.x
        self.x_states_topic[4] = mult*msg.twist.twist.linear.y
        self.x_states_topic[5] = mult*msg.twist.twist.linear.z

        tmp_eul = self.quaternion_to_euler_angle_vectorized1( \
                                            msg.pose.pose.orientation.w, \
                                            msg.pose.pose.orientation.x, \
                                            msg.pose.pose.orientation.y, \
                                            msg.pose.pose.orientation.z)
                                                        
        self.x_states_topic[6] = tmp_eul[0]   # Roll
        self.x_states_topic[7] = tmp_eul[1]   # Pitch
        self.yaw = tmp_eul[2]
        self.x_states = self.x_states_topic
        #rospy.logwarn("GET STATE")
        #print(self.x_states_topic)

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

        return test_pos, test_vel          
    
if __name__ == '__main__':
    try:
        multirotor = UAV()
        multirotor.run()
    except rospy.ROSInterruptException:
        pass
        







