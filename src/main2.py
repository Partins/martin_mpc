#!/usr/bin/env python3

import rospy
import time
import math
import numpy as np
from scipy.signal import cont2discrete as c2d
import cvxpy as cp
import opengen as og
import casadi.casadi as cs 


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

        self.Q_hovering_traj  = [config.Rt1, config.Rt2, config.Rt3]

        self.Q_u = [config.Rd1, config.Rd2, config.Rd3]

        self.Q_u_traj = [config.Rdt1, config.Rdt2, config.Rdt3]

        self.K  = [config.K0, config.K1, config.K2, config.K3, \
                            config.K4, config.K5, config.K6, config.K7]

        self.E  = [config.E0, config.E1, config.E2, config.E3, \
                            config.E4, config.E5, config.E6, config.E7]
        self.E_uav = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 100.0, 100.0]

        self.P = config.P

        #self.generate_trajectory_optimizer(self.xr, self.point_xr)
        print(self.P)
    

    def __init__(self):

        self.gazebo = True

        rospy.init_node('UAV_control', anonymous=True)
        self.x_states_topic   = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        ## Subscribers
        self.tag_pos            = rospy.Subscriber('tag_detections', AprilTagDetectionArray, self.get_tag)
        self.control_command    = rospy.Subscriber('control_command', String, self.user_command)
        self.x_states   = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] 
        if self.gazebo:
            self.pub_command    = rospy.Publisher('command', Command, queue_size=1)
            #self.state_update   = rospy.Subscriber('multirotor/truth/NWU', Odometry, self.get_state)
            self.state_update   = rospy.Subscriber('/multirotor/odom/gazebo/body/sensorfusion', Odometry, self.get_state)
            
        else:
            self.state_update   = rospy.Subscriber('vicon/shafter_martin/shafter_martin/odom', Odometry, self.get_state)
            self.pub_command     = rospy.Publisher('command/roll_pitch_yawrate_thrust', RollPitchYawrateThrust, queue_size=1)  


        ## Publishers
        self.path_pub           = rospy.Publisher('path', Path, queue_size=1)
        self.uav_path_pub       = rospy.Publisher('uav_path', Path, queue_size=1)
        self.pub_plotter        = rospy.Publisher('plotter', float_array, queue_size=1)

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
        self.goto_point_srv           = rospy.Service('Point_GOTO', landsrv, self.goto_pointsrv)
        self.abort_srv           = rospy.Service('abort', landsrv, self.abort)
        self.tag_detected       = False
        
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
        self.tag_follow = False
        self.tag_align = False
        self.u = 0
        self.thrust = Vector3()
        self.thrust.x = 0
        self.thrust.y = 0
        self.thrust.z = 0
        self.tag_orientation = [0,0,0]

        self.path_msg = Path() # Trajectory
        self.uav_path_msg = Path() # UAV
        self.P = 0
        self.I = 0
        self.D = 0

        self.Q_state=[4.0, 4.0, 15.0, 1.20, 1.20, 4.0, 1.0, 1.0]
        self.Q_hovering=[1.0,1.0,1.0]
        self.Q_u=[1.0,1.0,2.0]
        self.E_uav = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 100.0, 100.0]

        self.K = [1.0, 1.0, 5.0, 1.0, 1.0, 8.0, 5.0, 5.0]
        self.Q_hovering_traj=[1.0,1.0,15.0]
        self.Q_u_traj=[10,10,15]
        self.E = [1.0, 1.0, 6.0, 3.0, 3.0, 30.0, 5.0, 5.0]

        

        
        


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
        
        while self.x_states[0] == 0.0:
            rospy.logwarn("Waiting for state update")
            #self.get_current_state(False)
            self.zero_pos = [self.x_states[0], self.x_states[1], self.x_states[2]]
            rospy.logwarn(self.x_states)
        self.point_xr = [self.x_states[0], self.x_states[1], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.mng = og.tcp.OptimizerTcpManager('python_build/uav_controller')
        self.mng.start()

        self.mng_traj = og.tcp.OptimizerTcpManager('python_build2/trajectory_generator')
        self.mng_traj.start()
        while not self.mng.ping():
            rospy.logwarn("Waiting for solver")
            print('.')
            time.sleep(1)
        while not self.mng_traj.ping():
            rospy.logwarn("Waiting for solver")
            print('.')
            time.sleep(1)
        self.dyn_client = dynamic_reconfigure.client.Client("martin_mpc_param_node", timeout=30, config_callback=self.dyn_callback)

        
        

    def run(self):
        
             # check if the server is alive
        if self.gazebo:
            rospy.logwarn('ARMING...')
            self.arm_srv.call(True)

            rospy.logwarn("Enabling computer control")
            self.pc_control.call(True)
            self.pub_command.publish(self.msg) # Spamming to avoid timeout

        self.zero_pos = [self.x_states[0], self.x_states[1], self.x_states[2]]
        
        # Generate "fake" trajectory to stay in place when starting. This is 
        # mainly to have to code working further down
        #self.traj_pos, self.traj_vel = self.generate_trajectory([self.zero_pos[0], self.zero_pos[1], self.zero_pos[2]], \
        #                                                        [0,0,0], \
        #                                                        [self.zero_pos[0], self.zero_pos[1], self.zero_pos[2]], \
        #                                                        [0,0,0], 1,1)

        

####################################### main thing ############################################
        
        # Counters
        self.traj_counter = 0
        cntr = 0

        if self.gazebo:
            self.msg.F = 0
        else: 
            self.msg.thrust = 0  
        N = 20
        self.generate_trajectory_optimizer(self.x_states, self.x_states)
        self.traj_index = 0
        self.point_going = False
        while not rospy.is_shutdown():
            
            if not self.aborting:
                self.traj_counter += 1


                if self.tag_landing == True: # if tag-landing then generate a new one
                    self.tag_landing_cntr +=1
                    #if self.tag_landing_cntr >= 40:
                    #    self.tag_alignsrv([''])
                    

                
                if self.tag_follow:
                    self.follow_tag([''])

                if self.traj_index >= N and self.point_going != True:
                    if self.tag_landing:
                        self.tag_landsrv([''])
                    if self.tag_align:
                        self.tag_alignsrv([''])
                    
                    self.generate_trajectory_optimizer(self.xr, self.point_xr)
                    #print("WHOOPS")
                    self.traj_index = 0
                    self.traj_counter = 0
                # Setting the next point in the trajectory as reference

                # Control if UAV follows trajectory or goes to point

                if self.point_going:
                    x_0=self.x_states+ self.point_xr + self.ref_inputs + self.Q_state+ self.Q_hovering + self.Q_u + self.E_uav#concatenate lists in Python

                else:
                    self.xr[0] = self.traj_pos[self.traj_index,0]  #
                    self.xr[1] = self.traj_pos[self.traj_index,1]  # Position
                    self.xr[2] = self.traj_pos[self.traj_index,2]  #
                    self.xr[3] = self.traj_vel[self.traj_index,0]  #
                    self.xr[4] = self.traj_vel[self.traj_index,1]  # Velocity
                    self.xr[5] = self.traj_vel[self.traj_index,2]  #
                    #self.convertref2global()
                    x_0 = self.x_states + self.xr + self.ref_inputs + self.Q_state+ self.Q_hovering + self.Q_u + self.E_uav#concatenate lists in Python
                plot_msg = float_array()
                plot_msg.data = self.xr
                plot_msg.header.stamp = rospy.Time.now()
                self.pub_plotter.publish(plot_msg)

                solution = self.mng.call(x_0)
                u0=solution[u'solution']
                self.u = u0
                #print(u0)
                resp1 = u0[0:3]
                yaw_error = self.yaw_setpoint - self.yaw
               
                if self.gazebo:

                    self.msg.header.stamp = rospy.Time.now()
                    self.msg.mode = Command.MODE_ROLL_PITCH_YAWRATE_THROTTLE
                    roll = resp1[0]
                    pitch = resp1[1]
                    self.msg.x = roll#math.cos(self.yaw) * roll + math.sin(self.yaw) * pitch # ROLL
                    self.msg.y = -pitch#-math.sin(self.yaw) * roll + math.cos(self.yaw) * pitch # PITCH
                    #self.msg.y = -self.msg.y
                    #self.msg.x = math.cos(self.yaw) * resp1[1] + math.sin(self.yaw) * resp1[0]
                    #self.msg.y = -1 * ( math.sin(self.yaw) * resp1[1] + math.cos(self.yaw) * resp1[0] )
                    if self.tag_detected:
                        self.msg.z = 0.1*self.tag_orientation[2]
                    else:
                        self.msg.z = 0
                    #print(resp1[2])
                    self.msg.F = resp1[2]/(14.961+3) #self.P
                    self.pub_command.publish(self.msg)
                    

                #else:
                #    self.yaw = -self.yaw
                #    self.msg.header.stamp = rospy.Time.now()
                #    self.msg.roll =  math.cos(self.yaw) * resp1[0] + math.sin(self.yaw) * resp1[1]
                #    self.msg.pitch = (-math.sin(self.yaw) * resp1[0] + math.cos(self.yaw) * resp1[1])
                #    self.msg.yaw_rate = yaw_error
                #    self.thrust.z = resp1[2]/(14.961+9.82)
                #    self.msg.thrust = self.thrust
                #    self.pub_command.publish(self.msg)
                if self.traj_counter >= 10:
                    self.traj_index += 1
                    #print(i)
                    self.traj_counter = 0
                #print(self.yaw)
                #print(self.traj_pos[-1,0:3])
                #print(self.point_xr)
                #rospy.logwarn(cntr)
                self.msg.z = 5*self.tag_orientation[2]
                self.show_path_uav()
                self.rate.sleep()
            

########################################### MAIN END ###################################################################
    
    def convertref2global(self):
        self.trajectory = self.point_xr
        x_ref = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        x_ref[0]=math.cos(self.yaw)*(self.trajectory[0])+math.sin(self.yaw)*(self.trajectory[1])
        x_ref[1]=-math.sin(self.yaw)*(self.trajectory[0])+math.cos(self.yaw)*(self.trajectory[1])
        x_ref[2]=self.trajectory[2]
        x_ref[3]=self.trajectory[3]
        x_ref[4]=self.trajectory[4]
        x_ref[5]=self.trajectory[5]
        x_ref[6] = self.trajectory[6]
        x_ref[7] = self.trajectory[7]

        self.point_xr = x_ref

    def landsrv(self, req):
        self.tag_landing = True
        if self.x_states[2]-0.2 <= 0:
            zerr = 0
        else:
            zerr = self.x_states[2]-0.01

        rospy.logwarn("Landing Service Initiated")
        self.traj_counter = 0

        self.point_xr = [self.x_states[0], self.x_states[1], zerr, 0.0, 0.0, -0.01, 0.0, 0.0] 
        self.ref_inputs = [0.0, 0.0, 9.82]
        self.point_going = False
        self.tag_align = True
        self.tag_landing = False
        self.traj_index = 100

        return True
    def follow_tag(self, req):
        self.tag_follow = True
        rospy.logwarn("Following tag")
        #self.point_xr = [self.x_states[0]-self.tag_y, self.x_states[1]-self.tag_x, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0] 
        self.point_xr = [self.x_states[0]-self.tag_y, self.x_states[1]-self.tag_x, 1.0, -self.tag_y, -self.tag_x, 0.0, 0.0, 0.0]

        
    def tag_alignsrv(self, req):
        T =3# - self.tag_landing_cntr # Time to tag

        rospy.logwarn("Tag Landing Service Initiated")
        self.point_xr = [self.x_states[0]-self.tag_y, self.x_states[1]-self.tag_x, self.point_xr[2], 0.0, 0.0, 0.0, 0.0, 0.0] 
        #rospy.logwarn('Zerror')
        #rospy.logwarn(zerr)
       
        #self.generate_trajectory_optimizer(self.x_states, self.point_xr)
        self.point_going = False
        self.tag_align = True
        self.tag_landing = False
        self.traj_index = 100
        return True
    def tag_landsrv(self, req):
        #T =10# - self.tag_landing_cntr # Time to tag
        

        #old_goal_height = self.point_xr[2]
        #start = self.x_states
        #start[2] = old_goal_height
        self.point_xr = [self.x_states[0]-self.tag_y, self.x_states[1]-self.tag_x, 0.05, 0.0, 0.0, -0.01, 0.0, 0.0] 
        #self.generate_trajectory_optimizer(start, self.point_xr)
        self.traj_counter = 0
        self.tag_landing = True
        self.point_going = False
        self.tag_align = False
        self.traj_index = 100
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
        self.traj_counter = 0
        self.traj_pos, self.traj_vel = self.generate_trajectory(  \
            [self.x_states[0],self.x_states[1],self.x_states[2]], \
            [self.x_states[3],self.x_states[4],self.x_states[5]], \
            [self.x_states[0],self.x_states[1],self.x_states[2]], \
            [0,0,0], self.RATE,10)
        self.ref_inputs = [0, 0, 0]
        self.aborting = False
        return True

    def takeoffsrv(self, req):

        self.point_xr = [self.zero_pos, self.zero_pos[1], 1.50, 0.0, 0.0, 0.0, 0.0, 0.0] 
        self.generate_trajectory_optimizer(self.x_states, self.point_xr)
        
        self.tag_landing = False
        return True

    def gotosrv(self, msg):
        if msg.z == 0.0:
            goal_height = self.point_xr[2]
        else:
            goal_height = msg.z
            self.ref_inputs = [0,0,9.82]
        #if msg.T == 0.0:
        #    msg.T = 1.0
        #self.traj_pos, self.traj_vel = self.generate_trajectory( \
        #        [self.x_states[0], self.x_states[1], self.x_states[2]], \
        #        [self.x_states[3], self.x_states[4], self.x_states[5]], \
        #        [self.zero_pos[0]+msg.x,self.zero_pos[1]+msg.y, goal_height], [0,0,0], self.RATE, int(msg.T))
        #self.traj_counter = 0
        old_goal_height = self.point_xr[2]
        self.point_xr = [self.zero_pos[0]+msg.x,self.zero_pos[1]+msg.y, goal_height, 0.0, 0.0, 0.0, 0.0, 0.0]
        start = self.xr
        #start[2] = old_goal_height
        self.generate_trajectory_optimizer(start, self.point_xr)
        print("GOTO")
        print(self.u)
        self.tag_landing = False
        self.point_going = False
        self.tag_align = False
        self.tag_landing = False
        self.traj_index = 100

        return True
    
    def goto_pointsrv(self, msg):

        if self.point_going:
            self.point_going = False
        else:
            self.point_going = True


        return True       

    ## Get current state
    #   This function gets the state of the UAV without race conditions. 
    #   If the tag_tracking is set to True the position will be relative to the
    #   QR-code while setting it to False it gives the absolute position in 
    #   the world frame. 

    #def get_current_state(self, tag_tracking=True):
    #    #self.states = self.get_model('multirotor', 'ground_plane')
    #    self.states = self.x_states_topic
    #    # Relative vs. absolute positioning
    #    if tag_tracking:
    #        self.x_states[0] = self.tag_y# - 0.05
    #        self.x_states[1] = self.tag_x# - 0.05
    #    else:
    #        self.x_states[0] = self.states[0]
    #        self.x_states[1] = self.states[1]
    #    self.x_states[2] = self.states[2]
    #    
    #    # Linear velocities
    #    self.x_states[3] = self.states[3]
    #    self.x_states[4] = self.states[4]
    #    self.x_states[5] = self.states[5]
    #   
    #    self.x_states[6] = self.states[6]   # Roll
    #    self.x_states[7] = self.states[7]   # Pitch
    #    
        #self.uav_path_msg.header.frame_id = "world"
        #self.uav_path_msg.header.stamp = rospy.Time.now()
        #pose = PoseStamped()
        #pose.pose.position.x = self.x_states[0]
        #pose.pose.position.y = self.x_states[1]
        #pose.pose.position.z = self.x_states[2]
        #self.uav_path_msg.poses.append(pose)
        #self.uav_path_pub.publish(self.uav_path_msg)
        #rospy.logwarn("GET CURRENT STATE")
        #print(self.x_states)
    #    return [0,0,self.yaw]
    def user_command(self, msg):
        if msg.data == "land":
            self.land = 1
        else:
            self.land = 0

    def quaternion_to_euler_angle_vectorized1(self, q0, q1, q2, q3):

        #t0 = +2.0 * (w * x + y * z)
        #t1 = +1.0 - 2.0 * (x * x + y * y)
        #roll_x = math.atan2(t0, t1)
     #
        #t2 = +2.0 * (w * y - z * x)
        #t2 = +1.0 if t2 > +1.0 else t2
        #t2 = -1.0 if t2 < -1.0 else t2
        #pitch_y = math.asin(t2)
     #
        #t3 = +2.0 * (w * z + x * y)
        #t4 = +1.0 - 2.0 * (y * y + z * z)
        #yaw_z = math.atan2(t3, t4)

        t0 = +2.0 * (q0 * q1 + q2 * q3)
        t1 = q0*q0 - q1*q1 - q2*q2 + q3*q3 
        roll_x = math.atan2(t0, t1)

        s0 = 2*(q0*q2-q3*q1)
        pitch_y = math.asin(s0)

        r0 = +2 * (q0*q3+q1*q2)
        r1 = q0*q0 + q1*q1 - q2*q2 - q3*q3
        yaw_z = math.atan2(r0,r1)

        
        return roll_x, pitch_y, yaw_z 

    ## Get State
    # Subscribes to the Odometry topic and converts to world frame. 

    def get_state(self, msg):
 
        self.x_states[0] = msg.pose.pose.position.x
        self.x_states[1] = msg.pose.pose.position.y
        self.x_states[2] = msg.pose.pose.position.z
        
        # Linear velocities
        self.x_states[3] = msg.twist.twist.linear.x
        self.x_states[4] = msg.twist.twist.linear.y
        self.x_states[5] = msg.twist.twist.linear.z

        tmp_eul = self.quaternion_to_euler_angle_vectorized1( \
                                            msg.pose.pose.orientation.w, \
                                            msg.pose.pose.orientation.x, \
                                            msg.pose.pose.orientation.y, \
                                            msg.pose.pose.orientation.z)
                                                        
        #self.x_states[6] = tmp_eul[0]   # Roll
        #self.x_states[7] = tmp_eul[1]   # Pitch
        self.x_states[6] = msg.pose.pose.orientation.x
        self.x_states[7] = msg.pose.pose.orientation.y
        self.yaw = msg.pose.pose.orientation.z


    def get_tag(self, msg):

        try:
            tag_pos = np.array([[msg.detections[0].pose.pose.pose.position.x],\
                                [msg.detections[0].pose.pose.pose.position.y],\
                                [msg.detections[0].pose.pose.pose.position.z]])

            tmp = np.array([[msg.detections[0].pose.pose.pose.orientation.w],\
                                [msg.detections[0].pose.pose.pose.orientation.x],\
                                [msg.detections[0].pose.pose.pose.orientation.y],\
                                [msg.detections[0].pose.pose.pose.orientation.z]])

            self.tag_orientation = self.quaternion_to_euler_angle_vectorized1(\
                tmp[0],\
                tmp[1],\
                tmp[2],\
                tmp[3])

            #tag_pos = self.rotz(-3.14/2, tag_pos)   # Rotate around Z
            #tag_pos = self.roty(3.14, tag_pos)      # Rotate around y
            
            self.tag_x = tag_pos[0,0]
            self.tag_y = tag_pos[1,0]
            self.tag_detected = True
        except:
            self.tag_detected = False
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
    def generate_trajectory_optimizer(self, start, goal):
        #self.ref_inputs =  [0.0, 0.0, 9.82]
        #x_0=self.x_states+ goal + ref_inputs + self.Q_state+ self.Q_hovering + self.Q_u #concatenate lists in Python
        x_0=start+ goal + self.ref_inputs + self.K+ self.Q_hovering_traj + self.Q_u_traj + self.E
        solution = self.mng_traj.call(x_0)
        u0=solution[u'solution']
        g = 9.82#9.80
        Ax = 0.1#0.01
        Ay = 0.1#0.01
        Az = 0.2#0.01
        # gravity = 9.80 ?????
        tau_roll = 0.5#0.05 # Time constant
        tau_pitch = 0.5#0.05 # Time constant
        K_roll = 1#0.15  # Roll angle gain
        K_pitch = 1#0.15 # Pitch angle gain
        X = np.array([start[0:3]])
        x = start[0]
        y = start[1]
        z = start[2]
        dX = np.array([start[3:6]])
        dx = start[3]
        dy = start[4]
        dz = start[5]
        roll = 0
        pitch = 0
        dt = 10/20
        N = 20
        
        for i in range(1,N):

            #print("LOOP")
            #print(i)


            #print("dX")
            #print(dX)

            resp1 = u0[(i-1)*3:(i-1)*3+3]
            #print("U")
            #print(u0)
            ddx     = -Ax * dX[-1,0]    + g * pitch
            ddy     = -Ay * dX[-1,1]    - g * roll 
            ddz     = -Az * dX[-1,2]    + resp1[2] - g
            dphi    = (-1/tau_roll)*roll   + K_roll/tau_roll * resp1[0]
            dtheta  = (-1/tau_pitch)*pitch  + K_pitch/tau_pitch * resp1[1]
            
            x = X[-1,0] + dX[-1,0]*dt
            y = X[-1,1] + dX[-1,1]*dt
            z = X[-1,2] + dX[-1,2]*dt
            dx = dx + ddx*dt
            dy = dy + ddy*dt
            dz = dz + ddz*dt
            print(dz)
            roll = roll + dphi*dt
            pitch = pitch + dtheta*dt
            
            X = np.append(X, np.array([[x, y, z]]), axis=0)
            dX = np.append(dX, np.array([[dx, dy, dz]]), axis=0)
        self.traj_pos = X
        self.traj_vel = dX 
        self.traj_counter = 0
        self.traj_index = 0
        self.show_path(N, self.traj_pos)
        #print("GENERATERD")
    def show_path(self, N, path):
        self.path_msg = Path()
        self.path_msg.header.frame_id = "world"
        self.path_msg.header.stamp = rospy.Time.now()
        for i in range(N):
            pose = PoseStamped()
            pose.pose.position.x = path[i,0]
            pose.pose.position.y = path[i,1]
            pose.pose.position.z = path[i,2]
            self.path_msg.poses.append(pose)
        self.path_pub.publish(self.path_msg)

    def show_path_uav(self):
        self.uav_path_msg.header.frame_id = "world"
        self.uav_path_msg.header.stamp = rospy.Time.now()
        pose = PoseStamped()
        pose.pose.position.x = self.x_states[0]
        pose.pose.position.y = self.x_states[1]
        pose.pose.position.z = self.x_states[2]
        self.uav_path_msg.poses.append(pose)
        self.uav_path_pub.publish(self.uav_path_msg)
if __name__ == '__main__':
    try:
        multirotor = UAV()
        multirotor.run()
    except rospy.ROSInterruptException:
        pass
        






