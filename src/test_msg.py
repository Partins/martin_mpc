#!/usr/bin/env python3


import rospy
import actionlib

import martin_mpc.msg


def traj_client():

    client = actionlib.SimpleActionClient('traj_gen', martin_mpc.msg.generate_trajectoryAction)

    client.wait_for_server()
    rospy.logwarn("Action oNLINE")
    goal = martin_mpc.msg.generate_trajectoryGoal()
    
    goal.goal_pos = [0.0, 0.0, 2.0]
    goal.goal_vel = [0.0, 0.0, 0.0]
    goal.rate = 20
    goal.time = 100
    tic = rospy.get_time()
    client.send_goal(goal)
    client.wait_for_result()
    toc = rospy.get_time()
    rospy.logwarn(toc-tic)
if __name__ == '__main__':
    try:
        rospy.init_node('trajgen_client')
        result = traj_client()
        print("Result: ")
    except rospy.ROSInterruptException:
        print("program interrupted before completion", file=sys.stderr)