#!/usr/bin/env python

import rospy
from dynamic_reconfigure.server import Server
from martin_mpc.cfg import mpc_paramsConfig

def callback(config, level):
    #rospy.loginfo("""Reconfigure request: {int_param}, \
    #    {str_param}""".format(**config))
    rospy.logwarn(type(config))
    return config

if __name__ == "__main__":
    rospy.init_node("martin_mpc_param_node")

    srv = Server(mpc_paramsConfig, callback)
    rospy.spin()