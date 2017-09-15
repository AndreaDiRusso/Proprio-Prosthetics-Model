#
#   Hello World server in Python
#   Binds REP socket to tcp://*:5555
#   Expects b"Hello" from client, replies with b"World"
#

import zmq
import math
import time, pdb
import numpy as np
import matplotlib.pyplot as plt
import mujoco2py_pb2 as mj2py

context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://*:5556")

num_poses = 100

poses = {
'right_hip_x': np.linspace(-40, 90, num_poses),
'right_knee': np.linspace(-60, 40, num_poses),
'right_ankle': np.linspace(-60, 30, num_poses)
}
pdb.set_trace()
for t in range(num_poses):
    message = socket.recv()
    in_mj_msg = mj2py.mujoco_msg()
    in_mj_msg.ParseFromString(message)

    out_msg = mj2py.mujoco_msg()
    for joint_name in poses.keys():
        #  Wait for next request from client
        #socket.setsockopt( zmq.RCVTIMEO, 500 )

        joint = out_msg.joint.add()
        joint.name =  joint_name
        joint.qpos = poses[joint_name][t]

    out_msg_str = out_msg.SerializeToString()
    socket.send(out_msg_str)

input("Please click enter to continue")
