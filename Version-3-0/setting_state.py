#!/usr/bin/env python3
"""
Example for how to modifying the MuJoCo qpos during execution.
"""

import os, pdb, argparse
from mujoco_py import load_model_from_xml, MjSim, MjViewer
from helper_functions import *

curfilePath = os.path.abspath(__file__)
#print(curfilePath)
curDir = os.path.abspath(os.path.join(curfilePath,os.pardir)) # this will return current directory in which python file resides.
#print(curDir)
parentDir = os.path.abspath(os.path.join(curDir,os.pardir)) # this will return parent directory.
#print(parentDir)

parser = argparse.ArgumentParser()
parser.add_argument('--kinematicsFile', default = 'W:/ENG_Neuromotion_Shared/group/Proprioprosthetics/Training/T_1.txt')

args = parser.parse_args()

kinematicsFile = args.kinematicsFile

with open(curDir + '/murdoc.xml', 'r') as f:
    MODEL_XML = f.read()

MODEL_XML = MODEL_XML.replace('$MURDOC_DIR$', curDir + '/Resources/Murdoc')

model = load_model_from_xml(MODEL_XML)
sim = MjSim(model)
#viewer = MjViewer(sim)

#Get kinematics
kin = get_kinematics(kinematicsFile,
    selectHeaders = ['MT_Left', 'M_Left', 'C_Left', 'GT_Left', 'K_Left'],
    selectTime = [26, 50])

states = [{'box:x': +0.8, 'box:y': +0.8},
          {'box:x': -0.8, 'box:y': +0.8},
          {'box:x': -0.8, 'box:y': -0.8},
          {'box:x': +0.8, 'box:y': -0.8},
          {'box:x': +0.0, 'box:y': +0.0}]

# MjModel.joint_name2id returns the index of a joint in
# MjData.qpos.

x_joint_i = sim.model.get_joint_qpos_addr("left_knee")
#y_joint_i = sim.model.get_joint_qpos_addr("box:y")

#print_box_xpos(sim)

for idx, row in kin.iterrows():
    sim_state = sim.get_state()
    #sim_state.qpos[x_joint_i] = state["box:x"]
    #sim_state.qpos[y_joint_i] = state["box:y"]
    #sim.set_state(sim_state)
    sim.forward()
    print("updated state to", state)
    #print_box_xpos(sim)
    #viewer.render()
