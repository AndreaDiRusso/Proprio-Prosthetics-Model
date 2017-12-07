#!/usr/bin/env python3
"""
Example for how to modifying the MuJoCo qpos during execution.
"""

import os, argparse, pickle, copy
import mujoco_py
import glfw
from mujoco_py import load_model_from_xml, MjSim, MjViewer
from mujoco_py.utils import rec_copy, rec_assign
from helper_functions import *
import matplotlib.pyplot as plt

curfilePath = os.path.abspath(__file__)
#print(curfilePath)
curDir = os.path.abspath(os.path.join(curfilePath,os.pardir)) # this will return current directory in which python file resides.
#print(curDir)
parentDir = os.path.abspath(os.path.join(curDir,os.pardir)) # this will return parent directory.
#print(parentDir)

parser = argparse.ArgumentParser()
parser.add_argument('--kinematicsFile', default = 'W:/ENG_Neuromotion_Shared/group/Proprioprosthetics/Data/201709261100-Proprio/T_1_kinematics.pickle')
parser.add_argument('--modelFile', default = 'murdoc_template_toes_treadmill.xml')
parser.add_argument('--meshScale', default = '1.1e-3')

args = parser.parse_args()

kinematicsFile = args.kinematicsFile
modelFile = args.modelFile
meshScale = float(args.meshScale)

resourcesDir = curDir + '/Resources/Murdoc'
templateFilePath = curDir + '/' + modelFile

fcsvFilePath = resourcesDir + '/Mobile Foot/Fiducials.fcsv'

specification = fcsv_to_spec(fcsvFilePath)
modelXML = populate_model(templateFilePath, specification, resourcesDir = resourcesDir,
    meshScale = meshScale, showTendons = True)

model = load_model_from_xml(modelXML)

simulation = MjSim(model)

viewer = MjViewer(simulation)

with open(kinematicsFile, 'rb') as f:
    kinematics = pickle.load(f)
#viewer.vopt.flags[10] = viewer.vopt.flags[11] = not viewer.vopt.flags[10]

sitesToShow = ['Sole_Tip_Right']
colIdx = pd.MultiIndex.from_product([sitesToShow, ['x', 'y', 'z']], names = ['joint', 'coordinate'])
modelKin = pd.DataFrame(index = kinematics['site_pos'].index, columns = colIdx)
dontPose = ['World:xq', 'World:yq', 'World:zq','World:xt', 'World:yt', 'World:zt' ]
import time
for t, kinSeries in kinematics['site_pos'].iterrows():

    jointDict = series_to_dict( kinematics['qpos'].loc[t, :])
    #for jointName in dontPose:
    #    jointDict.pop(jointName)
    pose_model(simulation, jointDict)
    modelKin.loc[t, :] = get_site_pos(modelKin.loc[t, :], simulation)

    viewer._hide_overlay = True
    viewer._render_every_frame = True
    #render_targets(viewer, kinematics['orig_site_pos'].loc[t, :])
    viewer.render()

    #time.sleep(0.1)

plt.plot(modelKin.loc[:, (sitesToShow[0], 'z')])
plt.show()
print('Kinematics:')
print(modelKin.min(axis = 0))
