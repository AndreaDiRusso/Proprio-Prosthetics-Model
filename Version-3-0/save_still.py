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

curfilePath = os.path.abspath(__file__)
#print(curfilePath)
curDir = os.path.abspath(os.path.join(curfilePath,os.pardir)) # this will return current directory in which python file resides.
#print(curDir)
parentDir = os.path.abspath(os.path.join(curDir,os.pardir)) # this will return parent directory.
#print(parentDir)

parser = argparse.ArgumentParser()
parser.add_argument('--modelKinematicsFile', default = '')
parser.add_argument('--outputFile')
parser.add_argument('--t', default = 'first')
parser.add_argument('--modelFile', default = 'murdoc_gen.xml')

args = parser.parse_args()

modelKinematicsFile = args.modelKinematicsFile
modelFile = args.modelFile
t = float(args.t)
outputFile = args.outputFile if args.outputFile else None

resourcesDir = curDir + '/Resources/Murdoc'

with open(curDir + '/' + modelFile, 'r') as f:
    model = load_model_from_xml(f.read())

with open(modelKinematicsFile, 'rb') as f:
    kinematics = pickle.load(f)

simulation = MjSim(model)

viewer = MjViewer(simulation)

#set to a keyframe
nJoints = simulation.model.key_qpos.shape[1]
allJoints = [simulation.model.joint_id2name(i) for i in range(nJoints)]
keyPos = pd.Series({jointName: simulation.model.key_qpos[1][i] for i, jointName in enumerate(allJoints)})

pose_model(simulation, keyPos)

kinSeries = kinematics['site_pos'].loc[t, :]

pose_model(simulation, kinematics['qpos'].loc[t, :])

if outputFile is not None:
    import numpy as np
    import cv2

    markers = series_to_markers(kinSeries)
    # Reads pixels with markers and overlay from the same camera as screen.
    resolution = glfw.get_framebuffer_size(
        simulation._render_context_window.window)
    if simulation._render_context_offscreen is None:
        simulation.render(*resolution)
    offscreen_ctx = simulation._render_context_offscreen
    offscreen_ctx._markers[:] = markers[:]

    img = simulation.render(*resolution, camera_name = 'Sagittal_Left')
    img = cv2.flip(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), 0)

    # write the frame
    cv2.imshow('frame',img)
    cv2.waitKey(0)
    cv2.imwrite(outputFile, img)

cv2.destroyAllWindows()
