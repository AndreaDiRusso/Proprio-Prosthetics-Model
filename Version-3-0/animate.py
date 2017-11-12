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
parser.add_argument('--kinematicsFile', default = 'W:/ENG_Neuromotion_Shared/group/Proprioprosthetics/Data/201709261100-Proprio/T_1_kinematics.pickle')
parser.add_argument('--outputFile', dest='outputFile', action='store_true')
parser.add_argument('--outputRawFile', dest='outputRawFile', action='store_true')
parser.set_defaults(outputFile = False)
parser.set_defaults(outputRawFile = False)
parser.add_argument('--cameraId', default = None)
parser.add_argument('--modelFile', default = 'murdoc_template_toes_treadmill.xml')
parser.add_argument('--meshScale', default = '1.1e-3')

args = parser.parse_args()

kinematicsFile = args.kinematicsFile
modelFile = args.modelFile
outputFile = args.outputFile
outputRawFile = args.outputRawFile
meshScale = float(args.meshScale)
cameraId = args.cameraId

resourcesDir = curDir + '/Resources/Murdoc'
templateFilePath = curDir + '/' + modelFile

fcsvFilePath = resourcesDir + '/Mobile Foot/Fiducials.fcsv'

specification = fcsv_to_spec(fcsvFilePath)
modelXML = populate_model(templateFilePath, specification, resourcesDir = resourcesDir,
    meshScale = meshScale, showTendons = True)

model = load_model_from_xml(modelXML)

simulation = MjSim(model)

viewer = MjViewer(simulation)
viewer._render_every_frame = True



if cameraId is not None:
    viewer.cam.fixedcamid += int(cameraId)
    viewer.cam.type = const.CAMERA_FIXED
#viewer.vopt.flags[10] = viewer.vopt.flags[11] = not viewer.vopt.flags[10]
#get resting lengths
with open(kinematicsFile, 'rb') as f:
    kinematics = pickle.load(f)

nJoints = simulation.model.njnt
allJoints = [simulation.model.joint_id2name(i) for i in range(nJoints)]

simulation = pose_to_key(simulation, 0)

for t, kinSeries in kinematics['orig_site_pos'].iterrows():

    jointDict = series_to_dict( kinematics['qpos'].loc[t, :])
    pose_model(simulation, jointDict)

    if outputFile or outputRawFile:
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

        img = simulation.render(*resolution, camera_name = 'Seat_Cam')
        img = cv2.flip(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), 0)

        if outputFile and 'output' not in locals():
            # Define the codec and create VideoWriter object

            outputFileName = kinematicsFile.split('_kinematics')[0] + '_video.avi'
            fourcc = cv2.VideoWriter_fourcc(*'MPEG')
            output = cv2.VideoWriter(outputFileName, fourcc, int(1 / viewer._time_per_render), (img.shape[1], img.shape[0]))

        if outputRawFile and 'outputRaw' not in locals():
            # Define the codec and create VideoWriter object
            outputRawFileName = kinematicsFile.split('_kinematics')[0] + '_raw_video.avi'
            fourcc = cv2.VideoWriter_fourcc(*'DIB ')
            outputRaw = cv2.VideoWriter(outputRawFileName, fourcc, int(1 / viewer._time_per_render), (img.shape[1], img.shape[0]))

        #pdb.set_trace()
        # write the frame
        cv2.imshow('frame',img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
          break

        if outputFile:
            output.write(img)
        if outputRawFile:
            outputRaw.write(img)
    else:
        viewer._hide_overlay = True
        viewer._render_every_frame = True
        render_targets(viewer, kinematics['orig_site_pos'].loc[t, :])
        viewer.render()

if outputFile:
    output.release()
    cv2.destroyAllWindows()
if outputRawFile:
    outputRaw.release()
    cv2.destroyAllWindows()
