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
from collections import deque
from scipy import signal
import time

curfilePath = os.path.abspath(__file__)
#print(curfilePath)
curDir = os.path.abspath(os.path.join(curfilePath,os.pardir)) # this will return current directory in which python file resides.
#print(curDir)

parser = argparse.ArgumentParser()
parser.add_argument('--kinematicsFile', default = 'W:/ENG_Neuromotion_Shared/group/Proprioprosthetics/Data/201709261100-Proprio/T_1_filtered_kinematics.pickle')
parser.add_argument('--modelFile', default = 'murdoc_template_toes_treadmill.xml')
parser.add_argument('--meshScale', default = '1.1e-3')
parser.add_argument('--slowDown', default = '0')
args = parser.parse_args()

kinematicsFile = args.kinematicsFile
modelFile = args.modelFile
meshScale = float(args.meshScale)
slowDown = float(args.slowDown)

resourcesDir = curDir + '/Resources/Murdoc'
templateFilePath = curDir + '/' + modelFile

fcsvFilePath = resourcesDir + '/Mobile Foot/Fiducials.fcsv'

specification = fcsv_to_spec(fcsvFilePath)

with open(kinematicsFile, 'rb') as f:
    kinematics = pickle.load(f)

#meshScale = kinematics['meshScale']
extraCoords = {}
if modelFile == 'murdoc_template_seat.xml':

    t0 = kinematics['qpos'].index[0]
    worldQ = quat.from_euler_angles(
        kinematics['qpos'].loc[t0, 'World:xq'],
        kinematics['qpos'].loc[t0, 'World:yq'],
        kinematics['qpos'].loc[t0, 'World:zq']
        )
    extraCoords = {
                'World:xt': kinematics['qpos'].loc[t0, 'World:xt'],
                'World:yt': kinematics['qpos'].loc[t0, 'World:yt'],
                'World:zt': kinematics['qpos'].loc[t0, 'World:zt'],
                'World:wq': worldQ.w,
                'World:xq': worldQ.x,
                'World:yq': worldQ.y,
                'World:zq': worldQ.z }

if modelFile == 'murdoc_template_toes_treadmill.xml':

    t0 = kinematics['qpos'].index[0]
    worldQ = quat.from_euler_angles(
        kinematics['qpos'].loc[t0, 'World:xq'],
        kinematics['qpos'].loc[t0, 'World:yq'],
        kinematics['qpos'].loc[t0, 'World:zq']
        )
    extraCoords = {
                'World:xt': kinematics['qpos'].loc[t0, 'World:xt'],
                'World:yt': kinematics['qpos'].loc[t0, 'World:yt'],
                'World:zt': kinematics['qpos'].loc[t0, 'World:zt'],
                'World:wq': worldQ.w,
                'World:xq': worldQ.x,
                'World:yq': worldQ.y,
                'World:zq': worldQ.z,
                'Floor:x' : 0,
                'Floor:y' : 0,
                'Floor:z' : -0.38,
                'minT12Height': 0.5
                }

modelXML = populate_model(templateFilePath, specification, extraCoords, resourcesDir,
    meshScale = meshScale, showTendons = True)

model = load_model_from_xml(modelXML)
simulation = MjSim(model)

simulation = pose_to_key(simulation, 1)
dt = simulation.model.opt.timestep

viewer = MjViewer(simulation)
viewer.vopt.flags[10] = viewer.vopt.flags[11] = not viewer.vopt.flags[10]
#viewer._hide_overlay = True
viewer._render_every_frame = True

#get resting lengths
nJoints = simulation.model.njnt
allJoints = [simulation.model.joint_id2name(i) for i in range(nJoints)]

modelQAcc = {time : {joint : [] for joint in allJoints} for time in kinematics['site_pos'].index}
modelQFrcInverse = {time : {joint : [] for joint in allJoints} for time in kinematics['site_pos'].index}
modelQfrcConstraint = {time : {joint : [] for joint in allJoints} for time in kinematics['site_pos'].index}

jointMask = dict.fromkeys(allJoints)
for jointName in allJoints:
    jointId = simulation.model.get_joint_qvel_addr(jointName)
    jointMask[jointName] = jointId
    if type(jointMask[jointName]) == tuple:
        jointMask[jointName] = list(range(jointMask[jointName][0], jointMask[jointName][1]))
    #print(jointName)

#Do an initial pose to avoid discontinuities:
jointDict = series_to_dict( kinematics['qpos'].iloc[0, :])

simulation = pose_model(simulation,jointDict, method = 'forward')

bufferSize = 3

dummyTime = deque(np.tile([simulation.data.time], (bufferSize,1)))
dummyQVel = deque(np.tile(simulation.data.qvel, (bufferSize,1)))
dummyQPos = deque(np.tile(simulation.data.qpos, (bufferSize,1)))

mybuffer = {
    'time': dummyTime,
    'qvel': dummyQVel,
    'qpos': dummyQPos
    }

allActiveContacts = {}

counter = 0
for t, kinSeries in kinematics['site_pos'].iterrows():
    time.sleep(slowDown)

    #constraintsSummary = constraints_summary(simulation)
    #if constraintsSummary is not None:
    #    print(t)
    #    print(constraintsSummary)

    # calculate qAcc and pass to pose model
    qPosMat = np.asarray(mybuffer['qpos'])

    tempQVel = copy.deepcopy(mybuffer['qvel'].popleft())
    functions.mj_differentiatePos(simulation.model, tempQVel,
        dt, qPosMat[-2, :], qPosMat[-1, :])
    mybuffer['qvel'].append(tempQVel)
    #print(tempQVel)

    qVelMat = np.asarray(mybuffer['qvel'])
    qAcc = np.gradient(qVelMat, dt, axis = 0)[-1]

    tempQFrc = copy.deepcopy(simulation.data.qfrc_inverse)
    tempQfrcConstraint = copy.deepcopy(simulation.data.qfrc_constraint)

    for jointName in allJoints:
        modelQAcc[t][jointName] = qAcc[jointMask[jointName]] if counter > 3 else 0
        modelQFrcInverse[t][jointName] = tempQFrc[jointMask[jointName]] if counter > 3 else 0
        modelQfrcConstraint[t][jointName] = tempQfrcConstraint[jointMask[jointName]] if counter > 3 else 0

    activeContacts = contact_summary(simulation, zeroPad = True)

    if activeContacts:
        allActiveContacts.update({t: activeContacts})


    jointDict = series_to_dict( kinematics['qpos'].loc[t, :])

    """
        skip = ['World:' + i for i in ['xt', 'yt', 'zt']]
        for name in skip:
            jointDict.pop(name)
            """

    simulation = pose_model(simulation,jointDict,
        qAcc = qAcc, qVel = qVelMat[-1], method = 'inverse')

    _ = mybuffer['qpos'].popleft()
    newValue = copy.deepcopy(simulation.data.qpos)
    mybuffer['qpos'].append(newValue)

    #simulation = pose_model(simulation,jointDict, method = 'step')
    simulation.forward()
    viewer.render()
    counter = counter + 1

results = {
    'qacc' : modelQAcc,
    'qfrc_inverse' : modelQFrcInverse,
    'qfrc_constraint': modelQfrcConstraint,
    'active_contacts': allActiveContacts
    }

saveName = kinematicsFile.split('_kinematics')[0] + "_kinetics.pickle"

with open(saveName, 'wb') as f:
    pickle.dump(results, f)
