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
curfilePath = os.path.abspath(__file__)
#print(curfilePath)
curDir = os.path.abspath(os.path.join(curfilePath,os.pardir)) # this will return current directory in which python file resides.
#print(curDir)
parentDir = os.path.abspath(os.path.join(curDir,os.pardir)) # this will return parent directory.
#print(parentDir)

parser = argparse.ArgumentParser()
parser.add_argument('--kinematicsFile', default = 'W:/ENG_Neuromotion_Shared/group/Proprioprosthetics/Data/201709261100-Proprio/T_1_kinematics.pickle')
parser.add_argument('--modelFile', default = 'murdoc_gen.xml')

args = parser.parse_args()

kinematicsFile = args.kinematicsFile
modelFile = args.modelFile

resourcesDir = curDir + '/Resources/Murdoc'

with open(curDir + '/' + modelFile, 'r') as f:
    model = load_model_from_xml(f.read())

with open(kinematicsFile, 'rb') as f:
    kinematics = pickle.load(f)

simulation = MjSim(model)
dt = simulation.model.opt.timestep

viewer = MjViewer(simulation)
viewer.vopt.flags[10] = viewer.vopt.flags[11] = not viewer.vopt.flags[10]
#viewer._hide_overlay = True
viewer._render_every_frame = True

#get resting lengths
nJoints = simulation.model.njnt
allJoints = [simulation.model.joint_id2name(i) for i in range(nJoints)]

bufferSize = 3

dummyTime = deque(np.tile([simulation.data.time], (bufferSize,1)))
dummyQVel = deque(np.tile(simulation.data.qvel, (bufferSize,1)))
dummyQPos = deque(np.tile(simulation.data.qpos, (bufferSize,1)))

mybuffer = {
    'time': dummyTime,
    'qvel': dummyQVel,
    'qpos': dummyQPos
    }


modelQAcc = pd.DataFrame(index = kinematics['site_pos'].index,
    columns = kinematics['qpos'].columns)
modelQFrcInverse = pd.DataFrame(index = kinematics['site_pos'].index,
    columns = kinematics['qpos'].columns)
modelQfrcConstraint = pd.DataFrame(index = kinematics['site_pos'].index,
    columns = kinematics['qpos'].columns)

jointMask = dict.fromkeys(kinematics['qpos'].columns)
for jointName in kinematics['qpos'].columns:
    jointId = simulation.model.get_joint_qvel_addr(jointName)
    jointMask[jointName] = jointId

for i in range(int(2e3)):
    #settle model
    simulation.step()
    viewer.render()

allActiveContacts = []
for t, kinSeries in kinematics['site_pos'].iterrows():
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
    for jointName in modelQAcc.columns:
        modelQAcc.loc[t, jointName] = qAcc[jointMask[jointName]]
        modelQFrcInverse.loc[t, (jointName)] = tempQFrc[jointMask[jointName]]
        modelQfrcConstraint.loc[t, (jointName)] = tempQfrcConstraint[jointMask[jointName]]

    activeContacts = contact_summary(simulation)
    allActiveContacts.append(activeContacts)

    simulation = pose_model(simulation, kinematics['qpos'].loc[t, :],
        qAcc = qAcc, qVel = qVelMat[-1], method = 'inverse')

    _ = mybuffer['qpos'].popleft()
    newValue = copy.deepcopy(simulation.data.qpos)
    mybuffer['qpos'].append(newValue)

    #simulation.forward()
    render_targets(viewer, kinematics['orig_site_pos'].loc[t, :])
    viewer.render()

results = {
    'qacc' : modelQAcc,
    'qfrc_inverse' : modelQFrcInverse,
    'qfrc_constraint': modelQfrcConstraint,
    'active_contacts': activeContacts
    }

saveName = kinematicsFile.split('_kinematics')[0] + "_kinetics.pickle"

with open(saveName, 'wb') as f:
    pickle.dump(results, f)
