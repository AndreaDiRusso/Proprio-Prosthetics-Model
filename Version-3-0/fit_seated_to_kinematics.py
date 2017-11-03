#!/usr/bin/env python3
"""
Example for how to modifying the MuJoCo qpos during execution.
"""

import os, pdb, argparse, pickle
from mujoco_py import load_model_from_xml, MjSim, MjViewerBasic, MjViewer
from lmfit import minimize, Minimizer, Parameters, Parameter, report_fit
from helper_functions import *
from inverse_kinematics import *
import math
import quaternion as quat
import numpy as np

curfilePath = os.path.abspath(__file__)
#print(curfilePath)
curDir = os.path.abspath(os.path.join(curfilePath,os.pardir)) # this will return current directory in which python file resides.
#print(curDir)
parentDir = os.path.abspath(os.path.join(curDir,os.pardir)) # this will return parent directory.
#print(parentDir)

parser = argparse.ArgumentParser()
parser.add_argument('--kinematicsFile', default = 'W:/ENG_Neuromotion_Shared/group/Proprioprosthetics/Data/201709261100-Proprio/T_1.txt')
parser.add_argument('--startTime', default = '27.760')
parser.add_argument('--stopTime', default = '49.960')
parser.add_argument('--showViewer', dest='showViewer', action='store_true')
parser.add_argument('--reIndex', dest='reIndex', type = tuple, nargs = 1)
parser.set_defaults(showViewer = False)
parser.set_defaults(reIndex = None)
args = parser.parse_args()

kinematicsFile = args.kinematicsFile
startTime = float(args.startTime)
stopTime = float(args.stopTime)
showViewer = args.showViewer
reIndex = args.reIndex
showContactForces = True

print(reIndex)

resourcesDir = curDir + '/Resources/Murdoc'

templateFilePath = curDir + '/murdoc_template_static_seat.xml'
fcsvFilePath = resourcesDir + '/Aligned-To-Pelvis/Fiducials.fcsv'

specification = fcsv_to_spec(fcsvFilePath)
modelXML = populate_model(templateFilePath, specification,
    resourcesDir, meshScale = 1e-3, showTendons = True)

alignModel= load_model_from_xml(modelXML)
simulation = MjSim(alignModel)

#viewer = MjViewerBasic(simulation) if showViewer else None
#TODO: make flag for enabling and disabling contact force rendering
if showContactForces and showViewer:
    viewer = MjViewer(simulation) if showViewer else None
    viewer.vopt.flags[10] = viewer.vopt.flags[11] = not viewer.vopt.flags[10]
else:
    viewer = None

whichSide = 'Right'
sitesToFit = ['MT_' + whichSide, 'M_' + whichSide, 'C_' + whichSide, 'GT_' + whichSide, 'K_' + whichSide]

jointsToFit = {
    'World:xt':[0.06,-10, 10],
    'World:yt':[0.02,-10, 10],
    'World:zt':[-0.001,-10, 10],
    'World:x':[0,math.radians(-180),math.radians(180)],
    'World:y':[0,math.radians(-180),math.radians(180)],
    'World:z':[1.72,math.radians(-180),math.radians(180)],
    'Hip_' + whichSide + ':x':[-1,math.radians(-60),math.radians(120)],
    'Hip_' + whichSide + ':y':[0,math.radians(-30),math.radians(30)],
    'Hip_' + whichSide + ':z':[0,math.radians(-15),math.radians(15)],
    'Knee_' + whichSide + ':x':[-1.57,math.radians(0),math.radians(120)],
    'Ankle_' + whichSide + ':x':[1.58,math.radians(-90),math.radians(30)],
    'Ankle_' + whichSide + ':y':[0.1,math.radians(-60),math.radians(60)],
    } if whichSide == 'Right' else {
        'World:xt':[0.06,-10, 10],
        'World:yt':[0.02,-10, 10],
        'World:zt':[-0.001,-10, 10],
        'World:x':[1.73,math.radians(-180),math.radians(180)],
        'World:y':[-0.56,math.radians(-180),math.radians(180)],
        'World:z':[1.72,math.radians(-180),math.radians(180)],
        'Hip_' + whichSide + ':x':[-1,math.radians(-120),math.radians(60)],
        'Hip_' + whichSide + ':y':[0,math.radians(-30),math.radians(30)],
        'Hip_' + whichSide + ':z':[0.05,math.radians(-15),math.radians(15)],
        'Knee_' + whichSide + ':x':[-1.57,math.radians(-120),math.radians(0)],
        'Ankle_' + whichSide + ':x':[1.58,math.radians(-30),math.radians(90)],
        'Ankle_' + whichSide + ':y':[0.1,math.radians(-60),math.radians(60)],
        }

referenceJoint = 'C_' + whichSide
solver = IKFit(simulation, sitesToFit, jointsToFit,
    alignTo = referenceJoint, mjViewer = viewer, method = 'nelder',
    simulationType = 'forward')

#Get kinematics
timeSelection = [startTime, stopTime]
kinematics = get_kinematics(kinematicsFile,
    selectHeaders = sitesToFit,
    selectTime = timeSelection, reIndex = reIndex)

#provide initial fit
kinIterator = kinematics.iterrows()
t, kinSeries = next(kinIterator)
stats = solver.fit(t, kinSeries)

#set initial guess to result of initial fit
initialResults = params_to_dict(stats.params)
for key, value in jointsToFit.items():
    jointsToFit[key][0] = initialResults[key]

# second model does not contain world joints:
#TODO: make this less kludgy
meshScale = 0.9e-3

origGravity = np.array([0,0,0,-9.78])

xAxis = np.array([0,1,0,0])
xAxisAngle = (jointsToFit['World:x'][0]*0.5) * xAxis/np.linalg.norm(xAxis)
xQLog = quat.quaternion(*xAxisAngle)
xQ = np.exp(xQLog)

yAxis = np.array([0,0,1,0])
yAxisAngle = (jointsToFit['World:y'][0]*0.5) * yAxis/np.linalg.norm(yAxis)
yQLog = quat.quaternion(*yAxisAngle)
yQ = np.exp(yQLog)

zAxis = np.array([0,0,0,1])
zAxisAngle = (jointsToFit['World:z'][0]*0.5) * zAxis/np.linalg.norm(zAxis)
zQLog = quat.quaternion(*zAxisAngle)
zQ = np.exp(zQLog)

vec = quat.quaternion(*origGravity)
q = xQ * yQ * zQ
rotatedGravity = q * vec * np.conjugate(q)

worldCoords = pd.DataFrame({
    '1' : {
        'label' : 'World',
        'x': math.degrees(jointsToFit['World:x'][0])/meshScale,
        'y': math.degrees(jointsToFit['World:y'][0])/meshScale,
        'z': math.degrees(jointsToFit['World:z'][0])/meshScale
        },
    '2': {
        'label' : 'World_t',
        'x': jointsToFit['World:xt'][0]/meshScale,
        'y': jointsToFit['World:yt'][0]/meshScale,
        'z': jointsToFit['World:zt'][0]/meshScale
        },
    '3': {
        'label' : 'gravity',
        'x': (rotatedGravity.x)/meshScale,
        'y': (rotatedGravity.y)/meshScale,
        'z': (rotatedGravity.z)/meshScale
        }
    }).transpose()
#TODO: populate specification with world transformation
specification = specification.append(worldCoords, ignore_index = True)
secondTemplateFilePath = '/'.join(templateFilePath.split('/')[:-1]) +\
    '/murdoc_seated_template.xml'
modelXML2 = populate_model(secondTemplateFilePath, specification, resourcesDir,
    showTendons = True)

#model that will be varied for optimization fitting
optModel = load_model_from_xml(modelXML2)
optSim = MjSim(optModel)
# model that will be posed for inverse dynamics
poseModel = load_model_from_xml(modelXML2)
poseSim = MjSim(poseModel)

#viewer = MjViewerBasic(simulation) if showViewer else None
#TODO: make flag for enabling and disabling contact force rendering
if showContactForces and showViewer:
    viewer2 = MjViewer(optSim) if showViewer else None
    viewer2.vopt.flags[10] = viewer2.vopt.flags[11] = not viewer2.vopt.flags[10]
else:
    viewer2 = None

skip = [
    'World:xt',
    'World:yt',
    'World:zt',
    'World:x',
    'World:y',
    'World:z',
    ]

for joint in skip:
    jointsToFit.pop(joint)

solver2 = IKFit(optSim, sitesToFit, jointsToFit,
    alignTo = referenceJoint, mjViewer = viewer2,
    method = 'nelder', simulationType = 'forward')
solver2.jointsParam = dict_to_params(jointsToFit)

modelKin = pd.DataFrame(index = kinematics.index, columns = kinematics.columns)
modelQpos = pd.DataFrame(index = kinematics.index, columns = params_to_series(solver2.jointsParam).index)
alignedKin = pd.DataFrame(index = kinematics.index, columns = kinematics.columns)

solver2.nelderTol = 1e-4
statistics = {
    'nfev': [],
    'redchi': []
    }

for i in range(int(2e3)):
    #settle model
    optSim.step()
    if viewer2:
        viewer2.render()

printing = True
for t, kinSeries in kinematics.iterrows():
    stats = solver2.fit(t, kinSeries)

    if printing:
        try:
            print("SSQ: ")
            print(np.sum(stats.residual**2))
            print(stats.message)
            report_fit(stats)

            statistics['nfev'].append(stats.nfev)
            statistics['redchi'].append(stats.redchi)
        except:
            pass

    solver2.jointsParam = stats.params

    modelKin.loc[t, :] = get_site_pos(kinSeries, optSim)
    modelQpos.loc[t, :] = params_to_series(stats.params)
    alignedKin.loc[t, :] = alignToModel(optSim, kinSeries, referenceJoint)

results = {
    'site_pos' : modelKin,
    'orig_site_pos': alignedKin,
    'qpos' : modelQpos
}

saveName = kinematicsFile.split('.')[0] + "_kinematics.pickle"
with open(saveName, 'wb') as f:
    pickle.dump(results, f)

print('Finished with %4.2f average function calls' % np.mean(statistics['nfev']))
print('Finished with %4.6f average reduced chisquare' % np.mean(statistics['redchi']))
