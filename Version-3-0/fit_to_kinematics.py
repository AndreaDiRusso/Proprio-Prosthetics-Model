#!/usr/bin/env python3
"""
Example for how to modifying the MuJoCo qpos during execution.
"""

import os, pdb, argparse, pickle
from mujoco_py import load_model_from_xml, MjSim, MjViewerBasic
from lmfit import minimize, Minimizer, Parameters, Parameter, report_fit
from helper_functions import *
from inverse_kinematics import *
import math

curfilePath = os.path.abspath(__file__)
#print(curfilePath)
curDir = os.path.abspath(os.path.join(curfilePath,os.pardir)) # this will return current directory in which python file resides.
#print(curDir)
parentDir = os.path.abspath(os.path.join(curDir,os.pardir)) # this will return parent directory.
#print(parentDir)

parser = argparse.ArgumentParser()
parser.add_argument('--kinematicsFile', default = 'W:/ENG_Neuromotion_Shared/group/Proprioprosthetics/Training/T_1.txt')
parser.add_argument('--startTime', default = '26')
parser.add_argument('--stopTime', default = '63')
parser.add_argument('--showViewer', dest='showViewer', action='store_true')
parser.set_defaults(showViewer = False)
args = parser.parse_args()

kinematicsFile = args.kinematicsFile
startTime = float(args.startTime)
stopTime = float(args.stopTime)
showViewer = args.showViewer

resourcesDir = curDir + '/Resources/Murdoc'

templateFilePath = curDir + '/murdoc_template.xml'
fcsvFilePath = resourcesDir + '/Fiducials.fcsv'

specification = fcsv_to_spec(fcsvFilePath)
modelXML = populate_model(templateFilePath, specification, resourcesDir, showTendons = True)

model = load_model_from_xml(modelXML)
simulation = MjSim(model)

viewer = MjViewerBasic(simulation) if showViewer else None

sitesToFit = ['MT_Left', 'M_Left', 'C_Left', 'GT_Left', 'K_Left']

jointsToFit = {
    'World:xt':[.1,-1, 1],
    'World:yt':[0.08,-1, 1],
    'World:zt':[0.07,-1, 1],
    'World:x':[0.15,math.radians(-180),math.radians(180)],
    'World:y':[0.3,math.radians(-180),math.radians(180)],
    'World:z':[-1,math.radians(-180),math.radians(180)],
    'Hip_Left:x':[-0.5,math.radians(-60),math.radians(90)],
    'Hip_Left:y':[0.05,math.radians(-15),math.radians(15)],
    'Hip_Left:z':[0,math.radians(-15),math.radians(15)],
    'Knee_Left:x':[0.4,math.radians(-90),math.radians(70)],
    'Ankle_Left:x':[-0.5,math.radians(-60),math.radians(120)],
    'Ankle_Left:y':[0.1,math.radians(-60),math.radians(60)],
    }

referenceJoint = 'C_Left'
solver = IKFit(simulation, sitesToFit, jointsToFit,
    alignTo = referenceJoint, mjViewer = viewer, method = 'nelder')
#Get kinematics
kinematics = get_kinematics(kinematicsFile,
    selectHeaders = sitesToFit,
    selectTime = [startTime, stopTime], reIndex = None)

#provide initial fit
kinIterator = kinematics.iterrows()
t, kinSeries = next(kinIterator)
stats = solver.fit(t, kinSeries)

# skip world coords after initial fit
initialResults = params_to_dict(stats.params)
for key, value in jointsToFit.items():
    jointsToFit[key][0] = initialResults[key]

skip = [
    'World:xt',
    'World:yt',
    'World:zt',
    'World:x',
    'World:y',
    'World:z',
    ]

solver.jointsParam = dict_to_params(jointsToFit, skip)

modelKin = pd.DataFrame(index = kinematics.index, columns = kinematics.columns)
modelQpos = pd.DataFrame(index = kinematics.index, columns = params_to_series(stats.params).index)
alignedKin = pd.DataFrame(index = kinematics.index, columns = kinematics.columns)

for t, kinSeries in kinematics.iterrows():
    solver.nelderTol = 2e-3
    stats = solver.fit(t, kinSeries)

    print("SSQ: ")
    print(np.sum(stats.residual**2))
    print(stats.message)
    report_fit(stats)

    solver.jointsParam = stats.params
    modelKin.loc[t, :] = get_site_pos(kinSeries, simulation)
    modelQpos.loc[t, :] = params_to_series(stats.params)
    alignedKin.loc[t, :] = alignToModel(simulation, kinSeries, referenceJoint)


results = {
    'site_pos' : modelKin,
    'orig_site_pos': alignedKin,
    'qpos' : modelQpos
}

saveName = kinematicsFile.split('.')[0] + "_model.pickle"
with open(saveName, 'wb') as f:
    pickle.dump(results, f)
