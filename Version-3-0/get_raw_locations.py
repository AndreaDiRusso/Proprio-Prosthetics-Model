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

curfilePath = os.path.abspath(__file__)
#print(curfilePath)
curDir = os.path.abspath(os.path.join(curfilePath,os.pardir)) # this will return current directory in which python file resides.
#print(curDir)

parser = argparse.ArgumentParser()
parser.add_argument('--kinematicsFile', default = 'W:/ENG_Neuromotion_Shared/group/Proprioprosthetics/Data/201709261100-Proprio/T_1.txt')
parser.add_argument('--meshScale', default = '1.1e-3')
parser.add_argument('--startTime', default = '27.760')
parser.add_argument('--stopTime', default = '49.960')
parser.add_argument('--whichSide', default = 'Left')
args = parser.parse_args()

kinematicsFile = args.kinematicsFile
meshScale = float(args.meshScale)
startTime = float(args.startTime)
stopTime = float(args.stopTime)
whichSide = args.whichSide

resourcesDir = curDir + '/Resources/Murdoc'

templateFilePath = curDir + '/murdoc_template_toes_floating.xml'
fcsvFilePath = resourcesDir + '/Mobile Foot/Fiducials.fcsv'

specification = fcsv_to_spec(fcsvFilePath)
modelXML = populate_model(templateFilePath, specification, extraLocations = {},
    resourcesDir = resourcesDir, meshScale = meshScale, showTendons = True)

model = load_model_from_xml(modelXML)
simulation = MjSim(model)

sitesToFit = ['C_' + whichSide, 'MT_' + whichSide, 'T_' + whichSide]

#Get kinematics
kinematics = get_kinematics(kinematicsFile, selectHeaders = sitesToFit)

#provide initial fit
kinIterator = kinematics.iterrows()
t, kinSeries = next(kinIterator)

referenceJoint = 'C_' + whichSide
referenceSeries =\
    copy.deepcopy(get_site_pos(kinSeries, simulation).loc[referenceJoint, :])\
    -copy.deepcopy(kinSeries.loc[referenceJoint, :])

alignedKin = pd.DataFrame(index = kinematics.index, columns = kinematics.columns)
for t, kinSeries in kinematics.iterrows():
    alignedKin.loc[t, :] = alignToModel(simulation, kinSeries, referenceSeries)

print('Kinematics:')
print(alignedKin.min(axis = 0))
