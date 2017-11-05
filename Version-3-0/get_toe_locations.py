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
parser.add_argument('--whichSide', default = 'Left')
parser.add_argument('--showViewer', dest='showViewer', action='store_true')
parser.set_defaults(showViewer = False)
args = parser.parse_args()

kinematicsFile = args.kinematicsFile
meshScale = float(args.meshScale)
whichSide = args.whichSide
showViewer = args.showViewer
showContactForces = True

resourcesDir = curDir + '/Resources/Murdoc'

templateFilePath = curDir + '/murdoc_template_floating.xml'
fcsvFilePath = resourcesDir + '/Aligned-To-Pelvis/Fiducials.fcsv'

specification = fcsv_to_spec(fcsvFilePath)
modelXML = populate_model(templateFilePath, specification, extraLocations = {},
    resourcesDir = resourcesDir, meshScale = meshScale, showTendons = True)

model = load_model_from_xml(modelXML)
simulation = MjSim(model)

#viewer = MjViewerBasic(simulation) if showViewer else None
#TODO: make flag for enabling and disabling contact force rendering
if showContactForces and showViewer:
    viewer = MjViewer(simulation) if showViewer else None
    viewer.vopt.flags[10] = viewer.vopt.flags[11] = not viewer.vopt.flags[10]
else:
    viewer = None

sitesToFit = ['C_' + whichSide, 'T_' + whichSide]

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

alignedKin.min(axis = 0)
