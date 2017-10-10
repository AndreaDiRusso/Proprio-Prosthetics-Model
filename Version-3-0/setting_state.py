#!/usr/bin/env python3
"""
Example for how to modifying the MuJoCo qpos during execution.
"""

import os, pdb, argparse
from mujoco_py import load_model_from_xml, MjSim, MjViewer
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

args = parser.parse_args()

kinematicsFile = args.kinematicsFile
# kinematicsFile = 'W:/ENG_Neuromotion_Shared/group/Proprioprosthetics/Training/T_1.txt'

resourcesDir = curDir + '/Resources/Murdoc'

templateFilePath = curDir + '/murdoc_template.xml'
fcsvFilePath = resourcesDir + '/Fiducials_old.fcsv'

specification = fcsv_to_spec(fcsvFilePath)
modelXML = populate_model(templateFilePath, specification, resourcesDir, showTendons = False)

model = load_model_from_xml(modelXML)
simulation = MjSim(model)
viewer = MjViewer(simulation)

sitesToFit = ['MT_Left', 'M_Left', 'C_Left', 'GT_Left', 'K_Left']

jointsToFit = {
    'World:xt':(0,-1e1, 1e1) ,
    'World:yt':(0,-1e1, 1e1) ,
    'World:zt':(0,-1e1, 1e1),
    'World:x':(math.radians(0),math.radians(-180),math.radians(180)),
    'World:y':(math.radians(0),math.radians(-180),math.radians(180)),
    'World:z':(math.radians(0),math.radians(-180),math.radians(180)),
    'Hip_Left:x':(math.radians(0),math.radians(-60),math.radians(90)),
    'Hip_Left:y':(math.radians(0),math.radians(-15),math.radians(15)),
    'Hip_Left:z':(math.radians(0),math.radians(-15),math.radians(15)),
    'Knee_Left:x':(math.radians(0),math.radians(-90),math.radians(70)),
    'Ankle_Left:x':(math.radians(0),math.radians(-60),math.radians(120)),
    'Ankle_Left:y':(math.radians(0),math.radians(-0),math.radians(30)),
    }

referenceJoint = 'C_Left'
solver = IKFit(simulation, sitesToFit, jointsToFit, alignTo = referenceJoint, mjViewer = viewer)
#Get kinematics
kin = get_kinematics(kinematicsFile,
    selectHeaders = sitesToFit,
    selectTime = [26, 50])


for t, kinSeries in kin.iterrows():
    stats = solver.fit(t, kinSeries)

    print("SSQ: ")
    print(np.sum(stats.residual**2))

    render_targets(viewer, alignToModel(simulation, kinSeries, referenceJoint))
    
