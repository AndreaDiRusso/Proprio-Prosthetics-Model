#!/usr/bin/env python3
"""
Example for how to modifying the MuJoCo qpos during execution.
"""

import os, pdb, argparse
from mujoco_py import load_model_from_xml, MjSim, MjViewer
from helper_functions import *
from inverse_kinematics import *

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
sim = MjSim(model)
viewer = MjViewer(sim)

sitesToFit = ['MT_Left', 'M_Left', 'C_Left', 'GT_Left', 'K_Left']

jointsToFit = {
    'World:xt':(0,-3.14, 3.14) ,
    'World:yt':(0,-3.14, 3.14) ,
    'World:zt':(0,-3.14, 3.14),
    'World:x':(0,-3.14, 3.14),
    'World:y':(0,-3.14, 3.14),
    'World:z':(0,-3.14, 3.14),
    'Hip_Left:x':(0,-3.14, 3.14),
    'Hip_Left:y':(0,-3.14, 3.14),
    'Hip_Left:z':(0,-3.14, 3.14),
    'Knee_Left:x':(0,-3.14, 3.14),
    'Ankle_Left:x':(0,-3.14, 3.14),
    'Ankle_Left:y':(0,-3.14, 3.14),
    }

solver = IKFit(sim, sitesToFit, jointsToFit)
#Get kinematics
kin = get_kinematics(kinematicsFile,
    selectHeaders = sitesToFit,
    selectTime = [26, 50])

for t, row in kin.iterrows():
    stats = solver.fit(t, row)
    report_fit(stats)

    sim_state = sim.get_state()
    sim.forward()
    viewer.render()
