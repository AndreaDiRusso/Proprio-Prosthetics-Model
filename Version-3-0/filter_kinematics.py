import os, argparse, pickle
from scipy import signal
import numpy as np
from helper_functions import *
from mujoco_py import load_model_from_xml, MjSim, MjViewerBasic, MjViewer

curfilePath = os.path.abspath(__file__)
#print(curfilePath)
curDir = os.path.abspath(os.path.join(curfilePath,os.pardir)) # this will return current directory in which python file resides.
#print(curDir)
parentDir = os.path.abspath(os.path.join(curDir,os.pardir)) # this will return parent directory.
#print(parentDir)

parser = argparse.ArgumentParser()
parser.add_argument('--kinematicsFile', default = 'W:/ENG_Neuromotion_Shared/group/Proprioprosthetics/Data/201709261100-Proprio/T_1_kinematics.pickle')
parser.add_argument('--dt', default = '0.01')
parser.add_argument('--meshScale', default = '1.1e-3')
parser.add_argument('--lowCutoff', default = '5')
parser.add_argument('--modelFile', default = 'murdoc_template_toes_floating.xml')
args = parser.parse_args()

kinematicsFile = args.kinematicsFile
dt = float(args.dt)
modelFile = args.modelFile
meshScale = float(args.meshScale)
lowCutoff = float(args.lowCutoff)

resourcesDir = curDir + '/Resources/Murdoc'
templateFilePath = curDir + '/' + modelFile

fcsvFilePath = resourcesDir + '/Mobile Foot/Fiducials.fcsv'

specification = fcsv_to_spec(fcsvFilePath)
modelXML = populate_model(templateFilePath, specification, resourcesDir = resourcesDir,
    meshScale = meshScale, showTendons = True)

model = load_model_from_xml(modelXML)

simulation = MjSim(model)

with open(kinematicsFile, 'rb') as f:
    kinematics = pickle.load(f)

fr = 1 / dt
Wn = 2 * lowCutoff / fr
b, a = signal.butter(12, Wn, analog=False)
for column in kinematics['qpos']:
    kinematics['qpos'].loc[:, column] = signal.filtfilt(b, a, kinematics['qpos'].loc[:, column])

for t, kinSeries in kinematics['orig_site_pos'].iterrows():

    jointDict = series_to_dict( kinematics['qpos'].loc[t, :])
    pose_model(simulation, jointDict)
    kinematics['site_pos'].loc[t, :] = get_site_pos(kinSeries, simulation)

newName = kinematicsFile.split('_kinematics')[0] + '_filtered_kinematics.pickle'
with open(newName, 'wb') as f:
    pickle.dump(kinematics, f)
