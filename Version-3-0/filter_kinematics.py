import os, argparse, pickle
from scipy import signal
import numpy as np

curfilePath = os.path.abspath(__file__)
#print(curfilePath)
curDir = os.path.abspath(os.path.join(curfilePath,os.pardir)) # this will return current directory in which python file resides.
#print(curDir)
parentDir = os.path.abspath(os.path.join(curDir,os.pardir)) # this will return parent directory.
#print(parentDir)

parser = argparse.ArgumentParser()
parser.add_argument('--kinematicsFile', default = 'W:/ENG_Neuromotion_Shared/group/Proprioprosthetics/Data/201709261100-Proprio/T_1_kinematics.pickle')
parser.add_argument('--dt', default = '0.01')
args = parser.parse_args()

kinematicsFile = args.kinematicsFile
dt = float(args.dt)

with open(kinematicsFile, 'rb') as f:
    kinematics = pickle.load(f)

fr = 1 / dt
lowCutoff = 1
Wn = 2 * lowCutoff / fr
b, a = signal.butter(8, Wn, analog=False)
for column in kinematics['qpos']:
    kinematics['qpos'].loc[:, column] = signal.filtfilt(b, a, kinematics['qpos'].loc[:, column])

newName = kinematicsFile.split('_kinematics')[0] + '_filtered_kinematics.pickle'
with open(newName, 'wb') as f:
    pickle.dump(kinematics, f)
