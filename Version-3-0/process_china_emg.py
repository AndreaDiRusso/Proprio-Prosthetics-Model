import os, argparse, pickle
import scipy.io as sio
import numpy as np
import pandas as pd
curfilePath = os.path.abspath(__file__)
#print(curfilePath)
curDir = os.path.abspath(os.path.join(curfilePath,os.pardir)) # this will return current directory in which python file resides.
#print(curDir)
parentDir = os.path.abspath(os.path.join(curDir,os.pardir)) # this will return parent directory.
#print(parentDir)

parser = argparse.ArgumentParser()
parser.add_argument('--matFile', default = 'W:/ENG_Neuromotion_Shared/group/Proprioprosthetics/Data/201709261100-Proprio/T_1_kinematics.pickle')
parser.add_argument('--startTime', default = '27.760')
parser.add_argument('--stopTime', default = '49.960')
parser.add_argument('--trial')
parser.add_argument('--dt', default = '0.01')
args = parser.parse_args()

matFile = args.matFile
startTime = float(args.startTime)
stopTime = float(args.stopTime)
dt = float(args.dt)
trial = int(args.trial)

mat = sio.loadmat(matFile, struct_as_record=False, squeeze_me=True)
array = mat['Array']
t = array.Trials[trial].Time + array.Trials[trial].info.t0
timeMask = np.logical_and(t >= startTime , t <= stopTime)
selectT = t[timeMask]
timeIdx = pd.Index(selectT, name = 'Time')

emgArray = array.Trials[trial].EMG
columns = []

for emg in emgArray:
    columns.append(emg.name)

columnIdx = pd.Index(columns, name = 'Muscle')

emgDF = pd.DataFrame(index = timeIdx, columns = columnIdx)
for emg in emgArray:
    emgDF.loc[:, emg.name] = emg.Data[timeMask]

saveName = '.'.join(matFile.split('.')[:-1]) + "_emg.pickle"

results = {
    'emg' : emgDF
}
with open(saveName, 'wb') as f:
    pickle.dump(results, f)
