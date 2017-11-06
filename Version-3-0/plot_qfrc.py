import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import argparse, os, pickle
import seaborn as sns
from helper_functions import *

curfilePath = os.path.abspath(__file__)
#print(curfilePath)
curDir = os.path.abspath(os.path.join(curfilePath,os.pardir)) # this will return current directory in which python file resides.
#print(curDir)
parentDir = os.path.abspath(os.path.join(curDir,os.pardir)) # this will return parent directory.
#print(parentDir)

parser = argparse.ArgumentParser()
parser.add_argument('--kineticsFile', default = 'W:/ENG_Neuromotion_Shared/group/Proprioprosthetics/Data/201709261100-Proprio/T_1_filtered_kinetics.pickle')
parser.add_argument('--meanSubtract', dest='meanSubtract', action='store_true')
parser.add_argument('--whichSide', default = 'Left')
parser.add_argument('--whichQfrc', default = 'qfrc_inverse')
parser.set_defaults(meanSubtract = False)

args = parser.parse_args()
kineticsFile = args.kineticsFile
meanSubtract = args.meanSubtract
whichSide = args.whichSide
whichQfrc = args.whichQfrc

resourcesDir = curDir + '/Resources/Murdoc'

with open(kineticsFile, 'rb') as f:
    kinetics = pickle.load(f)

qFrcInverse = kinetics[whichQfrc]

jointsToPlot = [
    'Hip_' + whichSide,
    'Knee_' + whichSide,
    'Ankle_' + whichSide,
    'Toes_' + whichSide
    ]

t0 = next(iter(qFrcInverse.keys()))
times = []
for time, value in qFrcInverse.items():
    times.append(time)
timeIdx = pd.Index(times, name = 'Time')

columns = []
for joint, value in qFrcInverse[t0].items():
    if type(value) == np.float64:
        columns.append(joint.split(':')[0])
    if type(value) == np.ndarray:
        columns.append(joint)

columns = np.intersect1d(np.unique(columns), jointsToPlot)
coordinates = ['xt', 'yt', 'zt', 'x', 'y', 'z']
columnIdx = pd.MultiIndex.from_product([columns, coordinates],
    names=['joint', 'coordinate'])

qFrcInverseDF = pd.DataFrame(index = timeIdx, columns = columnIdx)

for time, reading in qFrcInverse.items():
    for joint, value in reading.items():
        if type(value) == np.float64:
            jointCoordinate = joint.split(':')[1]
            jointName = joint.split(':')[0]
            if jointName in jointsToPlot:
                qFrcInverseDF.loc[time, (jointName, jointCoordinate)] = value
        if type(value) == np.ndarray:
            if joint in jointsToPlot:
                qFrcInverseDF.loc[time, (joint , 'xt')] = value[0]
                qFrcInverseDF.loc[time, (joint , 'yt')] = value[1]
                qFrcInverseDF.loc[time, (joint , 'zt')] = value[2]
                qFrcInverseDF.loc[time, (joint , 'x')] = value[3]
                qFrcInverseDF.loc[time, (joint , 'y')] = value[4]
                qFrcInverseDF.loc[time, (joint , 'z')] = value[5]

qFrc = long_form_df(qFrcInverseDF,
    overrideColumns = ['Joint', 'Coordinate', 'Time (sec)', 'Joint Torque (N*m)'])
qFrc.fillna(0, inplace = True)
qFrc.sort_values(by='Time (sec)', inplace = True)

lineNames = np.unique(qFrc['Coordinate'])

hueOpts = {
    'ls' : ['solid' for i in range(6)],
    'label' : list(lineNames),
    'lw' : [3 for i in range(6)]
    }

sns.set_style('darkgrid')
plt.style.use('seaborn-darkgrid')
invertColors = True
matplotlib.rcParams.update({'font.size': 10})
matplotlib.rcParams.update({'text.color': 'black' if invertColors else 'white'})
matplotlib.rcParams.update({'axes.facecolor': 'white' if invertColors else 'black'})
matplotlib.rcParams.update({'axes.edgecolor': 'black' if invertColors else 'white'})
matplotlib.rcParams.update({'savefig.facecolor': 'white' if invertColors else 'black'})
matplotlib.rcParams.update({'savefig.edgecolor': 'black' if invertColors else 'white'})
matplotlib.rcParams.update({'figure.facecolor': 'white' if invertColors else 'black'})
matplotlib.rcParams.update({'figure.edgecolor': 'black' if invertColors else 'white'})
matplotlib.rcParams.update({'axes.labelcolor': 'black' if invertColors else 'white'})
matplotlib.rcParams.update({'xtick.color': 'black' if invertColors else 'white'})
matplotlib.rcParams.update({'ytick.color': 'black' if invertColors else 'white'})

g = sns.FacetGrid(qFrc, row = 'Joint', size = 3, aspect = 3,
    hue = 'Coordinate', hue_order = lineNames, hue_kws = hueOpts,
    despine = False, sharey = False, sharex = True)
g.map(plt.plot, 'Time (sec)', 'Joint Torque (N*m)')

for idx, ax in enumerate(g.axes.flat):
    box = ax.get_position()
    ax.set_position([box.x0,box.y0,box.width*0.75,box.height])

plt.legend(loc='center right', bbox_to_anchor = (1.15,0.5))

plt.savefig(kineticsFile.split('_kinetics')[0] + '_' + whichQfrc + '_plot.png')

pickleName = kineticsFile.split('_kinetics')[0] + '_' + whichQfrc + '_plot.pickle'
with open(pickleName, 'wb') as f:
    pickle.dump(g,f)
