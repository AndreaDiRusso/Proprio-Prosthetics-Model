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
parser.set_defaults(meanSubtract = False)

args = parser.parse_args()
kineticsFile = args.kineticsFile
meanSubtract = args.meanSubtract

resourcesDir = curDir + '/Resources/Murdoc'

with open(kineticsFile, 'rb') as f:
    kinetics = pickle.load(f)

qFrcConstraint = kinetics['qfrc_constraint']
t0 = next(iter(qFrcConstraint.keys()))
times = []
for time, value in qFrcConstraint.items():
    times.append(time)
columns = []
for joint, value in qFrcConstraint[t0].items():
    if type(value) == np.float64:
        columns.append(joint)
    if type(value) == np.ndarray:
        columns.append(joint + ':xt')
        columns.append(joint + ':yt')
        columns.append(joint + ':zt')
        columns.append(joint + ':x')
        columns.append(joint + ':y')
        columns.append(joint + ':z')

qFrcConstraintDF = pd.DataFrame(index = times, columns = columns)

for time, reading in qFrcConstraint.items():
    for joint, value in reading.items():
        if type(value) == np.float64:
            qFrcConstraintDF.loc[time, joint] = value
        if type(value) == np.ndarray:
            qFrcConstraintDF.loc[time, joint + ':xt'] = value[0]
            qFrcConstraintDF.loc[time, joint + ':yt'] = value[1]
            qFrcConstraintDF.loc[time, joint + ':zt'] = value[2]
            qFrcConstraintDF.loc[time, joint + ':x'] = value[3]
            qFrcConstraintDF.loc[time, joint + ':y'] = value[4]
            qFrcConstraintDF.loc[time, joint + ':z'] = value[5]

qFrcConstraint = long_form_df(qFrcConstraintDF,
    overrideColumns = ['Tendon', 'Time (sec)', 'Constraint Joint Torque (N*m)'])
qFrcConstraint.sort_values(by='Time (sec)', inplace = True)

sns.set_style('darkgrid')
plt.style.use('seaborn-darkgrid')
invertColors = False
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

g = sns.FacetGrid(qFrcConstraint, row = 'Tendon', size = 3, aspect = 3,
    despine = False, sharey = False, sharex = True)
g.map(plt.plot, 'Time (sec)', 'Constraint Joint Torque (N*m)', lw = 3)
#g.set(ylim=(-.25, .25))

plt.savefig(kineticsFile.split('_kinetics')[0] + '_qfrc_constraint_plot.png')

pickleName = kineticsFile.split('_kinetics')[0] + '_qfrc_constraint_plot.pickle'
with open(pickleName, 'wb') as f:
    pickle.dump(g,f)
