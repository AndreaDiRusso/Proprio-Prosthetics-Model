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
parser.add_argument('--emgFile', default = 'W:/ENG_Neuromotion_Shared/group/MI locomotion data/Biomechanical Model/q19d20131124tkTRDMdsNORMt401/Array_Q19_20131124_emg.pickle')
parser.add_argument('--meanSubtract', dest='meanSubtract', action='store_true')
parser.set_defaults(meanSubtract = False)

args = parser.parse_args()
emgFile = args.emgFile
meanSubtract = args.meanSubtract

with open(emgFile, 'rb') as f:
    emg = pickle.load(f)['emg']

musclesToPlot = [
    'RF',
    'ST',
    'GMD',
    'TA',
    'IPS',
    'GN'
    ]

selectMuscles = np.intersect1d(emg.columns, musclesToPlot)
emg = emg.loc[:, selectMuscles]

emgLDF = long_form_df(emg,
    overrideColumns = ['Muscle', 'Time (sec)', 'EMG (mV)'])
emgLDF.fillna(0, inplace = True)
emgLDF.sort_values(by='Time (sec)', inplace = True)

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

g = sns.FacetGrid(emgLDF, row = 'Muscle', size = 3, aspect = 3,
    despine = False, sharey = False, sharex = True)
g.map(plt.plot, 'Time (sec)', 'EMG (mV)')

plt.savefig(emgFile.split('_emg')[0] + '_emg_plot.png')

pickleName = emgFile.split('_emg')[0] + '_emg_plot.pickle'
with open(pickleName, 'wb') as f:
    pickle.dump(g,f)
