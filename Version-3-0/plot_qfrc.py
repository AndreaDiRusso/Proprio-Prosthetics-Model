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
parser.add_argument('--modelQfrcFile', default = 'W:/ENG_Neuromotion_Shared/group/Proprioprosthetics/Data/201709261100-Proprio/T_1_model_kinetics.pickle')
parser.add_argument('--outputFile')

args = parser.parse_args()
modelQfrcFile = args.modelQfrcFile
outputFile = args.outputFile if args.outputFile else None

resourcesDir = curDir + '/Resources/Murdoc'

with open(modelQfrcFile, 'rb') as f:
    kinematics = pickle.load(f)

qFrc = long_form_df(kinematics['qfrc_inverse'],
    overrideColumns = ['Tendon', 'Time (sec)', 'Joint Torque (N*m)'])

sns.set_style('darkgrid')
plt.style.use('seaborn-darkgrid')
invertColors = False
matplotlib.rcParams.update({'font.size': 30})
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

g = sns.FacetGrid(qFrc, row = 'Tendon', size = 3, aspect = 3,
    despine = False, sharey = False)
g.map(plt.plot, 'Time (sec)', 'Joint Torque (N*m)', lw = 3)

plt.savefig(outputFile)

pickleName = outputFile.split('.')[0] + '.pickle'
with open(pickleName, 'wb') as f:
    pickle.dump(g,f)
