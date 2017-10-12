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
parser.add_argument('--modelFrFile', default = 'W:/ENG_Neuromotion_Shared/group/Proprioprosthetics/Training/T_1_model.pickle')
parser.add_argument('--outputFile')

args = parser.parse_args()
modelFrFile = args.modelFrFile
outputFile = args.outputFile if args.outputFile else None

resourcesDir = curDir + '/Resources/Murdoc'

with open(modelFrFile, 'rb') as f:
    kinematics = pickle.load(f)

iARate = long_form_df(kinematics['iARate'], overrideColumns = ['Tendon', 'Time (sec)', 'Firing Rate (Hz)'])

sns.set_style('darkgrid')
plt.style.use('seaborn-darkgrid')
matplotlib.rcParams.update({'font.size': 20})
matplotlib.rcParams.update({'text.color': 'black'})
matplotlib.rcParams.update({'axes.facecolor': 'white'})
matplotlib.rcParams.update({'axes.edgecolor': 'black'})
matplotlib.rcParams.update({'savefig.facecolor': 'white'})
matplotlib.rcParams.update({'savefig.edgecolor': 'black'})
matplotlib.rcParams.update({'figure.facecolor': 'white'})
matplotlib.rcParams.update({'figure.edgecolor': 'black'})
matplotlib.rcParams.update({'axes.labelcolor': 'black'})

g = sns.FacetGrid(iARate, row = 'Tendon', size = 3, aspect = 3,
    despine = False, sharey = False)
g.map(plt.plot, 'Time (sec)', 'Firing Rate (Hz)', lw = 3)

plt.savefig(outputFile)
