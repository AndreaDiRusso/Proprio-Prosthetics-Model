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
parser.add_argument('--frFile', default = 'W:/ENG_Neuromotion_Shared/group/Proprioprosthetics/Data/201709261100-Proprio/T_1_filtered_fr.pickle')

args = parser.parse_args()
frFile = args.frFile

resourcesDir = curDir + '/Resources/Murdoc'

with open(frFile, 'rb') as f:
    kinematics = pickle.load(f)

iARate = long_form_df(kinematics['iARate'], overrideColumns = ['Tendon', 'Time (sec)', 'Firing Rate (Hz)'])

sns.set_style('darkgrid')
plt.style.use('seaborn-darkgrid')
invertColors = True
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

g = sns.FacetGrid(iARate, row = 'Tendon', size = 3, aspect = 3,
    despine = False, sharey = False)
g.map(plt.plot, 'Time (sec)', 'Firing Rate (Hz)', lw = 3)

plt.savefig(frFile.split('_fr')[0] + '_fr_plot.png')
plt.savefig(frFile.split('_fr')[0] + '_fr_plot.eps')

pickleName = frFile.split('_fr')[0] + '_fr_plot.pickle'
with open(pickleName, 'wb') as f:
    pickle.dump(g,f)
