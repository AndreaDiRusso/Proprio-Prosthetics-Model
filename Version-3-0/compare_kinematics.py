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
parser.add_argument('--modelKinematicsFile', default = 'Z:\\ENG_Neuromotion_Shared\\group\\Proprioprosthetics\\Data\\201709261100-Proprio\\T_1_model.pickle')
parser.add_argument('--outputFile')
parser.add_argument('--meanSubtract', dest='meanSubtract', action='store_true')
parser.set_defaults(meanSubtract = False)

args = parser.parse_args()
modelKinematicsFile = args.modelKinematicsFile
outputFile = args.outputFile if args.outputFile else None
meanSubtract = args.meanSubtract

resourcesDir = curDir + '/Resources/Murdoc'

with open(modelKinematicsFile, 'rb') as f:
    kinematics = pickle.load(f)

overrideColumns = ['Site', 'Coordinate', 'Time (sec)', 'Position (m)']

if meanSubtract:
    modelMeans = kinematics['site_pos'].mean(axis = 0)
    origShort = kinematics['orig_site_pos'].sub(modelMeans, axis = 1).round(decimals = 2)
    modelShort = kinematics['site_pos'].sub(modelMeans, axis = 1).round(decimals = 2)
else:
    origShort = kinematics['orig_site_pos']
    modelShort = kinematics['site_pos']

model = long_form_df(modelShort, overrideColumns = overrideColumns)
orig = long_form_df(origShort, overrideColumns = overrideColumns)

model['Coordinate'] = pd.Series(['Model ' + coord for coord in model['Coordinate']])
orig['Coordinate'] = pd.Series(['Original ' + coord for coord in orig['Coordinate']])

stack = pd.concat([model, orig], axis = 0)

lineNames = np.unique(stack['Coordinate'])

colors = sns.color_palette("Blues", n_colors = 2) + sns.color_palette("Reds", n_colors = 2) + sns.color_palette("Greens", n_colors = 2)
colors = [colors[i] for i in [0,2,4,1,3,5]]

hueOpts = {
    'ls' : ['dashed' for i in range(3)] + ['solid' for i in range(3)],
    'label' : list(lineNames),
    'lw' : [3 for i in range(6)]
}

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

g = sns.FacetGrid(stack, row = 'Site', palette = colors, size = 24/5, aspect = 3,
    hue = 'Coordinate', hue_order = lineNames, hue_kws = hueOpts, despine = False,
    sharey = False)

g.map(plt.plot, 'Time (sec)', 'Position (m)')

for idx, ax in enumerate(g.axes.flat):
    box = ax.get_position()
    ax.set_position([box.x0,box.y0,box.width*0.75,box.height])

plt.legend(loc='center right', bbox_to_anchor = (1.4,3))
plt.savefig(outputFile)

pickleName = outputFile.split('.')[0] + '.pickle'
with open(pickleName, 'wb') as f:
    pickle.dump(g,f)
