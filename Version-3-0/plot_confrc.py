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

activeContacts = kinetics['active_contacts']
t0 = next(iter(activeContacts.keys()))
times = []
for time, value in activeContacts.items():
    times.append(time)
timeIdx = pd.Index(times, name = 'Time')

columns = []
for contactId, contactForce in activeContacts[t0].items():
    columns.append(contactId)
coordinates = ['xt', 'yt', 'zt', 'x', 'y', 'z']
columnIdx = pd.MultiIndex.from_product([columns, coordinates],
    names=['contactId', 'coordinate'])

activeContactsDF = pd.DataFrame(index = timeIdx, columns = columnIdx)

for time, reading in activeContacts.items():
    for contactId, contactForce in reading.items():
        activeContactsDF.loc[time, (contactId, 'xt')] = contactForce[0]
        activeContactsDF.loc[time, (contactId, 'yt')] = contactForce[1]
        activeContactsDF.loc[time, (contactId, 'zt')] = contactForce[2]
        activeContactsDF.loc[time, (contactId, 'x')] = contactForce[3]
        activeContactsDF.loc[time, (contactId, 'y')] = contactForce[4]
        activeContactsDF.loc[time, (contactId, 'z')] = contactForce[5]

contactFrc = long_form_df(activeContactsDF,
    overrideColumns = ['Contact ID', 'Coordinate', 'Time (sec)', 'Contact Force (N)'])
contactFrc.sort_values(by='Time (sec)', inplace = True)

lineNames = np.unique(contactFrc['Coordinate'])
"""
    colors = sns.color_palette("Blues", n_colors = 3) +\
        sns.color_palette("Reds", n_colors = 3)
    colors = [colors[i] for i in [0,2,4,1,3,5]]
    """
hueOpts = {
    'ls' : ['solid' for i in range(6)],
    'label' : list(lineNames),
    'lw' : [3 for i in range(6)]
    }

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

g = sns.FacetGrid(contactFrc, row = 'Contact ID', size = 3,
    hue = 'Coordinate', hue_order = lineNames, aspect = 3,
    despine = False, hue_kws = hueOpts, sharey = False, sharex = True)
g.map(plt.plot, 'Time (sec)', 'Contact Force (N)')

for idx, ax in enumerate(g.axes.flat):
    box = ax.get_position()
    ax.set_position([box.x0,box.y0,box.width*0.75,box.height])

plt.legend(loc='center right', bbox_to_anchor = (1.15,0.5))

plt.savefig(kineticsFile.split('_kinetics')[0] + '_confrc_plot.png')

pickleName = kineticsFile.split('_kinetics')[0] + '_confrc_plot.pickle'
with open(pickleName, 'wb') as f:
    pickle.dump(g,f)
