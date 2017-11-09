import os, argparse, pickle, copy, time, math
import mujoco_py
import glfw
from mujoco_py import load_model_from_xml, MjSim, MjViewer, functions
from mujoco_py.utils import rec_copy, rec_assign
from helper_functions import *
import numpy as np
from constants import *
import seaborn as sns
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

curfilePath = os.path.abspath(__file__)
#print(curfilePath)
curDir = os.path.abspath(os.path.join(curfilePath,os.pardir)) # this will return current directory in which python file resides.
#curDir = 'C:/Users/Radu/Documents/GitHub/Proprio-Prosthetics-Model/Version-3-0'

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

modelFile = 'murdoc_template_toes_floating.xml'
meshScale = 0.9e-3
resourcesDir = curDir + '/Resources/Murdoc'
tendonNames = ['SOL_Right']

resourcesDir = curDir + '/Resources/Murdoc'
templateFilePath = curDir + '/' + modelFile

fcsvFilePath = resourcesDir + '/Mobile Foot/Fiducials.fcsv'

specification = fcsv_to_spec(fcsvFilePath)
modelXML = populate_model(templateFilePath, specification, resourcesDir =  resourcesDir,
    meshScale = meshScale, showTendons = True)

model = load_model_from_xml(modelXML)

functions.mj_setTotalmass(model, 6.5)
simulation = MjSim(model)

#viewer = MjViewer(simulation)

#get resting lengths
nJoints = simulation.model.njnt
simulation = pose_to_key(simulation, 0)
tendonL0 = pd.Series(index = tendonNames)
tendonL0 = get_tendon_length(tendonL0, simulation)

nSamples = 2000
jointRange = np.linspace(math.radians(-90), math.radians(0), num = nSamples)
tendonL = pd.Series(np.zeros(nSamples), index = jointRange)
tendonV = pd.Series(np.zeros(nSamples), index = jointRange)

jointDict = {
    'Ankle_Right:x' : {'value': -0.6},
    }

tendonId = functions.mj_name2id(simulation.model, mjtObj.OBJ_TENDON.value, 'SOL_Right')

for newAngle in jointRange:

    jointDict['Ankle_Right:x']['value'] = newAngle

    pose_model(simulation, jointDict)

    tendonL.loc[newAngle] = simulation.data.ten_length[tendonId]

    #viewer.render()

    #time.sleep(0.05)

"""
plt.plot(tendonL)
plt.show()
"""


#start 1 cm away from max, i.e. @0.15
#start manipulationg when tendon is 14.4 cm long
angleStart = (tendonL - 0.15).abs().idxmin()
#start manipulationg when tendon is 15.6 cm long
angleStop = (tendonL - 0.156).abs().idxmin()
#have to cover this much angle
deltaAngle = abs(angleStop - angleStart)

speeds = [2.5, 5, 10, 20, 35, 45]
dynamicIndices = pd.Series(index = speeds)

gains = [0.5, 160, 380, 7]
# replicate figure 5 oc cheney 1976
for speed in speeds:
    #6 mm at speed mm / s, how long is the ramp
    deltaT = 6 / speed
    # pad with 1 second of stationarity on either side
    t = np.linspace(0, 2 + deltaT, num = nSamples)
    nSamplesStationary = 1 * int(nSamples / (2 + deltaT))
    nSamplesRamp = nSamples - 2 * nSamplesStationary
    dt = (2 + deltaT) / nSamples

    jointRange = np.concatenate(( np.ones((nSamplesStationary )) * angleStart
        , np.linspace(angleStart, angleStop, num = nSamplesRamp)
        , np.ones((nSamplesStationary)) * angleStop ))

    tendonL = pd.Series(np.zeros(nSamples), index = pd.Index(t, name = 'Time'))

    simulation = pose_to_key(simulation, 0)

    for idx, newAngle in enumerate(jointRange):

        jointDict['Ankle_Right:x']['value'] = newAngle

        pose_model(simulation, jointDict)

        tendonL.iloc[idx] = simulation.data.ten_length[tendonId]
        #viewer.render()

    tendonL = pd.DataFrame(tendonL, columns = ['SOL_Right'])
    tendonV = tendonL.diff(axis = 0).fillna(0) / dt

    iARate = Ia_model_Radu(tendonL, tendonV,  tendonL0, gains)

    baseFR = iARate.iloc[-1, :]
    #print(baseFR)
    #pdb.set_trace()
    dynamicIndex = iARate.max(axis = 0) - baseFR
    dynamicIndices.loc[speed] = dynamicIndex.values[0]
    #print(dynamicIndex)

    if speed == 35:
        f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

        ax1.plot(t, iARate)
        plt.suptitle('Prochazka muscle spindle model')
        ax1.set_ylabel('Ia afferent firing rate (Hz)')

        ax2.plot(t, 1000*(tendonL - 0.15))
        ax2.set_xlabel('Time (sec)')
        ax2.set_ylabel('Muscle Length (m)')

        ax2.set_ylim((0,10))
        sns.despine(offset=10, trim=True)
        plt.savefig('D:\Dropbox (Brown)\Borton lab\Lab presentations\Radu Posters\SfN 2017\cheney_plot.png')
        plt.savefig('D:\Dropbox (Brown)\Borton lab\Lab presentations\Radu Posters\SfN 2017\cheney_plot.eps')

        pickleName = 'D:\Dropbox (Brown)\Borton lab\Lab presentations\Radu Posters\SfN 2017\cheney_plot.pickle'

        with open(pickleName, 'wb') as fil:
            pickle.dump(f, fil)
        plt.show()

f, ax = plt.subplots(1,1)
ax.plot(dynamicIndices.index, dynamicIndices.values)
#pdb.set_trace()
referenceIndices = dynamic_index(pd.Series(speeds) / 1000)
#plt.plot(speeds, referenceIndices, 'r-')
ax.set_xlabel('Stretch Rate (mm/sec)')
ax.set_ylim((0,100))
ax.set_ylabel('Dynamic Index (Hz)')

plt.savefig('D:\Dropbox (Brown)\Borton lab\Lab presentations\Radu Posters\SfN 2017\cheney_plot_dynamic_index.png')
plt.savefig('D:\Dropbox (Brown)\Borton lab\Lab presentations\Radu Posters\SfN 2017\cheney_plot_dynamic_index.eps')

pickleName = 'D:\Dropbox (Brown)\Borton lab\Lab presentations\Radu Posters\SfN 2017\cheney_plot_dynamic_index.pickle'
with open(pickleName, 'wb') as fil:
    pickle.dump(f, fil)
plt.show()

# replicate figure 6 oc cheney 1976
deltaLen = np.asarray([2, 4, 6, 8, 10])
lens = deltaLen / 1000 + np.ones(deltaLen.shape) * tendonL0.values[0]
steadyTenLen = pd.DataFrame(lens, index = deltaLen, columns = tendonL.columns)
zeroTenV = pd.DataFrame(lens * 0, index = deltaLen, columns = tendonL.columns)
iARate = Ia_model_Radu(steadyTenLen, zeroTenV,  tendonL0, gains)

f, ax = plt.subplots(1,1)
ax.plot(iARate)
referenceRates = base_freq(pd.Series(deltaLen) / 1000)
#plt.plot(deltaLen, referenceRates, 'r-')
ax.set_xlabel('Stretch Magnitude (mm)')
ax.set_ylabel('Afferent Frequency (Hz)')
plt.savefig('D:\Dropbox (Brown)\Borton lab\Lab presentations\Radu Posters\SfN 2017\cheney_plot_steady.png')
plt.savefig('D:\Dropbox (Brown)\Borton lab\Lab presentations\Radu Posters\SfN 2017\cheney_plot_steady.eps')

pickleName = 'D:\Dropbox (Brown)\Borton lab\Lab presentations\Radu Posters\SfN 2017\cheney_plot_steady.pickle'
plt.show()
