#!/usr/bin/env python3
"""
Example for how to modifying the MuJoCo qpos during execution.
"""

import os, pdb, argparse, pickle
from mujoco_py import load_model_from_xml, MjSim, MjViewerBasic, MjViewer
from lmfit import minimize, Minimizer, Parameters, Parameter, report_fit
from helper_functions import *
from inverse_kinematics import *
import math

curfilePath = os.path.abspath(__file__)
#print(curfilePath)
curDir = os.path.abspath(os.path.join(curfilePath,os.pardir)) # this will return current directory in which python file resides.
#print(curDir)
parentDir = os.path.abspath(os.path.join(curDir,os.pardir)) # this will return parent directory.
#print(parentDir)

parser = argparse.ArgumentParser()
parser.add_argument('--kinematicsFile', default = 'W:/ENG_Neuromotion_Shared/group/Proprioprosthetics/Data/201709261100-Proprio/T_1.txt')
parser.add_argument('--startTime', default = '27.760')
parser.add_argument('--stopTime', default = '49.960')
parser.add_argument('--meshScale', default = '1.1e-3')
parser.add_argument('--lowCutoff')
parser.add_argument('--whichSide', default = 'Left')
parser.add_argument('--solverMethod', default = 'nelder')
parser.add_argument('--modelFile', default = 'murdoc_template_toes_floating.xml')
parser.add_argument('--showViewer', dest='showViewer', action='store_true')
parser.add_argument('--reIndex', dest='reIndex', type = tuple, nargs = 1)
parser.set_defaults(showViewer = False)
parser.set_defaults(reIndex = None)
args = parser.parse_args()

kinematicsFile = args.kinematicsFile
startTime = float(args.startTime)
stopTime = float(args.stopTime)
meshScale = float(args.meshScale)
lowCutoff = float(args.lowCutoff) if args.lowCutoff else None
whichSide = args.whichSide
modelFile = args.modelFile
showViewer = args.showViewer
solverMethod = args.solverMethod
reIndex = args.reIndex
showContactForces = True

resourcesDir = curDir + '/Resources/Murdoc'

templateFilePath = curDir + '/' + modelFile
fcsvFilePath = resourcesDir + '/Mobile Foot/Fiducials.fcsv'

specification = fcsv_to_spec(fcsvFilePath)
modelXML = populate_model(templateFilePath, specification, extraLocations = {},
    resourcesDir = resourcesDir, meshScale = meshScale, showTendons = True)

model = load_model_from_xml(modelXML)
simulation = MjSim(model)

statistics = {
    'nfev': [],
    'redchi': [],
    'aic' : [],
    'bic' : [],
    }
#viewer = MjViewerBasic(simulation) if showViewer else None
#TODO: make flag for enabling and disabling contact force rendering
if showContactForces and showViewer:
    viewer = MjViewer(simulation) if showViewer else None
    viewer.vopt.flags[10] = viewer.vopt.flags[11] = not viewer.vopt.flags[10]
else:
    viewer = None

sitesToFit = ['MT_' + whichSide, 'M_' + whichSide, 'C_' + whichSide,
    'GT_' + whichSide, 'K_' + whichSide
    #, 'T_' + whichSide
    ]

#initial guesses for eitehr side
jointsToFit = {
    'World:xt':{'value':0.09,'min':-3,'max':3},
    'World:yt':{'value':0.07,'min':-3,'max':3},
    'World:zt':{'value':0.05,'min':-3,'max':3},
    'World:xq':{'value':1.07,'min':math.radians(-180),'max':math.radians(180)},
    'World:yq':{'value':-1.78,'min':math.radians(-180),'max':math.radians(180)},
    'World:zq':{'value':1,'min':math.radians(-180),'max':math.radians(180)},
    'Hip_' + whichSide + ':x':{'value':-1.05,'min':math.radians(-60),'max':math.radians(120)},
    'Hip_' + whichSide + ':y':{'value':0.08,'min':math.radians(-5),'max':math.radians(5)},
    'Hip_' + whichSide + ':z':{'value':0.03,'min':math.radians(-5),'max':math.radians(5)},
    'Knee_' + whichSide + ':x':{'value':1.64,'min':math.radians(0),'max':math.radians(120)},
    'Ankle_' + whichSide + ':x':{'value':-1,'min':math.radians(-90),'max':math.radians(30)},
    'Ankle_' + whichSide + ':y':{'value':-0.1,'min':math.radians(-60),'max':math.radians(60)},
    #'Toes_' + whichSide + ':x':{'value':0.02,'min':math.radians(-120),'max':math.radians(-30)}
    } if whichSide == 'Right' else {
        'World:xt':{'value':0.06,'min':-10, 'max':10},
        'World:yt':{'value':0.02,'min':-10, 'max':10},
        'World:zt':{'value':-0.001,'min':-10, 'max':10},
        'World:xq':{'value':1.73,'min':math.radians(-180),'max':math.radians(180)},
        'World:yq':{'value':-0.56,'min':math.radians(-180),'max':math.radians(180)},
        'World:zq':{'value':1.72,'min':math.radians(-180),'max':math.radians(180)},
        'Hip_' + whichSide + ':x':{'value':-1,'min':math.radians(-120),'max':math.radians(60)},
        'Hip_' + whichSide + ':y':{'value':0,'min':math.radians(-5),'max':math.radians(5)},
        'Hip_' + whichSide + ':z':{'value':0.05,'min':math.radians(-5),'max':math.radians(5)},
        'Knee_' + whichSide + ':x':{'value':-1.57,'min':math.radians(-120),'max':math.radians(0)},
        'Ankle_' + whichSide + ':x':{'value':1.58,'min':math.radians(-30),'max':math.radians(90)},
        'Ankle_' + whichSide + ':y':{'value':0.1,'min':math.radians(-60),'max':math.radians(60)},
        #'Toes_' + whichSide + ':x':{'value':0.14,'min':math.radians(30),'max':math.radians(120)}
        }

#Get kinematics
timeSelection = [startTime, stopTime]
kinematics = get_kinematics(kinematicsFile,
    selectHeaders = sitesToFit, lowCutoff = lowCutoff,
    selectTime = timeSelection, reIndex = reIndex)

#provide initial fit
kinIterator = kinematics.iterrows()
t, kinSeries = next(kinIterator)

referenceJoint = 'C_' + whichSide
referenceSeries =\
    copy.deepcopy(get_site_pos(kinSeries, simulation).loc[referenceJoint, :])\
    -copy.deepcopy(kinSeries.loc[referenceJoint, :])

solver = IKFit(simulation, sitesToFit, jointsToFit,
    #skipThese = ['Hip_' + whichSide + ':y'],
    alignTo = referenceSeries, mjViewer = viewer, method = solverMethod,
    simulationType = 'forward')

stats = solver.fit(t, kinSeries)
printing = True
if printing:
    try:
        print("SSQ: ")
        print(np.sum(stats.residual**2))
        print(stats.message)
        report_fit(stats)
    except:
        pass

initSolution = stats.params
initDict = params_to_dict(initSolution)
skipThese = [
    #'Hip_' + whichSide + ':y',
    'World:xt',
    'World:yt',
    'World:zt',
    'World:xq',
    'World:yq',
    'World:zq',
    ]
newParams = dict_to_params(initDict, skip = skipThese)
#pdb.set_trace()
solver.jointsParam = newParams

modelKin = pd.DataFrame(index = kinematics.index, columns = kinematics.columns)
modelQpos = pd.DataFrame(index = kinematics.index, columns = params_to_series(stats.params).index)
alignedKin = pd.DataFrame(index = kinematics.index, columns = kinematics.columns)

for t, kinSeries in kinematics.iterrows():
    #pdb.set_trace()
    stats = solver.fit(t, kinSeries)

    if printing:
        try:
            print("SSQ: ")
            print(np.sum(stats.residual**2))
            print(stats.message)
            report_fit(stats)
        except:
            pass

    statistics['nfev'].append(stats.nfev)
    statistics['redchi'].append(stats.redchi)
    statistics['aic'].append(stats.aic)
    statistics['bic'].append(stats.bic)

    solver.jointsParam = stats.params
    modelKin.loc[t, :] = get_site_pos(kinSeries, simulation)
    modelQpos.loc[t, :] = params_to_series(stats.params)
    alignedKin.loc[t, :] = alignToModel(simulation, kinSeries, referenceSeries)

results = {
    'site_pos' : modelKin,
    'orig_site_pos': alignedKin,
    'qpos' : modelQpos,
    'meshScale' : meshScale
}

saveName = '.'.join(kinematicsFile.split('.')[:-1]) + "_kinematics.pickle"
with open(saveName, 'wb') as f:
    pickle.dump(results, f)

saveName = '.'.join(kinematicsFile.split('.')[:-1]) + "_" + solverMethod + "_fit_statistics.pickle"
with open(saveName, 'wb') as f:
    pickle.dump(statistics, f)

print('Finished with %4.2f average function calls' % np.mean(statistics['nfev']))
print('Finished with %4.6f average reduced chisquare' % np.mean(statistics['redchi']))
print('Finished with mean AIC of  %4.2f' % np.mean(statistics['aic']))
print('Finished with mean BIC of %4.6f' % np.mean(statistics['bic']))
