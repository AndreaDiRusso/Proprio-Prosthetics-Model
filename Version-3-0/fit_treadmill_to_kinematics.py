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
import quaternion as quat
import numpy as np

curfilePath = os.path.abspath(__file__)
#print(curfilePath)
curDir = os.path.abspath(os.path.join(curfilePath,os.pardir)) # this will return current directory in which python file resides.
#print(curDir)
parentDir = os.path.abspath(os.path.join(curDir,os.pardir)) # this will return parent directory.
#print(parentDir)

parser = argparse.ArgumentParser()
parser.add_argument('--kinematicsFile', default = 'W:/ENG_Neuromotion_Shared/group/MI locomotion data/Biomechanical Model/q19d20131124tkTRDMdsNORMt401/Q19_20131124_pre_TRDM_(4.0)_1_KIN_processed.txt')
parser.add_argument('--startTime', default = '40.5')
parser.add_argument('--stopTime', default = '50')
parser.add_argument('--showViewer', dest='showViewer', action='store_true')
parser.add_argument('--reIndex', dest='reIndex', type = tuple, nargs = 1)
parser.set_defaults(showViewer = True)
parser.set_defaults(reIndex = None)
args = parser.parse_args()

kinematicsFile = args.kinematicsFile
startTime = float(args.startTime)
stopTime = float(args.stopTime)
showViewer = args.showViewer
reIndex = args.reIndex
showContactForces = True

print(reIndex)

resourcesDir = curDir + '/Resources/Murdoc'

templateFilePath = curDir + '/murdoc_template_static_treadmill.xml'
fcsvFilePath = resourcesDir + '/Aligned-To-Pelvis/Fiducials.fcsv'

meshScale = .9e-3
specification = fcsv_to_spec(fcsvFilePath)
modelXML = populate_model(templateFilePath, specification,
    resourcesDir, meshScale = meshScale, showTendons = True)

alignModel= load_model_from_xml(modelXML)
simulation = MjSim(alignModel)

#viewer = MjViewerBasic(simulation) if showViewer else None
#TODO: make flag for enabling and disabling contact force rendering
if showContactForces and showViewer:
    viewer = MjViewer(simulation) if showViewer else None
    viewer.vopt.flags[10] = viewer.vopt.flags[11] = not viewer.vopt.flags[10]
else:
    viewer = None

whichSide = 'Right'
sitesToFit = ['MT_' + whichSide, 'M_' + whichSide, 'C_' + whichSide, 'GT_' + whichSide, 'K_' + whichSide]

jointsToFit = {
    'World:xt':{'value':0.002,'min':-3,'max':3},
    'World:yt':{'value':-0.004,'min':-3,'max':3},
    'World:zt':{'value':0.015,'min':-3,'max':3},
    'World:xq':{'value':0.89,'min':math.radians(-180),'max':math.radians(180)},
    'World:yq':{'value':-0.51,'min':math.radians(-180),'max':math.radians(180)},
    'World:zq':{'value':-1.57,'min':math.radians(-180),'max':math.radians(180)},
    'Hip_' + whichSide + ':x':{'value':-0.34,'min':math.radians(-60),'max':math.radians(120)},
    'Hip_' + whichSide + ':y':{'value':0.08,'min':math.radians(-5),'max':math.radians(5)},
    'Hip_' + whichSide + ':z':{'value':0.085,'min':math.radians(-5),'max':math.radians(5)},
    'Knee_' + whichSide + ':x':{'value':1.64,'min':math.radians(0),'max':math.radians(120)},
    'Ankle_' + whichSide + ':x':{'value':-1.56,'min':math.radians(-90),'max':math.radians(30)},
    'Ankle_' + whichSide + ':y':{'value':0.14,'min':math.radians(-60),'max':math.radians(60)},
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
        }

#Get kinematics
timeSelection = [startTime, stopTime]
kinematics = get_kinematics(kinematicsFile,
    selectHeaders = sitesToFit,
    selectTime = timeSelection, reIndex = reIndex)

#provide initial fit
kinIterator = kinematics.iterrows()
t, kinSeries = next(kinIterator)

referenceJoint = 'C_' + whichSide
referenceSeries =\
    copy.deepcopy(get_site_pos(kinSeries, simulation).loc[referenceJoint, :])\
    -copy.deepcopy(kinSeries.loc[referenceJoint, :])

solver = IKFit(simulation, sitesToFit, jointsToFit,
    skipThese = ['Hip_' + whichSide + ':y'],
    alignTo = referenceSeries, mjViewer = viewer, method = 'nelder',
    simulationType = 'forward')

stats = solver.fit(t, kinSeries)
printing = True
if printing:
    try:
        print("SSQ: ")
        print(np.sum(stats.residual**2))
        print(stats.message)
        report_fit(stats)

        statistics['nfev'].append(stats.nfev)
        statistics['redchi'].append(stats.redchi)
    except:
        pass

#set initial guess to result of initial fit
initialResults = params_to_dict(stats.params)
for key, value in jointsToFit.items():
    jointsToFit[key]['value'] = initialResults[key]['value']

secondTemplateFilePath = '/'.join(templateFilePath.split('/')[:-1]) +\
    '/murdoc_template_mobile_treadmill_zero.xml'
modelXML2 = populate_model(secondTemplateFilePath, specification, resourcesDir,
    meshScale = meshScale, showTendons = True)

#model that will be varied for optimization fitting
optModel = load_model_from_xml(modelXML2)
optSim = MjSim(optModel)

#viewer = MjViewerBasic(simulation) if showViewer else None
#TODO: make flag for enabling and disabling contact force rendering
if showContactForces and showViewer:
    viewer2 = MjViewer(optSim) if showViewer else None
    viewer2.vopt.flags[10] = viewer2.vopt.flags[11] = not viewer2.vopt.flags[10]
else:
    viewer2 = None

worldQ = get_euler_rotation_quaternion(jointsToFit, 'World')

#TODO This doesn't seem to pose the new model in the right place!!
jointsToFit.update({
    'World:xq':{'value': jointsToFit['World:x']['value'], 'min':math.radians(-180), 'max':math.radians(180)},
    'World:yq':{'value': jointsToFit['World:y']['value'], 'min':math.radians(-180), 'max':math.radians(180)},
    'World:zq':{'value': jointsToFit['World:z']['value'], 'min':math.radians(-180), 'max':math.radians(180)},
    })

skip = [
    'World:x',
    'World:y',
    'World:z',
    ]

for joint in skip:
    jointsToFit.pop(joint)

solver2 = IKFit(optSim, sitesToFit, jointsToFit,
    skipThese = ['Hip_' + whichSide + ':y'],
    alignTo = referenceSeries, mjViewer = viewer2,
    method = 'nelder', simulationType = 'forward')

solver2.jointsParam = dict_to_params(jointsToFit)

modelKin = pd.DataFrame(index = kinematics.index, columns = kinematics.columns)
modelQpos = pd.DataFrame(index = kinematics.index, columns = params_to_series(solver2.jointsParam).index)
alignedKin = pd.DataFrame(index = kinematics.index, columns = kinematics.columns)

#solver2.nelderTol = 1e-3
statistics = {
    'nfev': [],
    'redchi': []
    }

"""
for i in range(int(2e3)):
    #settle model
    optSim.step()
    if viewer2:
        viewer2.render()
"""
printing = True
for t, kinSeries in kinematics.iterrows():
    """
    referenceSeries =\
        copy.deepcopy(get_site_pos(kinSeries, optSim).loc[referenceJoint, :])\
        -copy.deepcopy(kinSeries.loc[referenceJoint, :])

    solver2.alignTo = referenceSeries
    """
    stats = solver2.fit(t, kinSeries)

    if printing:
        try:
            print("SSQ: ")
            print(np.sum(stats.residual**2))
            print(stats.message)
            report_fit(stats)

            statistics['nfev'].append(stats.nfev)
            statistics['redchi'].append(stats.redchi)
        except:
            pass

    solver2.jointsParam = stats.params

    modelKin.loc[t, :] = get_site_pos(kinSeries, optSim)
    modelQpos.loc[t, :] = params_to_series(stats.params)
    alignedKin.loc[t, :] = alignToModel(optSim, kinSeries, referenceSeries)
    #pdb.set_trace()

results = {
    'site_pos' : modelKin,
    'orig_site_pos': alignedKin,
    'qpos' : modelQpos
}

saveName = kinematicsFile.split('.')[0] + "_kinematics.pickle"
with open(saveName, 'wb') as f:
    pickle.dump(results, f)

print('Finished with %4.2f average function calls' % np.mean(statistics['nfev']))
print('Finished with %4.6f average reduced chisquare' % np.mean(statistics['redchi']))
