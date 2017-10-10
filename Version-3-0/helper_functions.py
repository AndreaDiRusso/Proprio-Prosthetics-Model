import pandas as pd
import numpy as np
import seaborn as sns
import itertools
from lmfit import Parameters, Parameter
from mujoco_py.generated import const
import pdb, copy

def get_kinematics(kinematicsFile, selectHeaders = None, selectTime = None, reIndex = None):
    raw = pd.read_table(kinematicsFile, index_col = 0, skiprows = [1])
    if selectTime:
        raw = raw.loc[slice(selectTime[0], selectTime[1]), :]
    headings = sorted(raw.columns) # get column names
    coordinates = ['x', 'y', 'z']

    # reorder alphabetically by columns
    raw = raw.reindex_axis(headings, axis = 1)
    # create multiIndex column names
    if selectHeaders is not None:
        expandedHeadings = [
            name + ' X' for name in selectHeaders
        ] + [
            name + ' Y' for name in selectHeaders
        ] + [
            name + ' Z' for name in selectHeaders
        ]
        raw = raw[expandedHeadings]
        indexPD = pd.MultiIndex.from_product([sorted(selectHeaders), coordinates], names=['joint', 'coordinate'])
    else:
        uniqueHeadings = set([name[:-2] for name in headings])
        indexPD = pd.MultiIndex.from_product([sorted(uniqueHeadings), coordinates], names=['joint', 'coordinate'])

    proc = pd.DataFrame(raw.values, columns = indexPD, index = raw.index)

    return proc

def fcsv_to_spec(fcsvFilePath):
    fcsv = pd.read_csv(fcsvFilePath, skiprows = 2)
    fcsv = fcsv.loc[:, ['label', 'x', 'y', 'z']]
    return fcsv

def populate_model(templateFilePath, specification, resourcesDir, showTendons = False):

    with open(templateFilePath, 'r') as f:
        modelXML = f.read()

    meshScale = 1e-3

    tendonAlpha = 1 if showTendons else 0
    modelXML = modelXML.replace('$resourcesDir$', resourcesDir)
    modelXML = modelXML.replace('$showTendons$', str(tendonAlpha))
    modelXML = modelXML.replace('$meshScale$', str(meshScale))

    for idx, row in specification.iterrows():

        if ":x" in row['label'] or ":y" in row['label'] or ':z' in row['label']:
            jointAxis = row['label'].split(':')[-1][0]
            jointName = row['label'].split(':')[0]
            originName = jointName + ':o'
            row['x'] = - row['x'] + specification[specification['label'] == originName]['x'].values[0]
            row['y'] = - row['y'] + specification[specification['label'] == originName]['y'].values[0]
            row['z'] = - row['z'] + specification[specification['label'] == originName]['z'].values[0]

        placeHolder = '$' + row['label'] + ':' + 'x$'
        modelXML = modelXML.replace(placeHolder, str(row['x']*meshScale))

        placeHolder = '$' + row['label'] + ':' + 'y$'
        modelXML = modelXML.replace(placeHolder, str(row['y']*meshScale))

        placeHolder = '$' + row['label'] + ':' + 'z$'
        modelXML = modelXML.replace(placeHolder, str(row['z']*meshScale))

    templateDir = '/'.join(templateFilePath.split('/')[:-1])
    with open(templateDir + '/murdoc_gen.xml', 'w') as f:
        f.write(modelXML)

    return modelXML

def dict_to_params(jointDict):
    params = Parameters()
    for key, value in jointDict.items():
        # silly workaround because Parameter() does not allow ':' in name
        # TODO: Fix
        key = key.replace(':', '_')
        params.add(key, value=value[0], min=value[1], max=value[2])
    return params

def params_to_series(params):
    jointSeries = {}
    for key, value in params.valuesdict().items():
        # silly workaround because Parameter() does not allow ':' in name
        # TODO: fix
        key = key[::-1].replace('_', ':', 1)[::-1]
        jointSeries.update({key : value})

    jointSeries = pd.Series(jointSeries)
    return jointSeries

def pose_model(simulation, jointsToFit, jointSeries):
    simState = simulation.get_state()

    for jointName in jointsToFit.keys():
        jointId = simulation.model.get_joint_qpos_addr(jointName)
        simState.qpos[jointId] = jointSeries.loc[jointName]

    # make the changes to joint state
    simulation.set_state(simState)
    # advance the simulation one step
    simulation.forward()

    return simulation

def get_site_pos(kinSeries, simulation):
    sitePos = pd.Series(index = kinSeries.index)

    for siteName in np.unique(kinSeries.index.get_level_values('joint')):

        siteXYZ = simulation.data.get_site_xpos(siteName)

        sitePos.loc[(siteName, 'x')] = siteXYZ[0]
        sitePos.loc[(siteName, 'y')] = siteXYZ[1]
        sitePos.loc[(siteName, 'z')] = siteXYZ[2]

    return sitePos

def alignToModel(simulation, kinSeries, reference):
    modelSitePos = get_site_pos(kinSeries, simulation)

    alignedSitePos = copy.deepcopy(kinSeries)
    for siteName in np.unique(alignedSitePos.index.get_level_values('joint')):
        #e.g. kin[GT_left, x] = kin[GT_Left, x] - kin[Reference, x]
        alignedSitePos.loc[(siteName, 'x')] = alignedSitePos.loc[(siteName, 'x')] - alignedSitePos.loc[(reference, 'x')] + modelSitePos.loc[(reference, 'x')]
        alignedSitePos.loc[(siteName, 'y')] = alignedSitePos.loc[(siteName, 'y')] - alignedSitePos.loc[(reference, 'y')] + modelSitePos.loc[(reference, 'y')]
        alignedSitePos.loc[(siteName, 'z')] = alignedSitePos.loc[(siteName, 'z')] - alignedSitePos.loc[(reference, 'z')] + modelSitePos.loc[(reference, 'z')]

    return alignedSitePos

def render_targets(viewer, kinSeries):
    for siteName in np.unique(kinSeries.index.get_level_values('joint')):
        viewer.add_marker(
            pos=np.array([kinSeries.loc[(siteName, 'x')],
                kinSeries.loc[(siteName, 'y')],
                kinSeries.loc[(siteName, 'z')]]),
            size = np.ones(3) * 1e-2,
            label = siteName)

    viewer.render()
