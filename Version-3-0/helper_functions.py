import pandas as pd
import numpy as np
import seaborn as sns
import itertools
from lmfit import Parameters, Parameter
from mujoco_py.generated import const
from mujoco_py import functions
import pdb, copy

def get_kinematics(kinematicsFile, selectHeaders = None, selectTime = None, reIndex = None):
    raw = pd.read_table(kinematicsFile, index_col = 0, skiprows = [1])
    raw.index = np.around(raw.index, 3)

    if selectTime:
        raw = raw.loc[slice(selectTime[0], selectTime[1]), :]
    headings = sorted(raw.columns) # get column names
    coordinates = ['x', 'y', 'z']

    # reorder alphabetically by columns
    raw = raw.reindex_axis(headings, axis = 1)

    if reIndex is not None:
        uniqueHeadings = set([name[:-2] for name in headings])
        oldIndex = raw.columns
        newIndex = np.array(oldIndex)
        for name in uniqueHeadings:
            for reindexPair in reIndex:
                mask = [
                    oldIndex.str.contains(name + ' ' + reindexPair[0]),
                    oldIndex.str.contains(name + ' ' + reindexPair[1])
                ]
                temp = newIndex[mask[0]]
                newIndex[mask[0]] = newIndex[mask[1]]
                newIndex[mask[1]] = temp
        raw.columns = newIndex
        headings = sorted(newIndex)
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
        raw = raw[sorted(expandedHeadings)]
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

    meshScale = 1.1e-3

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

def dict_to_params(jointDict, skip = []):
    params = Parameters()
    for key, value in jointDict.items():
        vary = True if key not in skip else False
        # silly workaround because Parameter() does not allow ':' in name
        # TODO: Fix
        key = key.replace(':', '_')
        params.add(key, value=value[0], min=value[1], max=value[2], vary = vary)
    return params

def params_to_dict(params):
    jointSeries = {}
    for key, value in params.valuesdict().items():
        # silly workaround because Parameter() does not allow ':' in name
        # TODO: fix
        key = key[::-1].replace('_', ':', 1)[::-1]
        jointSeries.update({key : value})
    return jointSeries

def params_to_series(params):
    return pd.Series(params_to_dict(params))

def pose_model(simulation, jointSeries, qAcc = None, qVel = None, method = 'forward'):
    simState = simulation.get_state()

    jointsDict = jointSeries.to_dict()
    for jointName in jointsDict.keys():
        jointId = simulation.model.get_joint_qpos_addr(jointName)
        simState.qpos[jointId] = jointSeries.loc[jointName]

    # make the changes to joint state
    simulation.set_state(simState)
    # advance the simulation one step
    if method == 'forward':
        simulation.forward()
    if method == 'step':
        simulation.step()
    if method == 'inverse':
        for idx, newValue in enumerate(qAcc):
            debugging = False
            if debugging:
                print('Changed simulation.data.qacc[' + str(idx) + '] from: ')
                print(simulation.data.qacc[idx])
                print('to:')
                print(newValue)

            simulation.data.qacc[idx] = newValue

            if debugging:
                print("it is now:")
                print(simulation.data.qacc[idx])
                print('Changed simulation.data.qvel[' + str(idx) + '] from: ')
                print(simulation.data.qvel[idx])
                print('to:')
                print(qVel[idx])

            #simulation.data.qvel[idx] = qVel[idx]
            if debugging:
                print("it is now:")
                print(simulation.data.qvel[idx])
        functions.mj_inverse(simulation.model, simulation.data)

    return simulation

def get_tendon_length(tendonSeries, simulation):
    for tendonName in tendonSeries.index:
        tendonSeries.loc[tendonName] = simulation.data.actuator_length[simulation.model.actuator_name2id(tendonName)]
    return tendonSeries

def get_site_pos(kinSeries, simulation):
    sitePos = pd.Series(index = kinSeries.index)

    for siteName in np.unique(kinSeries.index.get_level_values('joint')):

        siteXYZ = simulation.data.get_site_xpos(siteName)

        sitePos[(siteName, 'x')] = siteXYZ[0]
        sitePos[(siteName, 'y')] = siteXYZ[1]
        sitePos[(siteName, 'z')] = siteXYZ[2]

    return sitePos

def alignToModel(simulation, kinSeries, reference):
    modelSitePos = get_site_pos(kinSeries, simulation)

    alignedSitePos = copy.deepcopy(kinSeries)
    #e.g. kin[GT_left, x] = kin[GT_Left, x] - kin[Reference, x]
    for siteName in np.unique(alignedSitePos.index.get_level_values('joint')):
        alignedSitePos[(siteName, 'x')] = alignedSitePos[(siteName, 'x')] - alignedSitePos[(reference, 'x')] + modelSitePos[(reference, 'x')]
        alignedSitePos[(siteName, 'y')] = alignedSitePos[(siteName, 'y')] - alignedSitePos[(reference, 'y')] + modelSitePos[(reference, 'y')]
        alignedSitePos[(siteName, 'z')] = alignedSitePos[(siteName, 'z')] - alignedSitePos[(reference, 'z')] + modelSitePos[(reference, 'z')]

    return alignedSitePos

def render_targets(viewer, kinSeries):
    # Markers and overlay are regenerated in every pass.
    viewer._markers[:] = []
    viewer._overlay.clear()

    for siteName in np.unique(kinSeries.index.get_level_values('joint')):
        viewer.add_marker(
            pos=np.array(
                [kinSeries[(siteName, 'x')],
                kinSeries[(siteName, 'y')],
                kinSeries[(siteName, 'z')]
                ]),
            size = np.ones(3) * 1e-2,
            type = const.GEOM_SPHERE,
            label = siteName)

def series_to_markers(kinSeries):
    markers = []
    for siteName in np.unique(kinSeries.index.get_level_values('joint')):
        markers = markers + [{
            'label': siteName,
            'pos': np.array([kinSeries[(siteName, 'x')],
                kinSeries[(siteName, 'y')],
                kinSeries[(siteName, 'z')]
                ]),
            'size': np.ones(3) * 1e-2,
            'type': const.GEOM_SPHERE
            }]
    return markers

def Ia_model(k, tendonL, tendonV, tendonL0):
    iAFiring = pd.DataFrame(index = tendonL.index, columns = tendonL.columns)
    for t, tendonSeries in tendonL.iterrows():
        positiveV = tendonV.loc[t,:].clip(lower = 0)
        iAFiring.loc[t, :] = k*(positiveV*21*(positiveV/tendonL0)**0.5+200*(tendonL.loc[t,:]-tendonL0)/tendonL0+60)
    return iAFiring

def long_form_df(kinDF, overrideColumns = None):
    longDF = pd.DataFrame(kinDF.unstack())
    longDF.reset_index(inplace=True)
    if overrideColumns is not None:
        longDF.columns = overrideColumns
    return longDF
