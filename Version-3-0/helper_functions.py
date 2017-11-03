import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import itertools
from lmfit import Parameters, Parameter
from mujoco_py.generated import const
from mujoco_py import functions
import pdb, copy
from mpl_toolkits.mplot3d import Axes3D
import math
import quaternion as quat

def get_kinematics(kinematicsFile, selectHeaders = None, selectTime = None,
    flip = None, reIndex = None):
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

    if flip is not None:
        uniqueHeadings = set([name[:-2] for name in headings])
        for name in uniqueHeadings:
            for flipAxis in flip:
                raw.loc[:,name + ' ' + flipAxis] = -raw.loc[:,name + ' ' + flipAxis]

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
        indexPD = pd.MultiIndex.from_product([sorted(selectHeaders),
            coordinates], names=['joint', 'coordinate'])
    else:
        uniqueHeadings = set([name[:-2] for name in headings])
        indexPD = pd.MultiIndex.from_product([sorted(uniqueHeadings),
            coordinates], names=['joint', 'coordinate'])

    proc = pd.DataFrame(raw.values, columns = indexPD, index = raw.index)

    return proc

def preproc_china_kinematics(kinematicsFile):
    raw = pd.read_table(kinematicsFile, index_col = 1)
    assert raw.index.name == 'Time'

    raw.drop(['FRAME'], axis = 1, inplace = True)

    lookupTable = {
        'left scapula X' : 'S_Left X',
        'left scapula Y' : 'S_Left Y',
        'left scapula Z' : 'S_Left Z',
        'left humerus head (H) X' : 'H_Left X',
        'left humerus head (H) Y' : 'H_Left Y',
        'left humerus head (H) Z' : 'H_Left Z',
        'left elbow joint (E) X' : 'E_Left X',
        'left elbow joint (E) Y' : 'E_Left Y',
        'left elbow joint (E) Z' : 'E_Left Z',
        'left distal head of ulna (U) X' : 'U_Left X',
        'left distal head of ulna (U) Y' : 'U_Left Y',
        'left distal head of ulna (U) Z' : 'U_Left Z',
        'Metacarpo-phalangeal (MCP) X' : 'MCP_Left X',
        'Metacarpo-phalangeal (MCP) Y' : 'MCP_Left Y',
        'Metacarpo-phalangeal (MCP) Z' : 'MCP_Left Z',
        'left tip of the 3rd digit (D) X' : 'D_Left X',
        'left tip of the 3rd digit (D) Y' : 'D_Left Y',
        'left tip of the 3rd digit (D) Z' : 'D_Left Z',
        'left crest X' : 'C_Right X',
        'left crest Y' : 'C_Right Y',
        'left crest Z' : 'C_Right Z',
        'left trochanter major (GT) X' : 'GT_Right X',
        'left trochanter major (GT) Y' : 'GT_Right Y',
        'left trochanter major (GT) Z' : 'GT_Right Z',
        'left knee-joint (K) X' : 'K_Right X',
        'left knee-joint (K) Y' : 'K_Right Y',
        'left knee-joint (K) Z' : 'K_Right Z',
        'left malleolus (M) X' : 'M_Right X',
        'left malleolus (M) Y' : 'M_Right Y',
        'left malleolus (M) Z' : 'M_Right Z',
        'left 5th metatarsal (MT) X' : 'MT_Right X',
        'left 5th metatarsal (MT) Y' : 'MT_Right Y',
        'left 5th metatarsal (MT) Z' : 'MT_Right Z',
        'left outside tip of 5th digit (T) X' : 'T_Right X',
        'left outside tip of 5th digit (T) Y' : 'T_Right Y',
        'left outside tip of 5th digit (T) Z' : 'T_Right Z',
        'right scapula X' : 'S_Right',
        'right scapula Y' : 'S_Right',
        'right scapula Z' : 'S_Right',
        'right humerus head (H) X' : 'H_Right X',
        'right humerus head (H) Y' : 'H_Right Y',
        'right humerus head (H) Z' : 'H_Right Z',
        'right elbow joint (E) X' : 'E_Right X',
        'right elbow joint (E) Y' : 'E_Right Y',
        'right elbow joint (E) Z' : 'E_Right Z',
        'right distal head of ulna (U) X' : 'U_Right X',
        'right distal head of ulna (U) Y' : 'U_Right Y',
        'right distal head of ulna (U) Z' : 'U_Right Z',
        #'Metacarpo-phalangeal (MCP) X' : 'MCP_Right X',
        #'Metacarpo-phalangeal (MCP) Y' : 'MCP_Right Y',
        #'Metacarpo-phalangeal (MCP) Z' : 'MCP_Right Z',
        'right tip of the 3rd digit (D) X' : 'D_Right X',
        'right tip of the 3rd digit (D) Y' : 'D_Right Y',
        'right tip of the 3rd digit (D) Z' : 'D_Right Z',
        'right crest X' : 'C_Left X',
        'right crest Y' : 'C_Left Y',
        'right crest Z' : 'C_Left Z',
        'right trochanter major (GT) X' : 'GT_Left X',
        'right trochanter major (GT) Y' : 'GT_Left Y',
        'right trochanter major (GT) Z' : 'GT_Left Z',
        'right knee-joint (K) X' : 'K_Left X',
        'right knee-joint (K) Y' : 'K_Left Y',
        'right knee-joint (K) Z' : 'K_Left Z',
        'right malleolus (M) X' : 'M_Left X',
        'right malleolus (M) Y' : 'M_Left Y',
        'right malleolus (M) Z' : 'M_Left Z',
        'right 5th metatarsal (MT) X' : 'MT_Left X',
        'right 5th metatarsal (MT) Y' : 'MT_Left Y',
        'right 5th metatarsal (MT) Z' : 'MT_Left Z',
        'right outside tip of 5th digit (T) X' : 'T_Left X',
        'right outside tip of 5th digit (T) Y' : 'T_Left Y',
        'right outside tip of 5th digit (T) Z' : 'T_Left Z'
        }

    raw.rename(columns = lookupTable, inplace = True)
    newName = '.'.join(kinematicsFile.split('.')[:-1]) + '_processed.txt'
    raw.to_csv(newName, sep='\t')

def fcsv_to_spec(fcsvFilePath):
    fcsv = pd.read_csv(fcsvFilePath, skiprows = 2)
    fcsv = fcsv.loc[:, ['label', 'x', 'y', 'z']]
    return fcsv

def populate_model(templateFilePath, specification, resourcesDir,
    meshScale = 1.1e-3, showTendons = False):

    with open(templateFilePath, 'r') as f:
        modelXML = f.read()

    tendonAlpha = 1 if showTendons else 0
    modelXML = modelXML.replace('$resourcesDir$', resourcesDir)
    modelXML = modelXML.replace('$showTendons$', str(tendonAlpha))
    modelXML = modelXML.replace('$meshScale$', str(meshScale))

    for idx, row in specification.iterrows():

        if ':x' in row['label'] or ':y' in row['label'] or ':z' in row['label']:
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
        params.add(key, value=value['value'], min=value['min'], max=value['max'], vary = vary)
    return params

def params_to_dict(params):
    jointDict = {}
    for key, value in params.valuesdict().items():
        # silly workaround because Parameter() does not allow ':' in name
        # TODO: fix
        key = key[::-1].replace('_', ':', 1)[::-1]
        jointDict.update({key: {'value': None, 'min' : None, 'max' : None}})
        jointDict[key].update({'value' : value})
    return jointDict

def params_to_series(params):
    jointDictAll = params_to_dict(params)
    jointDict = {key : value['value'] for key, value in jointDictAll.items()}
    return pd.Series(jointDict)

def series_to_dict(jointSeries):
    jointDict = { key : {'value': value} for key, value in jointSeries.to_dict().items()}
    return jointDict

def pose_model(simulation, jointDict, qAcc = None,
    qVel = None, method = 'forward'):

    simState = simulation.get_state()

    quatJointNames = set([])
    for jointName in jointDict.keys():
        try:
            #print(jointName)
            jointId = simulation.model.get_joint_qpos_addr(jointName)
            #print(jointId)
            simState.qpos[jointId] = jointDict[jointName]['value']
        except:
            # probably one of the quaternion joints
            quatJointName = jointName[:-3]
            quatJointNames.add(quatJointName)

    for quatJointName in quatJointNames:

        jointId = simulation.model.get_joint_qpos_addr(quatJointName)

        rotation = quat.from_euler_angles(
            jointDict[quatJointName + ':xq']['value'],
            jointDict[quatJointName + ':yq']['value'],
            jointDict[quatJointName + ':zq']['value']
            )

        simState.qpos[jointId[0]    ] = jointDict[quatJointName + ':xt']['value']
        
        simState.qpos[jointId[0] + 1] = jointDict[quatJointName + ':yt']['value']
        
        simState.qpos[jointId[0] + 2] = jointDict[quatJointName + ':zt']['value']

        simState.qpos[jointId[0] + 3] = rotation.normalized().w

        simState.qpos[jointId[0] + 4] = rotation.normalized().x

        simState.qpos[jointId[0] + 5] = rotation.normalized().y

        simState.qpos[jointId[0] + 6] = rotation.normalized().z

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

def alignToModel(simulation, kinSeries, referenceSeries):
    alignedSitePos = copy.deepcopy(kinSeries)

    #e.g. kin[GT_left, x] = kin[GT_Left, x] - kin[Reference, x]
    referenceName = np.unique(referenceSeries.index.get_level_values('joint'))[0]

    for siteName in np.unique(alignedSitePos.index.get_level_values('joint')):
        alignedSitePos[(siteName, 'x')] = alignedSitePos[(siteName, 'x')] + referenceSeries[(referenceName, 'x')]
        alignedSitePos[(siteName, 'y')] = alignedSitePos[(siteName, 'y')] + referenceSeries[(referenceName, 'y')]
        alignedSitePos[(siteName, 'z')] = alignedSitePos[(siteName, 'z')] + referenceSeries[(referenceName, 'z')]

    return alignedSitePos

def plot_sites_3D(kinDF, useRange = slice(None)):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for siteName in np.unique(kinDF.columns.get_level_values('joint')):
        ax.plot(kinDF[(siteName, 'x')].values[useRange],
            kinDF[(siteName, 'y')].values[useRange],
            kinDF[(siteName, 'z')].values[useRange], marker = 'o')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
    plt.show()

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

def get_euler_rotation_quaternion(jointsToFit, jointName):
    q = quat.from_euler_angles(jointsToFit[jointName + ':x']['value'],
        jointsToFit[jointName + ':y']['value'],
        jointsToFit[jointName + ':z']['value']
        )
    return q

def contact_summary(simulation, debugging = False):
    activeContacts = []
    for idx, contact in enumerate(simulation.data.contact):
        contactForce = np.zeros((6))
        functions.mj_contactForce(simulation.model, simulation.data, idx, contactForce)
        if np.sum(contactForce**2) > 0:
            activeContacts.append( {
                'contactIdx' : idx,
                'contactForce' : contactForce
                } )
            if debugging:
                print('Contact geom 1:')
                print(simulation.model.geom_id2name(contact.geom1))
                print('Contact geom 2:')
                print(simulation.model.geom_id2name(contact.geom2))
                print('Contact Force:')
                print(contactForce)
                print('------------------------------------------')
    return activeContacts
