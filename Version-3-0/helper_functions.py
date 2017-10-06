import pandas as pd
import numpy as np
import seaborn as sns
import itertools
from lmfit import Parameters, Parameter

def get_kinematics(kinematicsFile, selectHeaders = None, selectTime = None):
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

    tendonAlpha = 1 if showTendons else 0
    modelXML = modelXML.replace('$resourcesDir$', resourcesDir)
    modelXML = modelXML.replace('$showTendons$', str(tendonAlpha))

    for idx, row in specification.iterrows():

        if ":x" in row['label'] or ":y" in row['label'] or ':z' in row['label']:
            jointAxis = row['label'].split(':')[-1][0]
            jointName = row['label'].split(':')[0]
            originName = jointName + ':o'
            row['x'] = - row['x'] + specification[specification['label'] == originName]['x'].values[0]
            row['y'] = - row['y'] + specification[specification['label'] == originName]['y'].values[0]
            row['z'] = - row['z'] + specification[specification['label'] == originName]['z'].values[0]

        placeHolder = '$' + row['label'] + ':' + 'x$'
        modelXML = modelXML.replace(placeHolder, str(row['x']*1e-3))

        placeHolder = '$' + row['label'] + ':' + 'y$'
        modelXML = modelXML.replace(placeHolder, str(row['y']*1e-3))

        placeHolder = '$' + row['label'] + ':' + 'z$'
        modelXML = modelXML.replace(placeHolder, str(row['z']*1e-3))

    templateDir = '/'.join(templateFilePath.split('/')[:-1])
    with open(templateDir + '/murdoc_gen.xml', 'w') as f:
        f.write(modelXML)

    return modelXML

def list_to_params(jointList):
    params = Parameters()
    for key, value in jointList.items():
        # silly workaround because Parameter() does not allow ':' in name
        # TODO: Fix
        key = key.replace(':', '_')
        params.add(key, value=value[0], min=value[1], max=value[2])
    return params
