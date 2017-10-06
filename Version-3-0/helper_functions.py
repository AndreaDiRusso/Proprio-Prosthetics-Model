import pandas as pd
import numpy as np
import seaborn as sns
import itertools

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

def fscv_to_spec(fcsvFilePath):
    fcsv = pd.read_csv(fcsvFilePath, skiprows = 2)
    fcsv = fcsv.loc[:, ['label', 'x', 'y', 'z']]
    return fcsv

def populate_model(templateFilePath, specification):
    fcsv = fscv_to_spec(fcsvFilePath)

    with open(templateFilePath, 'r') as f:
        modelXML = f.read()

    modelXML = modelXML.replace('$Murdoc_Dir$', curDir + '/Resources/Murdoc')
    for idx, row in fcsv.iterrows():

        if ":x" in row['label'] or ":y" in row['label'] or ':z' in row['label']:
            jointAxis = row['label'].split(':')[-1][0]
            jointName = row['label'].split(':')[0]
            originName = jointName + ':o'
            row['x'] = - row['x'] + fcsv[fcsv['label'] == originName]['x'].values[0]
            row['y'] = - row['y'] + fcsv[fcsv['label'] == originName]['y'].values[0]
            row['z'] = - row['z'] + fcsv[fcsv['label'] == originName]['z'].values[0]

        placeHolder = '$' + row['label'] + ':' + 'x$'
        modelXML = modelXML.replace(placeHolder, str(row['x']*1e-3))

        placeHolder = '$' + row['label'] + ':' + 'y$'
        modelXML = modelXML.replace(placeHolder, str(row['y']*1e-3))

        placeHolder = '$' + row['label'] + ':' + 'z$'
        modelXML = modelXML.replace(placeHolder, str(row['z']*1e-3))

    with open(curDir + '/murdoc_gen.xml', 'w') as f:
        f.write(modelXML)
