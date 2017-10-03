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
