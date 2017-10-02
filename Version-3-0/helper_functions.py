import pandas as pd
import numpy as np
import seaborn as sns
import itertools

def get_kinematics(filePath):
    raw = pd.read_table(filePath, index_col = 0, skiprows = [1])

    headings = list(raw)
    uniqueHeadings = set([name[:-2] for name in headings])
    coordinates = ['x', 'y', 'z']

    indexPD = pd.MultiIndex.from_product([sorted(uniqueHeadings), coordinates], names=['joint', 'coordinate'])
    proc = pd.DataFrame(index = indexPD).T

    for idx, row in raw.iterrows():
        for col, value in row.iteritems():
            proc.loc[idx, (col[:-2], col[-1].lower())] = value
