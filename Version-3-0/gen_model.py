# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 13:55:14 2017

@author: adirusso
"""
import pandas as pd
import numpy as np
import seaborn as sns
import itertools
from lmfit import Parameters, Parameter
from mujoco_py.generated import const
import pdb, copy, os, argparse
from helper_functions import *
from constants import *
from mujoco_py import load_model_from_xml, MjSim, MjViewerBasic, MjViewer, functions

parser = argparse.ArgumentParser()
parser.add_argument('--modelFile', default = 'murdoc_template.xml')
parser.add_argument('--meshScale', default = '1.1e-3')
#meshScale = 1.1e-3
args = parser.parse_args()
modelFile = args.modelFile
#modelFile = 'murdoc_template_seat.xml'
meshScale = float(args.meshScale)


curfilePath = os.path.abspath(__file__)
#print(curfilePath)
curDir = os.path.abspath(os.path.join(curfilePath,os.pardir)) # this will return current directory in which python file resides.
#curDir = 'C:/Users/Radu/Documents/GitHub/Proprio-Prosthetics-Model/Version-3-0'

parentDir = os.path.abspath(os.path.join(curDir,os.pardir)) # this will return parent directory.
#print(parentDir)

resourcesDir = curDir + '/Resources/Murdoc'

templateFilePath = curDir + '/' + modelFile
fcsvFilePath = resourcesDir + '/Mobile Foot/Fiducials.fcsv'

specification = fcsv_to_spec(fcsvFilePath)
modelXML = populate_model(templateFilePath, specification, extraLocations = {},
    resourcesDir = resourcesDir, meshScale = meshScale, showTendons = True)

printStats = True
if printStats:
    model = load_model_from_xml(modelXML)

    functions.mj_setTotalmass(model, 10)
    simulation = MjSim(model)
    simulation = pose_to_key(simulation, 0)

    tendonNames = [
        'BF_Right',
        'TA_Right',
        'IL_Right',
        'RF_Right',
        'GMED_Right',
        'GAS_Right',
        'VAS_Right',
        'SOL_Right'
        ]
    tendonL0 = pd.Series(index = tendonNames)
    tendonL0 = get_tendon_length(tendonL0, simulation)

    for idx, geomSize in enumerate(model.geom_size):
        geomName = model.geom_id2name(idx)
        if geomName:
            print(geomName + ':')
            print(geomSize)

    print(tendonL0)
