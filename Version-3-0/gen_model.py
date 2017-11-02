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
import pdb, copy, os
from helper_functions import *

curfilePath = os.path.abspath(__file__)
#print(curfilePath)
curDir = os.path.abspath(os.path.join(curfilePath,os.pardir)) # this will return current directory in which python file resides.
#print(curDir)
parentDir = os.path.abspath(os.path.join(curDir,os.pardir)) # this will return parent directory.
#print(parentDir)

resourcesDir = curDir + '/Resources/Murdoc'

templateFilePath = curDir + '/murdoc_template.xml'
fcsvFilePath = resourcesDir + '/Aligned-To-Pelvis/Fiducials.fcsv'


specification = fcsv_to_spec(fcsvFilePath)
modelXML = populate_model(templateFilePath, specification, resourcesDir, showTendons = True)
