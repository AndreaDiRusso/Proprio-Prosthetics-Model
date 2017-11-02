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
import pdb, copy

resourcesDir = 'C:/Users/adirusso/Documents/GitHub/Proprio-Prosthetics-Model/Version-3-0/Resources/Murdoc'

templateFilePath = 'C:/Users/adirusso/Documents/GitHub\Proprio-Prosthetics-Model/Version-3-0/murdoc_seated_template-Copy.xml'
fcsvFilePath = 'C:/Users/adirusso/Documents/GitHub/Proprio-Prosthetics-Model/Version-3-0/Resources/Murdoc/Aligned-To-Pelvis/Fiducials.fcsv'

specification = fcsv_to_spec(fcsvFilePath)
modelXML = populate_model(templateFilePath, specification, resourcesDir, showTendons = True)
