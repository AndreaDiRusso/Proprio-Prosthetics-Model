import os, argparse, pickle
import pandas as pd
from helper_functions import *

parser = argparse.ArgumentParser()
parser.add_argument('--kinematicsFile', default = 'W:/ENG_Neuromotion_Shared/group/Proprioprosthetics/Data/201709261100-Proprio/T_1.txt')
args = parser.parse_args()

kinematicsFile = args.kinematicsFile
preproc_china_kinematics(kinematicsFile)
