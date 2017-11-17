from helper_functions import *
import os, argparse

curfilePath = os.path.abspath(__file__)
#print(curfilePath)
curDir = os.path.abspath(os.path.join(curfilePath,os.pardir)) # this will return current directory in which python file resides.
parser = argparse.ArgumentParser()
parser.add_argument('--kinematicsFile', default = 'W:/ENG_Neuromotion_Shared/group/Proprioprosthetics/Data/201709261100-Proprio/T_1.txt')
parser.add_argument('--reIndex', dest='reIndex', type = tuple, nargs = 1)

args = parser.parse_args()

kinematicsFile = args.kinematicsFile
reIndex = args.reIndex

whichSide = 'Right'
sitesToFit = ['MT_' + whichSide, 'M_' + whichSide, 'C_' + whichSide, 'GT_' + whichSide, 'K_' + whichSide]

kinematics = get_kinematics(kinematicsFile,
    selectHeaders = sitesToFit, reIndex = reIndex)

useRange = range(5)
plot_sites_3D(kinematics, useRange = [0])
