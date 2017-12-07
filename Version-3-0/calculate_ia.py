import os, argparse, pickle, copy
import mujoco_py
import glfw
from mujoco_py import load_model_from_xml, MjSim, MjViewer
from mujoco_py.utils import rec_copy, rec_assign
from helper_functions import *

curfilePath = os.path.abspath(__file__)
#print(curfilePath)
curDir = os.path.abspath(os.path.join(curfilePath,os.pardir)) # this will return current directory in which python file resides.
#print(curDir)
parentDir = os.path.abspath(os.path.join(curDir,os.pardir)) # this will return parent directory.
#print(parentDir)

parser = argparse.ArgumentParser()
parser.add_argument('--kinematicsFile', default = 'W:/ENG_Neuromotion_Shared/group/Proprioprosthetics/Data/201709261100-Proprio/T_1_filtered_kinematics.pickle')
parser.add_argument('--modelFile', default = 'murdoc_template_toes_treadmill.xml')
parser.add_argument('--whichSide', default = 'Left')
parser.add_argument('--dt', default = '0.01')
parser.add_argument('--meshScale', default = '1.1e-3')
parser.add_argument('--showViewer', dest='showViewer', action='store_true')
parser.set_defaults(showViewer = False)

args = parser.parse_args()

kinematicsFile = args.kinematicsFile
modelFile = args.modelFile
meshScale = float(args.meshScale)
showViewer = args.showViewer
whichSide = args.whichSide
dt = float(args.dt)

with open(kinematicsFile, 'rb') as f:
    kinematics = pickle.load(f)

resourcesDir = curDir + '/Resources/Murdoc'
tendonNames = [
    'BF_' + whichSide,
    'TA_' + whichSide,
    'IL_' + whichSide,
    'RF_' + whichSide,
    'GMED_' + whichSide,
    'GAS_' + whichSide,
    'VAS_' + whichSide,
    'SOL_' + whichSide
    ]

resourcesDir = curDir + '/Resources/Murdoc'
templateFilePath = curDir + '/' + modelFile

fcsvFilePath = resourcesDir + '/Mobile Foot/Fiducials.fcsv'

specification = fcsv_to_spec(fcsvFilePath)

modelXML = populate_model(templateFilePath, specification, resourcesDir = resourcesDir,
    meshScale = meshScale, showTendons = True)

model = load_model_from_xml(modelXML)

functions.mj_setTotalmass(model, 10)
simulation = MjSim(model)

gains = [0.5, 160, 380, 7]
viewer = MjViewer(simulation)

#get resting lengths
nJoints = simulation.model.njnt
simulation = pose_to_key(simulation, 0)
tendonL0 = pd.Series(index = tendonNames)
tendonL0 = get_tendon_length(tendonL0, simulation)

tendonL = pd.DataFrame(columns = tendonNames)
for t, kinSeries in kinematics['site_pos'].iterrows():

    jointDict = series_to_dict( kinematics['qpos'].loc[t, :])
    pose_model(simulation, jointDict)
    tendonL.loc[t, :] = get_tendon_length(tendonL0, simulation)

    if showViewer:
        viewer.render()

tendonV = tendonL.diff(axis = 0).fillna(0) / dt
iARate = Ia_model_Radu(tendonL, tendonV,  tendonL0, gains)

kinematics.update(
    {
        'tendonL0': tendonL0,
        'tendonL': tendonL,
        'tendonV': tendonV,
        'iARate' : iARate
        }
    )

newName = kinematicsFile.split('_kinematics')[0] + '_fr.pickle'
with open(newName, 'wb') as f:
    pickle.dump(kinematics, f)
