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
parser.add_argument('--modelKinematicsFile', default = 'Z:\\ENG_Neuromotion_Shared\\group\\Proprioprosthetics\\Data\\201709261100-Proprio\\T_1_model.pickle')
parser.add_argument('--modelFile', default = 'murdoc_gen.xml')
parser.add_argument('--showViewer', dest='showViewer', action='store_true')
parser.set_defaults(showViewer = False)

args = parser.parse_args()

modelKinematicsFile = args.modelKinematicsFile
modelFile = args.modelFile
showViewer = args.showViewer

resourcesDir = curDir + '/Resources/Murdoc'
tendonNames = [
    'BF_Left',
    'TA_Left',
    'IL_Left',
    'RF_Left',
    'GMED_Left',
    'GAS_Left',
    'VAS_Left',
    'SOL_Left'
    ]

with open(curDir + '/' + modelFile, 'r') as f:
    model = load_model_from_xml(f.read())

with open(modelKinematicsFile, 'rb') as f:
    kinematics = pickle.load(f)

simulation = MjSim(model)

viewer = MjViewer(simulation)

#get resting lengths
nJoints = simulation.model.key_qpos.shape[0]
allJoints = [simulation.model.joint_id2name(i) for i in range(nJoints)]
keyPos = pd.Series({jointName: simulation.model.key_qpos[0][i] for i, jointName in enumerate(allJoints)})

pose_model(simulation, keyPos)

tendonL0 = pd.Series(index = tendonNames)
tendonL0 = get_tendon_length(tendonL0, simulation)

tendonL = pd.DataFrame(columns = tendonNames)
for t, kinSeries in kinematics['site_pos'].iterrows():

    pose_model(simulation, kinematics['qpos'].loc[t, :])
    tendonL.loc[t, :] = get_tendon_length(tendonL0, simulation)

    if showViewer:
        viewer.render()

tendonV = tendonL.diff(axis = 0).fillna(0)
iARate = Ia_model(1, tendonL, tendonV,  tendonL0)

kinematics.update(
    {
        'tendonL0': tendonL0,
        'tendonL': tendonL,
        'tendonV': tendonV,
        'iARate' : iARate
        }
    )

newName = modelKinematicsFile.split('.')[0] + '_fr.pickle'
with open(newName, 'wb') as f:
    pickle.dump(kinematics, f)
