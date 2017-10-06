from lmfit import minimize, Minimizer, Parameters, Parameter, report_fit
from helper_functions import *

class IKFit:

    def __init__(self, simulation, sitesToFit, jointsToFit, alignTo = None):

        # which site to align model and data to
        if alignTo is None:
            self.alignTo = sitesToFit[0]
        else:
            self.alignTo = alignTo

        # simulation to use
        self.simulation = simulation
        # joints to vary in order to fit
        self.jointsToFit = jointsToFit
        # sites to check for fit
        self.sitesToFit = sitesToFit

    def joint_pos2site_pos(self, qpos, kin, restoreOriginal = False):

        simState = self.simulation.get_state()

        if restoreOriginal:
            # save original simulation state
            originalSimState = simState

        for jointName in self.jointsToFit.keys():
            jointId = self.simulation.model.get_joint_qpos_addr(jointName)
            simState.qpos[jointId] = qpos.loc[jointName]

        # make the changes to joint state
        self.simulation.set_state(simState)
        # advance the simulation one step
        self.simulation.forward()

        # get resulting changes
        sitePos = pd.Series(index = kin.index)
        simState = self.simulation.get_state()

        for siteName in self.sitesToFit:
            siteId = self.simulation.model.site_name2id(siteName)
            siteXYZ = self.simulation.model.site_pos[siteId]

            sitePos.loc[(siteName, 'x')] = siteXYZ[0]
            sitePos.loc[(siteName, 'y')] = siteXYZ[1]
            sitePos.loc[(siteName, 'z')] = siteXYZ[2]

        if restoreOriginal:
            # Reset the simulation
            self.simulation.set_state(originalSimState)
            # advance the simulation one step
            self.simulation.step()

        return sitePos

    # define objective function: returns the array to be minimized
    # params are the qpos
    # x is time, not used in the calculation
    # data is the true value

    def fcn2min(self, params, t, data):
        """ get kinematics from simulation, subtract aligned data"""

        qpos = {}
        for key, value in params.valuesdict().items():
            # silly workaround because Parameter() does not allow ':' in name
            # TODO: fix
            key = key[::-1].replace('_', ':', 1)[::-1]
            qpos.update({key : value})

        qpos = pd.Series(qpos)

        data = self.alignToModel(data, self.alignTo)
        model = self.joint_pos2site_pos(qpos, data)
        difference = model - data

        return difference.values

    def fit(self, t, data):
        # create a set of Parameters
        jointsParam = list_to_params(self.jointsToFit)

        # do fit, here with leastsq model
        minner = Minimizer(self.fcn2min, jointsParam, fcn_args=(t, data))
        return minner.minimize()

    def alignToModel(self, kin, reference):
        for idx, value in kin.iteritems():
            #e.g. kin[GT_left, x] = kin[GT_Left, x] - kin[Reference, x]
            kin[idx] = kin[idx] - kin[(reference, idx[1])]
        return kin
