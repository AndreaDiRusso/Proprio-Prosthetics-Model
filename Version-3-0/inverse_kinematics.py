from lmfit import minimize, Minimizer, Parameters, Parameter, report_fit
from helper_functions import *
import pdb
from mujoco_py import functions

def iter_cb(params, iterNo, resid, t, kinSeries, solver):
    printing = True
    if printing:
        try:
            print("Iteration number: %d" % iterNo, end = '\r')
        except:
            pass

    if solver.mjViewer is not None:
        render_targets(solver.mjViewer, alignToModel(solver.simulation, kinSeries, solver.alignTo))
        solver.mjViewer.render()
    #print("Residual array:")
    #print(resid)
    #print("params: ")
    #print(params)
    #print("SSQ: ")
    #print(np.sum(resid**2))
    pass

# define objective function: returns the array to be minimized
# params are the qpos
# x is time, not used in the calculation
# data is the true value
def fcn2min(params, t, kinSeries, solver):
    """ get kinematics from simulation, subtract aligned data"""

    jointSeries = params_to_series(params)
    kinSeries = alignToModel(solver.simulation, kinSeries, solver.alignTo)
    modelSeries = solver.joint_pos2site_pos(jointSeries, kinSeries)
    difference = kinSeries - modelSeries

    return difference.values

class IKFit:

    def __init__(self, simulation, sitesToFit, jointsToFit, skipThese = [],
        alignTo = None, mjViewer = None, method = 'leastsq',
        simulationType = 'forward'):

        # which site to align model and data to
        if alignTo is None:
            self.alignTo = None
        else:
            self.alignTo = alignTo

        self.mjViewer = mjViewer
        # simulation to use
        self.simulation = simulation
        # joints to vary in order to fit
        # as set of Parameters
        self.jointsParam = dict_to_params(jointsToFit, skip = skipThese)
        self.jointsDict = jointsToFit
        # sites to check for fit
        self.sitesToFit = sitesToFit
        # optimization method
        self.method = method
        self.simulationType = simulationType

        if method == 'nelder':
            self.nelderTol = 1e-7
        if method == 'leastsq':
            self.fTol = 1e-8

    def joint_pos2site_pos(self, jointSeries, kinSeries):

        self.simulation = pose_model(self.simulation, jointSeries,
            method = self.simulationType)
        # get resulting changes
        sitePos = get_site_pos(kinSeries, self.simulation)

        return sitePos

    def fit(self, t, kinSeries, mjViewer = None):

        methodParams = {}
        if self.method == 'leastsq':
            methodParams.update({
                'ftol': self.fTol,
                'xtol': 1e-8
            })

        if self.method == 'nelder':
            methodParams.update({
                'tol': self.nelderTol
            })

        minner = Minimizer(fcn2min, self.jointsParam,
            fcn_args=(t, kinSeries, self), iter_cb=iter_cb, **methodParams)
        stats = minner.minimize(method = self.method)
        return stats
