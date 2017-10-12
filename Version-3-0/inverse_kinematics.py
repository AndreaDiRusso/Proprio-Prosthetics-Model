from lmfit import minimize, Minimizer, Parameters, Parameter, report_fit
from helper_functions import *
import pdb

def iter_cb(params, iter, resid, t, kinSeries, solver):
    print("Iteration number: %d" % iter, end = '\r')
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

    def __init__(self, simulation, sitesToFit, jointsToFit, alignTo = None, mjViewer = None, method = 'leastsq'):

        # which site to align model and data to
        if alignTo is None:
            self.alignTo = sitesToFit[0]
        else:
            self.alignTo = alignTo

        self.mjViewer = mjViewer
        # simulation to use
        self.simulation = simulation
        # joints to vary in order to fit
        # as set of Parameters
        self.jointsParam = dict_to_params(jointsToFit)
        self.jointsDict = jointsToFit
        # sites to check for fit
        self.sitesToFit = sitesToFit
        # optimization method
        self.method = method

        if method == 'nelder':
            self.nelderTol = 1e-7

    def joint_pos2site_pos(self, jointSeries, kinSeries):

        self.simulation = pose_model(self.simulation, jointSeries)

        # get resulting changes
        sitePos = get_site_pos(kinSeries, self.simulation)

        return sitePos

    def fit(self, t, kinSeries, mjViewer = None):

        methodParams = {}
        if self.method == 'leastsq':
            methodParams.update({
                'ftol': 1e-8,
                'xtol': 1e-8
            })

        if self.method == 'nelder':
            methodParams.update({
                'tol': self.nelderTol
            })

        minner = Minimizer(fcn2min, self.jointsParam,
            fcn_args=(t, kinSeries, self), iter_cb=iter_cb, **methodParams)
        return minner.minimize(method = self.method)
