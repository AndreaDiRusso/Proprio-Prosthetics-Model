from lmfit import minimize, Minimizer, Parameters, Parameter, report_fit
from helper_functions import *
import pdb
from collections import deque
from mujoco_py import functions

def iter_cb(params, iterNo, resid, t, kinSeries, solver):
    print("Iteration number: %d" % iterNo, end = '\r')
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

    def __init__(self, simulation, sitesToFit, jointsToFit, alignTo = None,
        mjViewer = None, method = 'leastsq', simulationType = 'forward'):

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
        self.simulationType = simulationType

        bufferSize = 3

        dummyTime = deque(np.tile([self.simulation.data.time], (bufferSize,1)))
        dummyQVel = deque(np.tile(self.simulation.data.qvel, (bufferSize,1)))
        dummyQPos = deque(np.tile(self.simulation.data.qpos, (bufferSize,1)))

        self.buffer = {
            'time': dummyTime,
            'qvel': dummyQVel,
            'qpos': dummyQPos
        }

        if method == 'nelder':
            self.nelderTol = 1e-7

    def joint_pos2site_pos(self, jointSeries, kinSeries):
        #TODO: Consider moving this to the fit command,
        #such that qacc is only calculated in between real movements
        qAcc = None
        if self.simulationType == 'inverse':
            # calculate qAcc and pass to pose model
            qVelMat = np.array(self.buffer['qvel'])
            print('Calculating qacc, qvel is:')
            print(qVelMat)
            qAcc = np.gradient(qVelMat,
                self.simulation.model.opt.timestep, axis = 0)[-1]

            self.simulation = pose_model(self.simulation, jointSeries, qAcc = qAcc,
                method = self.simulationType)

        if self.simulationType == 'inverse':
            for key in self.buffer.keys():
                if key != 'qvel':
                    # qvel has to be calculated using mj_differentiatePos
                    self.buffer[key].popleft()
                    newValue = self.simulation.data.__getattribute__(key)
                    self.buffer[key].append(newValue)
                    print('added ' + key)
                    print(newValue)

            tempQVel = self.buffer['qvel'].popleft()
            dt = self.simulation.model.opt.timestep
            qPosMat = np.array(self.buffer['qpos'])
            functions.mj_differentiatePos(self.simulation.model, tempQVel,
                dt, qPosMat[-1], qPosMat[-2])
            self.buffer['qvel'].append(tempQVel)
            print('added qvel')
            print(tempQVel)
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
        stats = minner.minimize(method = self.method)
        return stats
