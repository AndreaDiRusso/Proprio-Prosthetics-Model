from neuron import h
import zmq,sys, pdb
import math as m
import mujoco2py_pb2 as mj2py
import simrun
import os

from PoissonSpiker import PoissonSpiker
from matplotlib import pyplot
import matplotlib
from neuronpy.graphics import spikeplot
from neuronpy.util import spiketrain
import cPickle as pickle

matplotlib.rcParams.update({'font.size': 20})

def Ia_model(k, l, l_dot, l0):
    return k*(cmp(l_dot,0)*21*m.pow((abs(l_dot/l0)),0.5)+200*(l-l0)/l0+60)

# ZMQ stuff
context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://*:5555")
in_msg = mj2py.mujoco_msg()

# Make the "neurons"
N = 8
cells = [PoissonSpiker() for i in range(N)]
time = []
fr = [[] for i in range(N)]
vel = [[] for i in range(N)]
lens = [[] for i in range(N)]
lens0 = [[] for i in range(N)]

t_vec = h.Vector()
id_vec = h.Vector()
for i in range(N):
    cells[i].netcon.record(t_vec, id_vec, i)

# Simulate
tstop = 2020 # must be frame_interval * number of samples in the site server
frame_interval = 10
sim_steps_per_frame = frame_interval / h.dt
num_iterations = m.floor(tstop / frame_interval)
print("Running for %d iterations" % num_iterations)

h.finitialize()

for a in range(int(num_iterations)):
    #  Wait for next request from client

    message = socket.recv()

    in_msg.ParseFromString(message)

    for b in range(N):
        rate = Ia_model(1, in_msg.tend[b].len, in_msg.tend[b].len_dot,  in_msg.tend[b].len0)
        cells[b].stim.interval = 1000/rate

        vel[b].append(in_msg.tend[b].len_dot)
        lens[b].append(in_msg.tend[b].len)
        lens0[b].append(in_msg.tend[b].len0)
        fr[b].append(rate)

    time.append(h.t)

    #if (h.cvode.active()):
    #    h.cvode.re_init()
    #else:
    #    h.fcurrent()

    #  Send reply back to client
    out_msg = mj2py.mujoco_msg()

    joint = out_msg.joint.add()
    joint.qpos =  h.t

    out_msg_str = out_msg.SerializeToString()
    socket.send(out_msg_str)

    for b in range(int(sim_steps_per_frame)):
        h.fadvance()
        #print("Advancing...")

    #os.system('cls')
    sys.stdout.write("Running ... %2.2f %% left\r" % (100*a/num_iterations))
    sys.stdout.flush()

print(" ")
spikes = spiketrain.netconvecs_to_listoflists(t_vec, id_vec)
spikes = list(reversed(spikes))

tendon_names = [in_msg.tend[a].name for a in range(N)]
tendon_names = list(reversed(tendon_names))
blank = [" "];
labels = blank + tendon_names

# Plotting
fig1, axarr1 = pyplot.subplots(N, sharex=True)
fig1.set_size_inches(6,15)
fig2, axarr2 = pyplot.subplots(N, sharex=True)
fig2.set_size_inches(6,15)
fig3, axarr3 = pyplot.subplots(N, sharex=True)
fig3.set_size_inches(6,15)

#pdb.set_trace()

for a in range(N):
    axarr1[a].plot(time, vel[a], 'k-',marker = 'o') # Returns a tuple of line objects, thus the comma
    axarr1[a].set_title(in_msg.tend[a].name)
    pyplot.setp(axarr1[a].get_yticklabels(), visible=False)
    for label in axarr1[a].get_yticklabels()[0::3]:
        label.set_visible(True)
    #axarr1[a].set_ylim([-2,2])
    axarr2[a].plot(time, lens[a], 'k-',marker = 'o') # Returns a tuple of line objects, thus the comma
    axarr2[a].set_title(in_msg.tend[a].name)
    pyplot.setp(axarr2[a].get_yticklabels(), visible=False)
    for label in axarr2[a].get_yticklabels()[0::3]:
        label.set_visible(True)
    #axarr2[a].set_ylim([-2,2])
    axarr3[a].plot(time, fr[a], 'k-',marker = 'o') # Returns a tuple of line objects, thus the comma
    axarr3[a].set_title(in_msg.tend[a].name)
    #axarr3[a].set_ylim([0,100])
    pyplot.setp(axarr3[a].get_yticklabels(), visible=False)
    for label in axarr3[a].get_yticklabels()[0::3]:
        label.set_visible(True)

axarr1[0].set_title("Tendon length rate of change (normalized) \n"+in_msg.tend[0].name)
axarr1[N-1].set_xlabel("Time (msec)")
fig1.text(0.02, 0.5, 'Tendon length rate of change (normalized)', va='center', rotation='vertical')

axarr2[0].set_title("Tendon displacement (normalized) \n"+in_msg.tend[0].name)
axarr2[N-1].set_xlabel("Time (msec)")
fig2.text(0.02, 0.5, 'Tendon displacement (normalized)', va='center', rotation='vertical')

axarr3[0].set_title("Tendon afferent firing rate (Hz) \n"+in_msg.tend[0].name)
axarr3[N-1].set_xlabel("Time (msec)")
fig3.text(0.02, 0.5, 'Tendon afferent firing rate (Hz)', va='center', rotation='vertical')

#pyplot.setp([a.get_xticklabels() for a in axarr1[0, :]], visible=False)
#pyplot.setp([a.get_xticklabels() for a in axarr2[0, :]], visible=False)
#pyplot.setp([a.get_xticklabels() for a in axarr3[0, :]], visible=False)

# Pre-process some figure variables
fig4=pyplot.figure()
fig4.set_size_inches(6,3)
ax4=fig4.add_subplot(111)
ax4.set_title('Raster Plot')
ax4.set_xlabel('Time (msec)') # Note LaTeX

sp = spikeplot.SpikePlot(fig=fig4)
sp.plot_spikes(spikes)

#print("There are %d ticks " % len(labels))
ax4.set_yticklabels(labels)

figuresDir = 'E:\\Google Drive\\Github\\tempdata\\Biomechanical Model\\figures\\'
tracesFilename = open(figuresDir + 'neuronserverdata.pickle','wb')
datadump = {
    'in_msg': in_msg,
    'time': time,
    'vel': vel,
    'lens': lens,
    'lens0': lens0,
    'fr': fr,
    'spikes': spikes
}
pickle.dump(datadump, tracesFilename)

fig1.savefig(figuresDir+"velocity.png", bbox_inches='tight')
fig2.savefig(figuresDir+"length.png", bbox_inches='tight')
fig3.savefig(figuresDir+"firing_rate.png", bbox_inches='tight')
fig4.savefig(figuresDir+"rasters.png", bbox_inches='tight')

pyplot.show()
input("Please click enter to continue")
