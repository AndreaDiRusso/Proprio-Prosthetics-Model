#
#   Hello World server in Python
#   Binds REP socket to tcp://*:5555
#   Expects b"Hello" from client, replies with b"World"
#

import zmq, pdb
import math
import numpy as np
import matplotlib
import matplotlib.pyplot as pyplot
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import mujoco2py_pb2 as mj2py
import scipy.io as sio
from helper_functions import *
import pickle

enable_plotting = 1

sitenames = ['right_iliac_crest', 'right_hip',  'right_knee',   'right_ankle', 'right_knuckle', 'right_toe','left_iliac_crest', 'left_hip']
names_to_titles = {
sitenames[0]:'Right Iliac Crest',
sitenames[1]:'Right Hip',
sitenames[2]:'Right Knee',
sitenames[3]:'Right Ankle',
sitenames[4]:'Right Knuckle',
sitenames[5]:'Right Toe',
sitenames[6]:'Left Iliac Crest',
sitenames[7]:'Left Hip'
}

context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://*:5556")

trials = [0]
sitenames, target = get_kin('TRM20', trials)

#pdb.set_trace()

N = 16 # muscles
nsites = 8
njoints = 12
scale_factor = 0.7

if enable_plotting:
    # Enlarge font size
    matplotlib.rcParams.update({'font.size': 20})
    plotColors = [cm.viridis(x) for x in np.linspace(0,1,3)]
    #pdb.set_trace()

forces = [[] for i in range(N)]
joint_forces = [[] for i in range(njoints)]
site_xpos = [[] for i in range(nsites)]
site_ypos = [[] for i in range(nsites)]
site_zpos = [[] for i in range(nsites)]

target_xpos = [[] for i in range(nsites)]
target_ypos = [[] for i in range(nsites)]
target_zpos = [[] for i in range(nsites)]

in_msg = mj2py.mujoco_msg()

total_num_samples = 0

for a in range(len(trials)):
    #pdb.set_trace()
    #angle_trace_deg = [trial[1,:],     trial[2,:],     trial[3,:]]

    #start_idx = 1
    #end_idx = -20
    #for idx in range(4):
    #    target[a][idx]['ypos'] = target[a][idx]['ypos'][start_idx:end_idx]
    #    target[a][idx]['zpos'] = target[a][idx]['zpos'][start_idx:end_idx]

    num_samples = len(target[a][0]['ypos'])

    total_num_samples += num_samples

    x = np.linspace(1, num_samples, num_samples)

    GT_coords = [0,0,0]
    for b in range(num_samples):

        message = socket.recv()

        in_msg.ParseFromString(message)
        for c in range(N):
            forces[c].append(in_msg.act[c].force)
            #print("Force[%d] = %f" % (c, in_msg.act[c].force))
            #print("Force[%d] = %f" % (c, forces[c][-1]))
        for c in range(njoints):
            joint_forces[c].append(in_msg.joint[c].force)
        #pdb.set_trace()
        for c in range(nsites):
            site_xpos[c].append(in_msg.site[c].x)
            site_ypos[c].append(in_msg.site[c].y)
            site_zpos[c].append(in_msg.site[c].z)
        #  Send reply back to client
        out_msg = mj2py.mujoco_msg()

        #if (b == 0):
        #    GT_coords = [in_msg.site[1].x, in_msg.site[1].y, in_msg.site[1].z]

        for c in range(nsites):
            site = out_msg.site.add()

            site.name = sitenames[c]
            #pdb.set_trace()
            bring_hip_to_origin = True
            # The greater trochanter is site_*pos[0], bring target to there
            if (bring_hip_to_origin):
                site.x = scale_factor*(target[a][c]['xpos'][b] - target[a][1]['xpos'][b])
                site.y = scale_factor*(target[a][c]['ypos'][b] - target[a][1]['ypos'][b])
                site.z = scale_factor*(target[a][c]['zpos'][b] - target[a][1]['zpos'][b])
            else: # bring hip to experimental hip
                site.x = scale_factor*(target[a][c]['xpos'][b] - target[a][1]['xpos'][b]) + site_xpos[1][b]
                site.y = scale_factor*(target[a][c]['ypos'][b] - target[a][1]['ypos'][b]) + site_ypos[1][b]
                site.z = scale_factor*(target[a][c]['zpos'][b] - target[a][1]['zpos'][b]) + site_zpos[1][b]

            target_xpos[c].append(site.x)
            target_ypos[c].append(site.y)
            target_zpos[c].append(site.z)

        out_msg_str = out_msg.SerializeToString()
        socket.send(out_msg_str)

        #start_time = 10
        #mid_time = 60
        #end_time = 110

        #if b == start_time or b == mid_time or b == end_time:
        #    print("t = %f" % b*10)
        #    pdb.set_trace()
        #Wait for next request from client
        #print(" ")

# Plotting
# pdb.set_trace()
if enable_plotting:
    fig1, axarr1 = pyplot.subplots(njoints, sharex=True)
    fig1.set_size_inches(6,15)
    fig2, axarr2 = pyplot.subplots(N, sharex=True)
    fig2.set_size_inches(12,30)
    fig3, axarr3 = pyplot.subplots(nsites, sharex=True)
    fig3.set_size_inches(6,15)
    fig4, axarr4 = pyplot.subplots(nsites, subplot_kw={'projection': '3d'})
    fig4.set_size_inches(6,15)

    time = np.array(range(len(joint_forces[0])))*10
    for a in range(njoints):
        axarr1[a].plot(time, joint_forces[a], 'k-') # Returns a tuple of line objects, thus the comma
        axarr1[a].set_title(in_msg.joint[a].name)
        pyplot.setp(axarr1[a].get_yticklabels(), visible=False)
        for label in axarr1[a].get_yticklabels()[0::3]:
            label.set_visible(True)
    axarr1[5].set_xlabel("Time (msec)")
    fig1.text(0.02, 0.5, 'Torque ($N\cdot m$)', va='center', rotation='vertical')

    for a in range(N):
        time = np.array(range(len(forces[a])))*10
        axarr2[a].plot(time, forces[a], 'k-') # Returns a tuple of line objects, thus the comma
        axarr2[a].set_title(in_msg.act[a].name)
        pyplot.setp(axarr2[a].get_yticklabels(), visible=False)
        for label in axarr2[a].get_yticklabels()[0::3]:
            label.set_visible(True)
    axarr2[N-1].set_xlabel("Time (msec)")
    fig2.text(0.02, 0.5, 'Force (N)', va='center', rotation='vertical')

    for a,b in enumerate(range(nsites)):
        time = np.array(range(len(site_ypos[b])))*10
        #pdb.set_trace()
        line1, = axarr3[a].plot(time, site_xpos[b], color = plotColors[0], linestyle='--', linewidth = 2) # Returns a tuple of line objects, thus the comma
        line1.set_label("Computed X position")
        line2, = axarr3[a].plot(time, target_xpos[b], color = plotColors[0], linestyle='-',  linewidth = 2) # Returns a tuple of line objects, thus the comma
        line2.set_label("Experimental X position")
        line1, = axarr3[a].plot(time, site_ypos[b],  color = plotColors[1], linestyle='--', linewidth = 2) # Returns a tuple of line objects, thus the comma
        line1.set_label("Computed Y position")
        line2, = axarr3[a].plot(time, target_ypos[b],  color = plotColors[1], linestyle='-', linewidth = 2) # Returns a tuple of line objects, thus the comma
        line2.set_label("Experimental Y position")
        line3, = axarr3[a].plot(time, site_zpos[b],  color = plotColors[2], linestyle='--', linewidth = 2) # Returns a tuple of line objects, thus the comma
        line3.set_label("Computed Z position")
        line4, = axarr3[a].plot(time, target_zpos[b],  color = plotColors[2], linestyle='-', linewidth = 2) # Returns a tuple of line objects, thus the comma
        line4.set_label("Experimental Z position")

        #pdb.set_trace()
        line5, = axarr4[a].plot([a_i - b_i for a_i, b_i in zip(site_xpos[b],site_xpos[1])], [a_i - b_i for a_i, b_i in zip(site_ypos[b],site_ypos[1])], [a_i - b_i for a_i, b_i in zip(site_zpos[b],site_zpos[1])], 'r--')
        line5.set_label("Computed XYZ trajectory")

        axarr3[a].set_title(names_to_titles[in_msg.site[b].name])
        axarr4[a].set_title(names_to_titles[in_msg.site[b].name])
        box = axarr3[a].get_position()
        axarr3[a].set_position([box.x0, box.y0, box.width * 0.8, box.height])
        pyplot.setp(axarr3[a].get_yticklabels(), visible=False)

        for label in axarr3[a].get_yticklabels()[0::3]:
            label.set_visible(True)

    axarr3[nsites-1].set_xlabel("Time (msec)")
    fig3.text(0.02, 0.5, 'Position (m)', va='center', rotation='vertical')
    axarr3[nsites-1].legend(loc='center left', bbox_to_anchor=(1, 0.5))

    axarr4[nsites-1].set_xlabel("Z position (cm)")
    fig4.text(0.02, 0.5, 'Y Position (m)', va='center', rotation='vertical')
    axarr4[nsites-1].legend(loc='center left', bbox_to_anchor=(1, 0.5))

figuresDir = 'D:\\tempdata\\Biomechanical Model\\figures\\'
fig1.savefig(figuresDir+"joint_forces.png")
fig2.savefig(figuresDir+"forces.png")
fig3.savefig(figuresDir+"fits.png")
fig4.savefig(figuresDir+"trajectories.png")

tracesFilename = open(figuresDir + 'siteserverdata.pickle','wb')

datadump = {
    'joint_forces': joint_forces,
    'forces': forces,
    'in_msg':in_msg,
    'time':time,
    'site_xpos': site_xpos,
    'site_ypos': site_ypos,
    'site_zpos': site_zpos,
    'target_xpos': target_xpos,
    'target_ypos': target_ypos,
    'target_zpos': target_zpos,
    'nsites': nsites,
}

pickle.dump(datadump, tracesFilename)

pyplot.show()
print("total_num_samples = %d" % total_num_samples)
input("Please click enter to continue")
