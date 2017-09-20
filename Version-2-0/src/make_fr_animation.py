import math, pdb
import numpy as np
import matplotlib
import matplotlib.pyplot as pyplot
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.lines import Line2D
import matplotlib.animation as animation
import scipy.io as sio
import cPickle as pickle

# Enlarge font size
# Customize default matplotlib parameters
pyplot.style.use('seaborn-darkgrid')
matplotlib.rcParams.update({'font.size': 20})
matplotlib.rcParams.update({'text.color': 'white'})
matplotlib.rcParams.update({'axes.facecolor': 'black'})
matplotlib.rcParams.update({'axes.edgecolor': 'white'})
matplotlib.rcParams.update({'savefig.facecolor': 'black'})
matplotlib.rcParams.update({'savefig.edgecolor': 'white'})
matplotlib.rcParams.update({'figure.facecolor': 'black'})
matplotlib.rcParams.update({'figure.edgecolor': 'white'})
matplotlib.rcParams.update({'axes.labelcolor': 'white'})
plotColors = [cm.viridis(x) for x in np.linspace(0.5,1,12)]

tracesFilename = open('E:\\Google Drive\\Github\\tempdata\\Biomechanical Model\\figures\\neuronserverdata.pickle','rb')

data = pickle.load(tracesFilename)

def find_bad_frames(vel):
    vel_abs = np.absolute(vel-vel.mean())
    threshold = 2*vel.std()
    bitmask = vel_abs > threshold

    corrected_vel = vel
    corrected_vel[bitmask] = vel.mean()
    #pdb.set_trace()
    return corrected_vel, bitmask

def correct_bad_len(len_vec, bad_frames):
    corrected_len_vec = len_vec
    for midx, this_frame in enumerate(bad_frames): # muscle idx and its associated badness lookup table
        for idx, val in enumerate(this_frame): # for each values
            if val:
                len_vec[midx][idx] = len_vec[midx][idx-1]
    return corrected_len_vec

def Ia_model(k, l, l_dot, l0):
    fr = k*(21*pow((abs(l_dot/l0)),0.5)+200*(l-l0)/l0+60)
    bitmask = fr < 0
    fr[bitmask] = 0
    return fr
# This example uses subclassing, but there is no reason that the proper
# function couldn't be set up and then use FuncAnimation. The code is long, but
# not really complex. The length is due solely to the fact that there are a
# total of 9 lines that need to be changed for the animation as well as 3
# subplots that need initial set up.
class SubplotAnimation(animation.TimedAnimation):
    def __init__(self, data):
        #self.N = data['N']
        self.N = 8

        # Customize default matplotlib parameters
        pyplot.style.use('seaborn-darkgrid')
        matplotlib.rcParams.update({'font.size': 20})
        matplotlib.rcParams.update({'text.color': 'white'})
        matplotlib.rcParams.update({'axes.facecolor': 'black'})
        matplotlib.rcParams.update({'axes.edgecolor': 'white'})
        matplotlib.rcParams.update({'savefig.facecolor': 'black'})
        matplotlib.rcParams.update({'savefig.edgecolor': 'white'})
        matplotlib.rcParams.update({'figure.facecolor': 'black'})
        matplotlib.rcParams.update({'figure.edgecolor': 'white'})
        matplotlib.rcParams.update({'axes.labelcolor': 'white'})

        self.time = np.array(data['time'])
        self.timestep = self.time[1] - self.time[0]

        self.fr = np.array(data['fr'])
        self.len = np.array(data['lens'])
        self.len0 = np.array(data['lens0'])

        self.vel = np.array(data['vel'])
        #self.vel = np.concatenate([np.diff(self.len), np.zeros([self.N, 1])],axis = 1)

        self.corrected_vel, self.bad_mask = find_bad_frames(self.vel)
        self.corrected_len = correct_bad_len(np.array(data['lens']), self.bad_mask)

        self.corrected_fr = Ia_model(1, self.corrected_len, self.corrected_vel, self.len0)
        #self.corrected_fr = self.fr

        axlims_fr = [[min(self.corrected_fr[idx]) for idx in range(self.N)],
            [max(self.corrected_fr[idx]) for idx in range(self.N)]]

        axlims_len = [[min(self.corrected_len[idx]) for idx in range(self.N)],
            [max(self.corrected_len[idx]) for idx in range(self.N)]]

        axlims_vel = [[min(self.corrected_vel[idx]) for idx in range(self.N)],
            [max(self.corrected_vel[idx]) for idx in range(self.N)]]

        self.in_msg = data['in_msg']
        muscleNames = ['r_IL',  'r_GMED', 'r_VAS', 'r_TA',
            'r_SOL', 'r_RF', 'r_BF', 'r_GAS']
        names_to_titles = {
        muscleNames[0]:'Iliacus',
        muscleNames[1]:'Gluteus Medius',
        muscleNames[2]:'Vastus',
        muscleNames[3]:'Tibialis Anterior',
        muscleNames[4]:'Soleus',
        muscleNames[5]:'Rectus Femoris',
        muscleNames[6]:'Biceps Femoris',
        muscleNames[7]:'Gastrocnemius'
        }

        self.fig, self.axarr = pyplot.subplots(self.N, sharex=True)
        self.still_fr_fig, self.still_fr_axarr = pyplot.subplots(self.N, sharex=True)
        self.still_len_fig, self.still_len_axarr = pyplot.subplots(self.N, sharex=True)
        self.still_vel_fig, self.still_vel_axarr = pyplot.subplots(self.N, sharex=True)

        self.fig.set_size_inches(8,15)
        self.axarr[self.N-1].set_xlabel("Time (msec)")
        self.fig.text(0.025, 0.5, 'Simulated Afferent Firing Rate (Hz)', va='center', rotation='vertical')

        self.still_fr_fig.set_size_inches(8,15)
        self.still_fr_axarr[self.N-1].set_xlabel("Time (msec)")
        self.still_fr_fig.text(0.025, 0.5, 'Afferent Firing Rate (Hz)', va='center', rotation='vertical')

        self.still_len_fig.set_size_inches(8,15)
        self.still_len_axarr[self.N-1].set_xlabel("Time (msec)")
        self.still_len_fig.text(0.025, 0.5, 'Muscle Length (Normalized to Rest)', va='center', rotation='vertical')

        self.still_vel_fig.set_size_inches(8,15)
        self.still_vel_axarr[self.N-1].set_xlabel("Time (msec)")
        self.still_vel_fig.text(0.025, 0.5, 'Muscle Velocity (Normalized to Rest)', va='center', rotation='vertical')

        # make empty lists to contain all of the line objects
        self.line = [None for i in range(self.N)]
        self.lineh = [None for i in range(self.N)] # head
        self.linet = [None for i in range(self.N)] # tail
        # still image
        self.still_fr_line = [None for i in range(self.N)]
        self.still_len_line = [None for i in range(self.N)]
        self.still_vel_line = [None for i in range(self.N)]

        self.still_corrected_fr_line = [None for i in range(self.N)]
        self.still_corrected_len_line = [None for i in range(self.N)]
        self.still_corrected_vel_line = [None for i in range(self.N)]

        for idx in range(self.N):
            #pdb.set_trace()
            self.line[idx] = Line2D([], [], color = plotColors[0], linestyle='--', linewidth = 2)
            self.line[idx].set_label("Computed X position")
            self.lineh[idx] = Line2D([], [], color = plotColors[0], linestyle='-', linewidth = 3)
            self.linet[idx] = Line2D([], [],marker ='o', markeredgecolor = plotColors[0], color = plotColors[0])
            self.axarr[idx].add_line(self.line[idx])
            self.axarr[idx].add_line(self.lineh[idx])
            self.axarr[idx].add_line(self.linet[idx])

            self.axarr[idx].set_xlim(0, max(self.time))
            self.axarr[idx].set_ylim(axlims_fr[0][idx], axlims_fr[1][idx])

            self.axarr[idx].set_title(names_to_titles[muscleNames[idx]])

            self.axarr[idx].set_yticks([0.8*axlims_fr[0][idx], 0.8*axlims_fr[1][idx]])
            self.axarr[idx].set_yticklabels([ '{:.{prec}E}'.format(0.8*axlims_fr[0][idx], prec=1), '{:.{prec}E}'.format(0.8*axlims_fr[1][idx], prec=1) ] )

            # still images
            # fr

            self.still_fr_line[idx], = self.still_fr_axarr[idx].plot(self.time, self.fr[idx], color = plotColors[0], linestyle='-', linewidth = 2)
            self.still_fr_line[idx].set_label("Original Firing Rate (Hz)")

            self.still_fr_axarr[idx].set_xlim(0, max(self.time))
            self.still_fr_axarr[idx].set_ylim(axlims_fr[0][idx], axlims_fr[1][idx])

            self.still_fr_axarr[idx].set_title(names_to_titles[muscleNames[idx]])

            self.still_fr_axarr[idx].set_yticks([0.8*axlims_fr[0][idx], 0.8*axlims_fr[1][idx]])
            self.still_fr_axarr[idx].set_yticklabels([ '{:.{prec}E}'.format(0.8*axlims_fr[0][idx], prec=1), '{:.{prec}E}'.format(0.8*axlims_fr[1][idx], prec=1) ] )
            # len
            self.still_len_line[idx], = self.still_len_axarr[idx].plot(self.time, self.len[idx], color = plotColors[0], linestyle='-', linewidth = 2)
            self.still_len_line[idx].set_label("Original Length (Normalized)")

            self.still_len_axarr[idx].set_xlim(0, max(self.time))
            self.still_len_axarr[idx].set_ylim(axlims_len[0][idx], axlims_len[1][idx])

            self.still_len_axarr[idx].set_title(names_to_titles[muscleNames[idx]])

            self.still_len_axarr[idx].set_yticks([0.8*axlims_len[0][idx], 0.8*axlims_len[1][idx]])
            self.still_len_axarr[idx].set_yticklabels([ '{:.{prec}E}'.format(0.8*axlims_len[0][idx], prec=1), '{:.{prec}E}'.format(0.8*axlims_len[1][idx], prec=1) ] )
            #vel
            self.still_vel_line[idx], = self.still_vel_axarr[idx].plot(self.time, self.vel[idx], color = plotColors[0], linestyle='-', linewidth = 2)
            self.still_vel_line[idx].set_label("Original Length (Normalized)")

            self.still_vel_axarr[idx].set_xlim(0, max(self.time))
            self.still_vel_axarr[idx].set_ylim(axlims_vel[0][idx], axlims_vel[1][idx])

            self.still_vel_axarr[idx].set_title(names_to_titles[muscleNames[idx]])

            self.still_vel_axarr[idx].set_yticks([0.8*axlims_vel[0][idx], 0.8*axlims_vel[1][idx]])
            self.still_vel_axarr[idx].set_yticklabels([ '{:.{prec}E}'.format(0.8*axlims_vel[0][idx], prec=1), '{:.{prec}E}'.format(0.8*axlims_vel[1][idx], prec=1) ] )

            # corrected still images
            # fr
            self.still_corrected_fr_line[idx], = self.still_fr_axarr[idx].plot(self.time, self.corrected_fr[idx], color = plotColors[0], linestyle='--', linewidth = 2)
            self.still_corrected_fr_line[idx].set_label("Corrected Firing Rate (Hz)")

            # len
            self.still_corrected_len_line[idx], = self.still_len_axarr[idx].plot(self.time, self.corrected_len[idx], color = plotColors[0], linestyle='--', linewidth = 2)
            self.still_corrected_len_line[idx].set_label("Corrected Length (Normalized)")

            #vel
            self.still_corrected_vel_line[idx], = self.still_vel_axarr[idx].plot(self.time, self.corrected_vel[idx], color = plotColors[0], linestyle='--', linewidth = 2)
            self.still_corrected_vel_line[idx].set_label("Corrected Length (Normalized)")

        self.still_len_axarr[round(self.N/2)].legend(loc='center left', bbox_to_anchor=(1, 0.5))
        self.still_fr_axarr[round(self.N/2)].legend(loc='center left', bbox_to_anchor=(1, 0.5))
        self.still_vel_axarr[round(self.N/2)].legend(loc='center left', bbox_to_anchor=(1, 0.5))

        animation.TimedAnimation.__init__(self, self.fig, interval=10, blit=True)

    def _draw_frame(self, framedata):
        i = framedata
        head = i
        head_len = 0
        head_slice = (self.time >= self.time[i] - head_len*self.timestep) & (self.time <= self.time[i])

        self._drawn_artists = []
        for idx in range(self.N):
            self.line[idx].set_data(self.time[:i], self.corrected_fr[idx][:i])
            self.lineh[idx].set_data(self.time[head_slice], self.corrected_fr[idx][head_slice])
            self.linet[idx].set_data(self.time[head], self.corrected_fr[idx][head])

            self._drawn_artists = self._drawn_artists + [self.line[idx], self.lineh[idx], self.linet[idx]]
        if i == (self.time.size - 1):
            self.fig.savefig('E:\\Google Drive\\Github\\tempdata\\Biomechanical Model\\figures\\fr_animation.png')
            #pdb.set_trace()

    def new_frame_seq(self):
        return iter(range(self.time.size))

    def _init_draw(self):
        #pdb.set_trace()
        for idx in range(self.N):
            lines = [self.line[idx], self.lineh[idx], self.linet[idx]]

            for l in lines:
                l.set_data([], [])

ani = SubplotAnimation(data)
ani.fig.subplots_adjust(left   = 0.25)
ani.fig.subplots_adjust(right  = 0.95)
ani.fig.subplots_adjust(bottom = 0.05)
ani.fig.subplots_adjust(top    = 0.95)
ani.fig.subplots_adjust(wspace = 0.2 )
ani.fig.subplots_adjust(hspace = 0.3 )
ani.save('E:\\Google Drive\\Github\\tempdata\\Biomechanical Model\\figures\\fr_animation.mp4')
#pyplot.show()
