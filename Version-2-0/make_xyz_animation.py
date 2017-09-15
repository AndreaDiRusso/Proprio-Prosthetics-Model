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
tracesFilename = open('E:\\Google Drive\\Github\\tempdata\\Biomechanical Model\\figures\\siteserverdata.pickle','rb')

data = pickle.load(tracesFilename)
# This example uses subclassing, but there is no reason that the proper
# function couldn't be set up and then use FuncAnimation. The code is long, but
# not really complex. The length is due solely to the fact that there are a
# total of 9 lines that need to be changed for the animation as well as 3
# subplots that need initial set up.
class SubplotAnimation(animation.TimedAnimation):
    def __init__(self, data):

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

        self.nsites = data['nsites']
        self.time = np.array(data['time'])
        self.timestep = self.time[1] - self.time[0]

        self.site_xpos = np.array(data['site_xpos'])
        self.target_xpos = np.array(data['target_xpos'])

        self.site_ypos = np.array(data['site_ypos'])
        self.target_ypos = np.array(data['target_ypos'])

        self.site_zpos = np.array(data['site_zpos'])
        self.target_zpos = np.array(data['target_zpos'])

        axlims = [[None for idx in range(self.nsites)] for i in range(2)]
        for idx in range(self.nsites):
            all_values = np.concatenate((self.site_xpos[idx] + self.target_xpos[idx],
                self.site_ypos[idx] + self.target_ypos[idx],
                self.site_zpos[idx] + self.target_zpos[idx]))
            axlims[0][idx] = min(all_values)
            axlims[1][idx] = max(all_values)

            #pdb.set_trace()

        self.in_msg = data['in_msg']
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

        self.fig, self.axarr = pyplot.subplots(self.nsites, sharex=True)
        self.fig.set_size_inches(13,15)
        self.axarr[self.nsites-1].set_xlabel("Time (msec)")
        self.fig.text(0.025, 0.5, 'Position (m)', va='center', rotation='vertical')

        # make empty lists to contain all of the line objects
        self.lineComputedX = [None for i in range(self.nsites)]
        self.lineComputedXh = [None for i in range(self.nsites)] # head
        self.lineComputedXt = [None for i in range(self.nsites)] # tail

        self.lineExperimentalX = [None for i in range(self.nsites)]
        self.lineExperimentalXh = [None for i in range(self.nsites)] # head
        self.lineExperimentalXt = [None for i in range(self.nsites)] # tail

        self.lineComputedY = [None for i in range(self.nsites)]
        self.lineComputedYh = [None for i in range(self.nsites)] # head
        self.lineComputedYt = [None for i in range(self.nsites)] # tail

        self.lineExperimentalY = [None for i in range(self.nsites)]
        self.lineExperimentalYh = [None for i in range(self.nsites)] # head
        self.lineExperimentalYt = [None for i in range(self.nsites)] # tail

        self.lineComputedZ = [None for i in range(self.nsites)]
        self.lineComputedZh = [None for i in range(self.nsites)] # head
        self.lineComputedZt = [None for i in range(self.nsites)] # tail

        self.lineExperimentalZ = [None for i in range(self.nsites)]
        self.lineExperimentalZh = [None for i in range(self.nsites)] # head
        self.lineExperimentalZt = [None for i in range(self.nsites)] # tail

        for idx in range(self.nsites):
            #pdb.set_trace()
            self.lineComputedX[idx] = Line2D([], [], color = plotColors[0], linestyle='--', linewidth = 2)
            self.lineComputedX[idx].set_label("Computed X position")
            self.lineComputedXh[idx] = Line2D([], [], color = plotColors[0], linestyle='-', linewidth = 3)
            self.lineComputedXt[idx] = Line2D([], [],marker ='o', markeredgecolor = plotColors[0], color = plotColors[0])
            self.axarr[idx].add_line(self.lineComputedX[idx])
            self.axarr[idx].add_line(self.lineComputedXh[idx])
            self.axarr[idx].add_line(self.lineComputedXt[idx])

            self.lineExperimentalX[idx] = Line2D([], [], color = plotColors[1], linestyle='-', linewidth = 2)
            self.lineExperimentalX[idx].set_label("Experimental X position")
            self.lineExperimentalXh[idx] = Line2D([], [], color = plotColors[1], linestyle='-', linewidth = 3)
            self.lineExperimentalXt[idx] = Line2D([], [],marker ='o', markeredgecolor = plotColors[1], color = plotColors[1])
            self.axarr[idx].add_line(self.lineExperimentalX[idx])
            self.axarr[idx].add_line(self.lineExperimentalXh[idx])
            self.axarr[idx].add_line(self.lineExperimentalXt[idx])

            self.lineComputedY[idx] = Line2D([], [], color = plotColors[4], linestyle='--', linewidth = 2)
            self.lineComputedY[idx].set_label("Computed Y position")
            self.lineComputedYh[idx] = Line2D([], [], color = plotColors[4], linestyle='-', linewidth = 3)
            self.lineComputedYt[idx] = Line2D([], [],marker ='o', markeredgecolor = plotColors[4], color = plotColors[4])
            self.axarr[idx].add_line(self.lineComputedY[idx])
            self.axarr[idx].add_line(self.lineComputedYh[idx])
            self.axarr[idx].add_line(self.lineComputedYt[idx])

            self.lineExperimentalY[idx] = Line2D([], [], color = plotColors[5], linestyle='-', linewidth = 2)
            self.lineExperimentalY[idx].set_label("Experimental Y position")
            self.lineExperimentalYh[idx] = Line2D([], [], color = plotColors[5], linestyle='-', linewidth = 3)
            self.lineExperimentalYt[idx] = Line2D([], [],marker ='o', markeredgecolor = plotColors[5], color = plotColors[5])
            self.axarr[idx].add_line(self.lineExperimentalY[idx])
            self.axarr[idx].add_line(self.lineExperimentalYh[idx])
            self.axarr[idx].add_line(self.lineExperimentalYt[idx])

            self.lineComputedZ[idx] = Line2D([], [], color = plotColors[8], linestyle='--', linewidth = 2)
            self.lineComputedZ[idx].set_label("Computed Z position")
            self.lineComputedZh[idx] = Line2D([], [], color = plotColors[8], linestyle='-', linewidth = 3)
            self.lineComputedZt[idx] = Line2D([], [],marker ='o', markeredgecolor = plotColors[8], color = plotColors[8])
            self.axarr[idx].add_line(self.lineComputedZ[idx])
            self.axarr[idx].add_line(self.lineComputedZh[idx])
            self.axarr[idx].add_line(self.lineComputedZt[idx])

            self.lineExperimentalZ[idx] = Line2D([], [], color = plotColors[9], linestyle='-', linewidth = 2)
            self.lineExperimentalZ[idx].set_label("Experimental Z position")
            self.lineExperimentalZh[idx] = Line2D([], [], color = plotColors[9], linestyle='-', linewidth = 3)
            self.lineExperimentalZt[idx] = Line2D([], [],marker ='o', markeredgecolor = plotColors[9], color = plotColors[9])
            self.axarr[idx].add_line(self.lineExperimentalZ[idx])
            self.axarr[idx].add_line(self.lineExperimentalZh[idx])
            self.axarr[idx].add_line(self.lineExperimentalZt[idx])

            self.axarr[idx].set_xlim(0, max(self.time))
            self.axarr[idx].set_ylim(axlims[0][idx], axlims[1][idx])

            self.axarr[idx].set_title(names_to_titles[self.in_msg.site[idx].name])
            box = self.axarr[idx].get_position()
            self.axarr[idx].set_position([box.x0, box.y0, box.width * 0.8, box.height])
            #yplot.setp(axarr[idx].get_yticklabels(), visible=False)
            self.axarr[idx].set_yticks([0.8*axlims[0][idx], 0.8*axlims[1][idx]])
            self.axarr[idx].set_yticklabels([ '{:.{prec}E}'.format(0.8*axlims[0][idx], prec=1), '{:.{prec}E}'.format(0.8*axlims[1][idx], prec=1) ] )
            #for label in axarr[idx].get_yticklabels()[0::3]:
            #    label.set_visible(True)

        animation.TimedAnimation.__init__(self, self.fig, interval=10, blit=True)

    def _draw_frame(self, framedata):
        i = framedata
        head = i
        head_len = 10
        head_slice = (self.time >= self.time[i] - head_len*self.timestep) & (self.time <= self.time[i])

        self._drawn_artists = []
        for idx in range(self.nsites):
            self.lineComputedX[idx].set_data(self.time[:i], self.site_xpos[idx][:i])
            self.lineComputedXh[idx].set_data(self.time[head_slice], self.site_xpos[idx][head_slice])
            self.lineComputedXt[idx].set_data(self.time[head], self.site_xpos[idx][head])

            self.lineExperimentalX[idx].set_data(self.time[:i], self.target_xpos[idx][:i])
            self.lineExperimentalXh[idx].set_data(self.time[head_slice], self.target_xpos[idx][head_slice])
            self.lineExperimentalXt[idx].set_data(self.time[head], self.target_xpos[idx][head])

            self.lineComputedY[idx].set_data(self.time[:i], self.site_ypos[idx][:i])
            self.lineComputedYh[idx].set_data(self.time[head_slice], self.site_ypos[idx][head_slice])
            self.lineComputedYt[idx].set_data(self.time[head], self.site_ypos[idx][head])

            self.lineExperimentalY[idx].set_data(self.time[:i], self.target_ypos[idx][:i])
            self.lineExperimentalYh[idx].set_data(self.time[head_slice], self.target_ypos[idx][head_slice])
            self.lineExperimentalYt[idx].set_data(self.time[head], self.target_ypos[idx][head])

            self.lineComputedZ[idx].set_data(self.time[:i], self.site_zpos[idx][:i])
            self.lineComputedZh[idx].set_data(self.time[head_slice], self.site_zpos[idx][head_slice])
            self.lineComputedZt[idx].set_data(self.time[head], self.site_zpos[idx][head])

            self.lineExperimentalZ[idx].set_data(self.time[:i], self.target_zpos[idx][:i])
            self.lineExperimentalZh[idx].set_data(self.time[head_slice], self.target_zpos[idx][head_slice])
            self.lineExperimentalZt[idx].set_data(self.time[head], self.target_zpos[idx][head])

            #print ("there are %d items in drawn_artists" % len(self._drawn_artists))

            self._drawn_artists = self._drawn_artists + [self.lineComputedX[idx], self.lineComputedXh[idx], self.lineComputedXt[idx],
                self.lineExperimentalX[idx], self.lineExperimentalXh[idx], self.lineExperimentalXt[idx],
                self.lineComputedY[idx], self.lineComputedYh[idx], self.lineComputedYt[idx],
                self.lineExperimentalY[idx], self.lineExperimentalYh[idx], self.lineExperimentalYt[idx],
                self.lineComputedZ[idx], self.lineComputedZh[idx], self.lineComputedZt[idx],
                self.lineExperimentalZ[idx], self.lineExperimentalZh[idx], self.lineExperimentalZt[idx]]

        if i == (self.time.size - 1):
            self.fig.savefig('E:\\Google Drive\\Github\\tempdata\\Biomechanical Model\\figures\\xyz_animation.png')
            #pdb.set_trace()
        self._drawn_artists = self._drawn_artists + [self.axarr[int(self.nsites/2)].legend(loc='center left', bbox_to_anchor=(1, 0.5))]
    def new_frame_seq(self):
        return iter(range(self.time.size))

    def _init_draw(self):
        #pdb.set_trace()
        for idx in range(self.nsites):
            lines = [self.lineComputedX[idx], self.lineComputedXh[idx], self.lineComputedXt[idx],
                self.lineExperimentalX[idx], self.lineExperimentalXh[idx], self.lineExperimentalXt[idx],
                self.lineComputedY[idx], self.lineComputedYh[idx], self.lineComputedYt[idx],
                self.lineExperimentalY[idx], self.lineExperimentalYh[idx], self.lineExperimentalYt[idx],
                self.lineComputedZ[idx], self.lineComputedZh[idx], self.lineComputedZt[idx],
                self.lineExperimentalZ[idx], self.lineExperimentalZh[idx], self.lineExperimentalZt[idx]]

            for l in lines:
                l.set_data([], [])

ani = SubplotAnimation(data)
ani.fig.subplots_adjust(left = 0.2)
ani.fig.subplots_adjust(right = 0.55)
ani.fig.subplots_adjust(bottom = 0.05)
ani.fig.subplots_adjust(top = 0.95)
ani.fig.subplots_adjust(wspace = 0.2)
ani.fig.subplots_adjust(hspace = 0.3)
ani.save('E:\\Google Drive\\Github\\tempdata\\Biomechanical Model\\figures\\xyz_animation.mp4')
#pyplot.show()
