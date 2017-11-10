
"""
=================
Animated subplots
=================

This example uses subclassing, but there is no reason that the proper function
couldn't be set up and then use FuncAnimation. The code is long, but not
really complex. The length is due solely to the fact that there are a total of
9 lines that need to be changed for the animation as well as 3 subplots that
need initial set up.

"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.animation as animation
from dataAnalysis.helperFunctions.helper_functions import *
import argparse, pickle, pdb, itertools
import seaborn as sns

parser = argparse.ArgumentParser()
parser.add_argument('--file', default = 'W:/ENG_Neuromotion_Shared/group/Proprioprosthetics/Data/201709261100-Proprio/T_1_kinematics.pickle')
parser.add_argument('--y', default = 'Position (m)')

args = parser.parse_args()
argFile = args.file
whichY = args.y

def get_new_lims(datum, x, y, ax):
    minXLim = datum[x].min() if datum[x].min() < ax.get_xlim()[0] else ax.get_xlim()[0]
    maxXLim = datum[x].max() if datum[x].max() > ax.get_xlim()[1] else ax.get_xlim()[1]
    minYLim = datum[y].min() if datum[y].min() < ax.get_ylim()[0] else ax.get_ylim()[0]
    maxYLim = datum[y].max() if datum[y].max() > ax.get_ylim()[1] else ax.get_ylim()[1]
    return minXLim, maxXLim, minYLim, maxYLim

#fig = reloadPlot(filePath = argFile)
with open(argFile, 'rb') as f:
    g = pickle.load(f)

class SubplotAnimation(animation.TimedAnimation):
    def __init__(self, facetGrid, x = 'Time (sec)', y = 'Position (m)'):

        sns.set_style('darkgrid')
        plt.style.use('seaborn-darkgrid')
        invertColors = False
        matplotlib.rcParams.update({'font.size': 12})
        matplotlib.rcParams.update({'text.color': 'black' if invertColors else 'white'})
        matplotlib.rcParams.update({'axes.facecolor': 'white' if invertColors else 'black'})
        matplotlib.rcParams.update({'axes.edgecolor': 'black' if invertColors else 'white'})
        matplotlib.rcParams.update({'savefig.facecolor': 'white' if invertColors else 'black'})
        matplotlib.rcParams.update({'savefig.edgecolor': 'black' if invertColors else 'white'})
        matplotlib.rcParams.update({'savefig.bbox': 'tight'})
        matplotlib.rcParams.update({'figure.figsize': (6, 9)})
        matplotlib.rcParams.update({'figure.facecolor': 'white' if invertColors else 'black'})
        matplotlib.rcParams.update({'figure.edgecolor': 'black' if invertColors else 'white'})
        matplotlib.rcParams.update({'axes.labelcolor': 'black' if invertColors else 'white'})
        matplotlib.rcParams.update({'xtick.color': 'black' if invertColors else 'white'})
        matplotlib.rcParams.update({'ytick.color': 'black' if invertColors else 'white'})

        self.fig = plt.figure()

        self.facetGrid = facetGrid
        #pdb.set_trace()
        facetDict = dict(facetGrid.facet_data())
        self.x = x
        self.y = y

        #pdb.set_trace()
        if facetGrid.hue_names is not None:
            self.hasHues = True
            self.nHues = len(facetGrid.hue_names)
            lws = facetGrid.hue_kws['lw']
            lss = facetGrid.hue_kws['ls']
        else:
            self.hasHues = False
            self.nHues = 1
            lws = [3]
            lss = ['-']

        self.nRows = facetGrid._nrow
        self.nCols = facetGrid._ncol
        if facetGrid.hue_names is not None:
            self.hueLabels = facetGrid.hue_kws['label']
            #pdb.set_trace()
        else:
            self.hueLabels = [whichY]

        self.allLabels = self.hueLabels

        if facetGrid._row_var is not None:
            self.rowLabels = facetGrid.data[facetGrid._row_var].unique()
            self.allLabels = list(itertools.product(self.rowLabels, self.allLabels))
            self.allLabels = [i[0] + ' ' + i[1] for i in self.allLabels]
            self.facetLabels = self.rowLabels

        if facetGrid._col_var is not None:
            self.colLabels = facetGrid.data[facetGrid._col_var].unique()
            self.allLabels = list(itertools.product(self.colLabels, self.allLabels))
            self.allLabels = [i[0] + ' ' + i[1] for i in self.allLabels]
            if facetGrid._row_var is not None:
                self.facetLabels = list(itertools.product(self.rowLabels, self.facetLabels))
                self.facetLabels = [i[0] + ' ' + i[1] for i in self.facetLabels]
            else:
                self.facetLabels = self.colLabels
        else:
            self.colLabels = None

        nameIterator = iter(self.facetLabels)
        ax  = [
        [self.fig.add_subplot(self.nRows, self.nCols, j + i*self.nCols + 1, title = next(nameIterator))
            for i in range(self.nCols)
            ]
        for j in range(self.nRows)
        ]
        #pdb.set_trace()
        #nameIterator = iter(self.allLabels)
        nameIterator = itertools.cycle(self.hueLabels)
        self.line = [
                [
                    [Line2D([], [], color=facetGrid._colors[i],
                        lw = lws[i],
                        ls = lss[i],
                        label = next(nameIterator)
                        )
                        for i in range(self.nHues)
                        ]
                    for j in range(self.nCols)
                    ]
                for k in range(self.nRows)
                ]

        #nameIterator = iter(self.allLabels)
        #pdb.set_trace()
        self.lineA = [
                [
                    [Line2D([], [], color=facetGrid._colors[i],
                        lw = lws[i],
                        ls = lss[i],
                        #label = next(nameIterator) + ' head'
                        )
                        for i in range(self.nHues)
                        ]
                    for j in range(self.nCols)
                    ]
                for k in range(self.nRows)
                ]

        #nameIterator = iter(self.allLabels)
        self.lineE = [
                [
                    [Line2D([], [], color=facetGrid._colors[i],
                        lw = lws[i],
                        ls = lss[i],
                        marker='o', markeredgecolor=facetGrid._colors[i],
                        #label = next(nameIterator) + ' tip'
                        )
                        for i in range(self.nHues)
                        ]
                    for j in range(self.nCols)
                    ]
                for k in range(self.nRows)
                ]

        legendSet = False
        plt.tight_layout()
        for rowIdx in range(self.nRows):
            #print("row: %d" % rowIdx)
            for colIdx in range(self.nCols):
                #print("col: %d" % colIdx)
                for lineIdx in range(self.nHues):
                    #print("line: %d" % lineIdx)
                    ax[rowIdx][colIdx].add_line(self.line[rowIdx][colIdx][lineIdx])
                    ax[rowIdx][colIdx].add_line(self.lineA[rowIdx][colIdx][lineIdx])
                    ax[rowIdx][colIdx].add_line(self.lineE[rowIdx][colIdx][lineIdx])

                    datum = facetDict[(rowIdx, colIdx, lineIdx)]
                    #pdb.set_trace()
                    if (lineIdx) == (0):
                        #first time around, make sure to override the defaults
                        minXLim = datum[self.x].min()
                        maxXLim = datum[self.x].max()
                        minYLim = datum[self.y].min()
                        maxYLim = datum[self.y].max()
                    else:
                        minXLim, maxXLim, minYLim, maxYLim =\
                            get_new_lims(datum, self.x, self.y, ax[rowIdx][colIdx])

                    ax[rowIdx][colIdx].set_xlim(minXLim, maxXLim)
                    ax[rowIdx][colIdx].set_ylim(minYLim, maxYLim)
                    ax[rowIdx][colIdx].set_xlabel(self.x)
                    ax[rowIdx][colIdx].set_ylabel(self.y)
                    box = ax[rowIdx][colIdx].get_position()
                    ax[rowIdx][colIdx].set_position([box.x0,box.y0,box.width*0.95,box.height])
                if not legendSet:
                    ax[rowIdx][colIdx].legend(loc='center right', bbox_to_anchor = (1.4,-2.5))
                    legendSet = True
        animation.TimedAnimation.__init__(self, self.fig, interval=10, blit=True)

    def _draw_frame(self, framedata):
        i = framedata
        head = i - 1

        self._drawn_artists = []

        facetDict = dict(self.facetGrid.facet_data())

        head_slice = ((facetDict[(0,0,0)][self.x] >\
            facetDict[(0,0,0)][self.x].iloc[i] - 1.0) &\
            (facetDict[(0,0,0)][self.x] <\
            facetDict[(0,0,0)][self.x].iloc[i])).values
        """
        head_slice = slice((head - 100),head)
        """
        for idx, datum in facetDict.items():

            """
            print("i")
            print(i)
            print("head_slice")
            print(head_slice)
            print('data')
            print(datum[self.x])
            headSliceEnd = head_slice[::-1].idxmax()
            """
            self.line[idx[0]][idx[1]][idx[2]].\
                set_data(datum[self.x].iloc[:i], datum[self.y].iloc[:i])
            self.lineA[idx[0]][idx[1]][idx[2]].\
                set_data(datum[self.x][head_slice], datum[self.y][head_slice])
            self.lineE[idx[0]][idx[1]][idx[2]].\
                set_data(datum[self.x].iloc[head], datum[self.y].iloc[head])

            self._drawn_artists = self._drawn_artists + \
                [
                self.line[idx[0]][idx[1]][idx[2]],
                self.lineA[idx[0]][idx[1]][idx[2]],
                self.lineE[idx[0]][idx[1]][idx[2]]
                ]
        """
        print('Drawn artists: ')
        print(self._drawn_artists)
        """
    def new_frame_seq(self):
        # TODO: does not support multiple x axes
        idx, datum = next(self.facetGrid.facet_data())
        #pdb.set_trace()
        return iter(datum[self.x].index)

    def _init_draw(self):
        lines = []
        for idx, datum in self.facetGrid.facet_data():
            lines = lines + \
                [
                self.line[idx[0]][idx[1]][idx[2]],
                self.lineA[idx[0]][idx[1]][idx[2]],
                self.lineE[idx[0]][idx[1]][idx[2]]
                ]
        for l in lines:
            l.set_data([], [])

ani = SubplotAnimation(facetGrid = g, y = whichY)
ani.save(argFile.split('_plot')[0] + '_animation.mp4')
#plt.show()
