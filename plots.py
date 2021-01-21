from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from PyQt5.QtWidgets import QSizePolicy
import numpy.random.common
import numpy.random.bounded_integers
import numpy.random.entropy
import pandas as pd
import scipy.spatial
from matplotlib import colors as mcolors


class PlotCanvas(FigureCanvas):

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi, tight_layout=True)
        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)
        FigureCanvas.setSizePolicy(self,
                                   QSizePolicy.Expanding,
                                   QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

    def plot(self, principalDf, components, groups=None, active='all data', text='2 component PCA',
             annotations=True):
        self.fig.clear()
        self.axes = self.fig.add_subplot(111)
        self.axes.clear()
        self.axes.set_xlabel('Principal Component %i' % components[0], fontsize=14)
        self.axes.set_ylabel('Principal Component %i' % components[1], fontsize=14)
        self.axes.set_title(text, fontsize=16)

        'each group always has the same color'
        colors = ['k', 'b', 'g', 'm', 'r', 'y', 'c', 'gray', 'lime', 'olive', 'darkgreen',
                  'pink','mediumvioletred','violet','purple','indigo','fuchsia','darkred','brown',
                  'mistyrose','firebrick']

        all_groups = pd.unique(principalDf['target'])
        groups = groups if groups is not None else all_groups
        targets = {x: y for x, y in zip(all_groups, colors) if x in groups}

        plots_for_legend = []
        try:
            for target, color in targets.items():
                if active != 'only test':
                    plot = self.plot_single_group(principalDf, components, target, color, active=1,
                                                  annotations=annotations)
                    if plot: plots_for_legend.append(plot)
                if active != 'only train':
                    plot = self.plot_single_group(principalDf, components, target, color, active=0,
                                                  annotations=annotations)
                    if plot: plots_for_legend.append(plot)
        except Exception as e: print(e)

        self.axes.legend(handles=plots_for_legend, loc='upper left')
        self.plot_centroids(principalDf, components, active, targets)
        self.axes.grid()
        self.draw()

    def plot_single_group(self, principalDf, components, target, color, active, annotations):
        marker = 'o' if active == 1 else 'v'
        label = str(target) if active == 0 or  'cluster' in str(target) else str(target) + ' train'
        indicesToKeep = (principalDf['target'] == target) & (principalDf['Active'] == active)
        x_comp=principalDf.loc[indicesToKeep, 'pc %i' % components[0]]
        y_comp=principalDf.loc[indicesToKeep, 'pc %i' % components[1]]
        plot = self.axes.scatter(x_comp, y_comp, c=color, s=40, marker=marker, label=label)

        data_labels = principalDf.index.values[indicesToKeep]
        if annotations:
            for x, y, txt in zip(x_comp, y_comp, data_labels):
                self.axes.annotate(txt, (x, y), xytext=(10, 10), textcoords='offset points')
       
        return plot if any(indicesToKeep) else None

    def plot_centroids(self, principalDf, components, active, targets):
        for target, color in targets.items():
            if active == 'all data':
                indicesToKeep = (principalDf['target'] == target)
            elif active == 'only train':
                indicesToKeep = (principalDf['target'] == target) & (principalDf['Active'] == 1)
            else:  # active == 'only test'
                indicesToKeep = (principalDf['target'] == target) & (principalDf['Active'] == 0)

            centroid = principalDf.loc[indicesToKeep].mean()
            self.axes.scatter(centroid.loc['pc %i' % components[0]],
                              centroid.loc['pc %i' % components[1]],
                              c=color, s=40, marker='x')


class ComponentHist(FigureCanvas):
    def __init__(self, width=8, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi, tight_layout=True)
        FigureCanvas.__init__(self, self.fig)
        FigureCanvas.updateGeometry(self)

    def plot_hist(self, square, data, indices_to_keep, group):
        self.axes = self.fig.add_subplot(111)
        self.fig.suptitle(group, y=0.96)
        self.data_labels = data.index.values[data.index.values[indices_to_keep]]
        data = square[indices_to_keep]
        sort_data = sorted([[label, num] for label, num in zip(self.data_labels, data)], key = lambda x:x[1])
        self.data_labels = [x[0] for x in sort_data]; data = [x[1] for x in sort_data]
        rects = self.axes.bar(range(len(data)), data, color='k')

        self.axes.axhline(y=square.mean() + square.std(), color='r', linestyle='--')
        self.axes.axhline(y=square.mean() - square.std(), color='r', linestyle='--')

        self.auto_label(rects, "center")

    def auto_label(self, rects, xpos='center'):
        ha = {'center': 'center', 'right': 'left', 'left': 'right'}
        offset = {'center': 0, 'right': 1, 'left': -1}
        for rect, xi in zip(rects, self.data_labels):
            self.axes.annotate('{}'.format(xi),
                               xy=(rect.get_x() + rect.get_width() / 2, rect.get_height()),
                               xytext=(offset[xpos] * 1, 1),  # use 3 points offset
                               textcoords="offset points",  # in both directions
                               ha=ha[xpos], va='bottom')
