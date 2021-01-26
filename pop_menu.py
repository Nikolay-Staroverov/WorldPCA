from PyQt5.QtWidgets import (QWidget, QFileDialog, QCheckBox, QDesktopWidget, QGroupBox,
                             QHBoxLayout, QVBoxLayout, QGridLayout, QFrame, QButtonGroup,QComboBox,
                             QPushButton, QLabel, QSlider, QRadioButton, QTableWidget, QTableWidgetItem,
                             QApplication, QSizePolicy, QAbstractScrollArea)
from PyQt5.QtCore import Qt, QSize
from abc import abstractmethod
from itertools import product
from math import ceil
import scipy.spatial.distance
from scipy import stats
import numpy as np

from widgets import *


class PopMenu(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.setWindowTitle('World PCA')
        self.setWindowFlags(Qt.Dialog)
        self.setWindowModality(Qt.WindowModal)

        self.btn_apply = QPushButton('apply', self)
        self.btn_apply.setFixedWidth(200)

        vbox = QVBoxLayout()
        self.setLayout(vbox)
        self.grid = QGridLayout(self)
        self.create_buttons()
        vbox.addLayout(self.grid)
        vbox.addWidget(self.btn_apply, alignment=Qt.AlignLeft)
        self.btn_apply.clicked.connect(self.close)
        self.btn_apply.clicked.connect(self.apply_function)

    @abstractmethod
    def apply_function(self):
        pass

    @abstractmethod
    def create_buttons(self):
        pass


class PopMenuFeatures(PopMenu):

    def apply_function(self):
        tg = self.parent

        tg.included_features = [col.text() for col in self.col_list if col.isChecked()]
        tg.train_data_groups = [btn.text() for btn in self.gbtn_list if btn.isChecked()]

        tg.PCA.df_train = tg.PCA.df_train[tg.PCA.df_train['target'].isin(self.parent.train_data_groups)]

        tg.PCA.df_train = tg.PCA.df_train[tg.included_features]
        tg.PCA.df = tg.PCA.df[tg.included_features]
        tg.PCA.fit()

    def create_buttons(self):
        self.grid.addWidget(QLabel('Features'), 0, 0, 1, 4, Qt.AlignCenter)
        self.col_list = []
        numbers = len(self.parent.PCA.df.columns.values)
        places = product(range(1, ceil(numbers / 3) + 1), range(4))
        for name, place in zip(self.parent.PCA.df.columns.values, places):
            temp = QCheckBox(name, self)
            self.grid.addWidget(temp, *place)
            if not self.parent.included_features or name in self.parent.included_features:
                temp.toggle()
            self.col_list.append(temp)

        self.gbtn_list = []
        start_num = ceil(numbers / 3) + 2
        self.grid.addWidget(QLabel('Groups for training'), start_num, 0, 1, 4, Qt.AlignCenter)
        groups = pd.unique(self.parent.PCA.df_train['target'])
        places = product(range(start_num + 1, start_num + 1 + len(groups)), range(4))
        for name, place in zip(groups, places):
            btn = QCheckBox(name, self)
            self.grid.addWidget(btn, *place)
            if not self.parent.train_data_groups or btn.text() in self.parent.train_data_groups:
                btn.toggle()
            self.gbtn_list.append(btn)


class PopMenuPlot(PopMenu):

    def apply_function(self):
        self.parent.data_groups = [btn.text() for btn in self.gbtn_list if btn.isChecked()]
        try:
            self.parent.principalDf = self.parent.PCA.transform()
            self.parent.canvas.plot(self.parent.principalDf, self.parent.princ_cmp,
                                    self.parent.data_groups, self.parent.active,
                                    annotations=self.parent.add_annotations)
            self.parent.create_pc_table()
        except AttributeError as e:
            print(e)

    def create_buttons(self):
        pc = self.parent.princ_cmp
        self.vis_data_group = RadioBtnGroup('principal\ncomponents', ['Components 1 and 2',
                                                                      'Components 3 and 4',
                                                                      'Components 1 and 3',
                                                                      'Components 2 and 3', ],
                                            f'Components {pc[0]} and {pc[1]}', self.parent.set_components)
        self.grid.addWidget(self.vis_data_group, 0, 0, 4, 1)

        groups = pd.unique(self.parent.PCA.target['target'])
        target_group = CheckBoxGroup(groups, self.parent.data_groups)
        self.gbtn_list = target_group.btn_list
        self.grid.addWidget(target_group, 0, 1, len(groups), 1)

        self.vis_data_group = RadioBtnGroup('data for\nvisualization', ['all data','only train','only test'],
                                            self.parent.active, self.parent.set_visual_data)
        self.grid.addWidget(self.vis_data_group, 0, 2, 3, 1)

        ann_btn = QCheckBox('add_annotations', self)
        if self.parent.add_annotations: ann_btn.toggle()
        ann_btn.stateChanged.connect(partial(show_annotations, self, ann_btn))
        self.grid.addWidget(ann_btn, 4, 2)


class PopMenuDistance(PopMenu):

    def apply_function(self):
        try:
            self.parent.data_groups = [btn.text() for btn in self.gbtn_list if btn.isChecked()]
            self.calc_dist(self.parent.principalDf, self.parent.data_groups,
                           self.parent.comp_for_calc, self.parent.active, self.parent.dist_method)
            self.parent.show_dist()
        except Exception as e:
            print(e)

    def create_buttons(self):

        pc = self.parent.comp_for_calc
        ref = 'components 1-4' if len(pc) == 4 else f'Components {pc[0]} and {pc[1]}'
        self.calc_comp_group = RadioBtnGroup('principal\ncomponents',
                                             ['components 1-4','Components 1 and 2','Components 3 and 4'],
                                             ref, self.parent.set_dist_components)
        self.grid.addWidget(self.calc_comp_group, 0, 0, 3, 1)

        groups = pd.unique(self.parent.PCA.target['target'])
        target_group = CheckBoxGroup(groups, self.parent.data_groups)
        self.gbtn_list = target_group.btn_list
        self.grid.addWidget(target_group, 0, 1, len(groups)+1, 1)

        self.dist_method_group = RadioBtnGroup('method', ['euclidean', 'mahalanobis'],
                                               self.parent.dist_method, self.parent.set_dist_method)
        self.grid.addWidget(self.dist_method_group, 0, 2, 2, 1)

        self.vis_data_group = RadioBtnGroup('data for\nvisualization', ['all data', 'only train', 'only test'],
                                            self.parent.active, self.parent.set_visual_data)
        self.grid.addWidget(self.vis_data_group, 0, 3, 3, 1)

    def calc_dist(self, principalDf, targets, comp_for_calc, active, method='euclidean'):
        all_groups = pd.unique(principalDf['target'])
        targets = targets if targets is not None else all_groups
        groups = []
        components = ['pc %i' % i for i in comp_for_calc]
        for target in targets:
            if active == 'all data':
                indicesToKeep = (principalDf['target'] == target)
            elif active == 'only train':
                indicesToKeep = (principalDf['target'] == target) & (principalDf['Active'] == 1)
            else:  # active == 'only test'
                indicesToKeep = (principalDf['target'] == target) & (principalDf['Active'] == 0)
            groups.append(principalDf.loc[indicesToKeep, components])

        intro_group_dist = np.array([scipy.spatial.distance.pdist(x, method).mean() for x in groups])
        distance = []
        try:
            for gr1, gr2 in combinations(groups, 2):
                distance.append(scipy.spatial.distance.cdist(gr1, gr2, method).mean())
            distance = scipy.spatial.distance.squareform(distance)
            distance = np.hstack((distance, np.reshape(intro_group_dist, (len(targets), 1))))
        except ValueError as e:
            self.parent.dist_information = str(e)
        else:
            self.parent.dist_information = {'name': 'distance',
                                            'targets': targets + ['intragroup dist'],
                                            'matrix': distance}


class PopMenuHist(PopMenu):

    def apply_function(self):
        self.parent.show_hist()

    def create_buttons(self):
        pc = self.parent.comp_for_calc
        ref = 'components 1-4' if len(pc) == 4 else f'Components {pc[0]} and {pc[1]}'

        groups = pd.unique(self.parent.PCA.target['target'])
        if not self.parent.hist_group: self.parent.hist_group = groups[0]

        self.comp = RadioBtnGroup('principal\ncomponents',
                                  ['components 1-4', 'Components 1 and 2', 'Components 3 and 4',
                                   'Components 2 and 3', 'Components 1 and 3',], ref,
                                   self.parent.set_dist_components)

        self.grid.addWidget(self.comp, 0, 0, 4, 1)
        self.vis_data_group = RadioBtnGroup('data for\nvisualization', ['all data','only train','only test'],
                                            self.parent.active, self.parent.set_visual_data)
        self.grid.addWidget(self.vis_data_group, 0, 2, 3, 1)

        target_group = RadioBtnGroup('group',groups, self.parent.hist_group, self.parent.set_hist_group)
        self.grid.addWidget(target_group, 0, 1, len(groups)+3, 1)


class PopMenuCluster(PopMenu):

    def apply_function(self):

        self.parent.data_groups = [btn.text() for btn in self.gbtn_list if btn.isChecked()]

        self.parent.show_cluster()

    def create_buttons(self):
        pc = self.parent.comp_for_calc
        ref = 'components 1-4' if len(pc) == 4 else f'Components {pc[0]} and {pc[1]}'
        self.calc_comp_group = RadioBtnGroup('principal\ncomponents',
                                             ['components 1-4', 'Components 1 and 2', 'Components 3 and 4'],
                                             ref, self.parent.set_dist_components)
        self.grid.addWidget(self.calc_comp_group, 0, 0, 3, 1)

        groups = pd.unique(self.parent.PCA.target['target'])
        target_group = CheckBoxGroup(groups, self.parent.data_groups)
        self.gbtn_list = target_group.btn_list
        self.grid.addWidget(target_group, 0, 2, len(groups)+2, 1)

        self.method_group = RadioBtnGroup('method', ['K-means', 'Birch','Agglomerative\nClustering'],
                                          self.parent.cluster_method,
                                          self.parent.set_cluster_method)
        self.grid.addWidget(self.method_group, 0, 3, 4, 1)

        self.vis_data_group = RadioBtnGroup('data for\nvisualization', ['all data', 'only train', 'only test'],
                                            self.parent.active, self.parent.set_visual_data)
        self.grid.addWidget(self.vis_data_group, 0, 5, 3, 1)

        pc = self.parent.princ_cmp
        self.comp = RadioBtnGroup('axes for\n visualisation',
                                  ['Components 1 and 2', 'Components 3 and 4',
                                   'Components 1 and 3', 'Components 2 and 3', ],
                                  f'Components {pc[0]} and {pc[1]}',
                                  self.parent.set_components)
        self.grid.addWidget(self.comp, 0, 1, 4, 1)

        clusters_num_box = ClusterComboBox(self.parent)
        self.grid.addWidget(clusters_num_box, 0, 4, 2, 1)

        ann_btn = QCheckBox('add_annotations', self)
        if self.parent.add_annotations: ann_btn.toggle()
        ann_btn.stateChanged.connect(partial(show_annotations, self, ann_btn))
        self.grid.addWidget(ann_btn, 4, 5)


def show_annotations(cls, ann_btn):
    if ann_btn.isChecked():
        cls.parent.add_annotations = True
    else:
        cls.parent.add_annotations = False
