from PyQt5.QtWidgets import (QWidget, QCheckBox, QVBoxLayout, QGridLayout, QButtonGroup,
                             QPushButton, QLabel, QRadioButton, QTableWidget, QTableWidgetItem,
                             QSizePolicy, QAbstractScrollArea, QComboBox)
from PyQt5.QtCore import QSize, pyqtSignal
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon
import numpy.random.common
import numpy.random.bounded_integers
import numpy.random.entropy
import numpy as np
import pandas as pd
import os.path
from functools import partial
from itertools import combinations
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import scipy.spatial.distance
from scipy import stats
from plots import ComponentHist, PlotCanvas
from sklearn.cluster import KMeans, Birch, AgglomerativeClustering


class MatrixWidget(QWidget):
    """
    information : dict, fields: name : str , targets: list(str), matrix np.array or pd.DataFrame
    """

    def __init__(self, information, parent):
        super().__init__()
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        v_box = QVBoxLayout()

        if isinstance(information, str):
            self.setWindowTitle('error')
            self.text = QLabel(information)
            v_box.addWidget(self.text)

        else:
            self.setWindowTitle(information['name'])
            targets = information['targets']
            self.matrix = information['matrix']

            if information['name'] in ['U-test']:
                self.matrix = self.matrix.round(20)
            else:
                self.matrix = self.matrix.round(4)
            self.table = QTableWidget(self)  # Создаём таблицу
            v_box.addWidget(self.table)
            self.table.setColumnCount(self.matrix.shape[1])
            self.table.setRowCount(self.matrix.shape[0])

            if information['name'] in ['distance', 'clusters', 'U-test']:
                self.table.setVerticalHeaderLabels(targets)
            if information['name'] in ['loadings','explained variance']:
                self.table.setVerticalHeaderLabels(['pc%i' %i for i in (1, 2, 3, 4)])

            self.table.setHorizontalHeaderLabels(targets)

            if isinstance(self.matrix, pd.DataFrame):
                self.matrix = self.matrix.to_numpy()

            for x in range(self.matrix.shape[0]):
                for y in range(self.matrix.shape[1]):
                    self.table.setItem(x, y, QTableWidgetItem(str(self.matrix[x, y])))

            self.table.setSizeAdjustPolicy(QAbstractScrollArea.AdjustToContents)
            self.table.setStyleSheet("QTableWidget::item { padding: 3px }")
            self.table.resizeColumnsToContents()
            self.table.cellChanged[int, int].connect(self.changed_row_column)
            save = parent.create_save_btn(self.matrix, targets)
            v_box.addWidget(save)
            self.setLayout(v_box)

    def changed_row_column(self, r, c):
        self.matrix[r, c] = float(self.table.item(r, c).text())


class RadioBtnGroup(QWidget):
    def __init__(self, label, btn_names, reference, function):
        super().__init__()
        self.data_group = QButtonGroup()
        v_box = QVBoxLayout()
        self.setLayout(v_box)
        self.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum)
        v_box.addWidget(QLabel(label))

        for i, btn_name in enumerate(btn_names):
            btn = QRadioButton(btn_name, self)
            if btn_name == reference: btn.setChecked(True)
            btn.clicked.connect(function)
            self.data_group.addButton(btn, i)
            v_box.addWidget(btn)


class CheckBoxGroup(QWidget):
    def __init__(self, targets, reference):
        super().__init__()
        v_box = QVBoxLayout()
        self.setLayout(v_box)
        v_box.addWidget(QLabel('groups'))
        self.btn_list = []
        for i, group in enumerate(targets):
            btn = QCheckBox(group, self)
            v_box.addWidget(btn)
            if not reference or btn.text() in reference:
                btn.toggle()
            self.btn_list.append(btn)


class HistWidget(QWidget):
    _count = 0
    close_handler = pyqtSignal()

    def __init__(self, parent):
        super().__init__()
        self.close_handler.connect(partial(self.clear, parent, __class__._count))
        __class__._count += 1
        self.parent = parent
        self.square_matrix, self.indices_to_keep, self.group_matrix = self.calc_distance()
        self.initUI()

    def initUI(self):
        canvas = ComponentHist(width=12, height=6)
        canvas.plot_hist(self.square_matrix.mean(axis=1), self.parent.principalDf, self.indices_to_keep,
                         self.parent.hist_group)
        self.navi_toolbar = NavigationToolbar(canvas, self)
        self.grid = QGridLayout()
        self.setLayout(self.grid)
        self.grid.addWidget(self.navi_toolbar, 0, 0, 1, 10)
        self.grid.addWidget(canvas, 1, 0, 2, 10)
        self.show_dist_btn = QPushButton('show all dist matrix', self)
        self.show_dist_btn.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
        self.grid.addWidget(self.show_dist_btn, 3, 0, 3, 1)
        self.show_dist_btn.clicked.connect(self.show_all_matrix)
        self.show_gr_dist_btn = QPushButton('show group dist matrix', self)
        self.show_gr_dist_btn.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
        self.grid.addWidget(self.show_gr_dist_btn, 3, 1, 3, 2)
        self.show_gr_dist_btn.clicked.connect(self.show_group_matrix)

    def calc_distance(self):
        comp = self.parent.comp_for_calc
        group = self.parent.hist_group
        data = self.parent.principalDf
        active = self.parent.active
        distance = scipy.spatial.distance.pdist(data[['pc %i' % i for i in comp]])
        square = scipy.spatial.distance.squareform(distance)
        if active == 'all data':
            indicesToKeep = (data['target'] == group)
        elif active == 'only train':
            indicesToKeep = (data['target'] == group) & (data['Active'] == 1)
        else:  # active == 'only test'
            indicesToKeep = (data['target'] == group) & (data['Active'] == 0)
        group_matrix_distance = scipy.spatial.distance.pdist(data.loc[indicesToKeep, ['pc %i' % i for i in comp]])
        group_matrix = scipy.spatial.distance.squareform(group_matrix_distance)
        return square, indicesToKeep, group_matrix

    def show_all_matrix(self):
        try:
            self.information = {'name': 'distance',
                                'targets': [str(x) for x in range(len(self.square_matrix))],
                                'matrix': self.square_matrix}
            self.dist_matrix = MatrixWidget(self.information, self.parent)
            self.dist_matrix.show()
        except Exception as e:
            print(e)

    def show_group_matrix(self):
        try:
            self.information = {'name': 'distance',
                                'targets': [str(x) for i, x in zip(self.indices_to_keep,
                                                                   range(len(self.square_matrix))) if i],
                                'matrix': self.group_matrix}
            self.group_dist_matrix = MatrixWidget(self.information, self.parent)
            self.group_dist_matrix.show()
        except Exception as e:
            print(e)

    def closeEvent(self, event):
        self.close_handler.emit()

    def clear(self, parent, num):
        if num >= len(parent.hist_widgets):
            parent.hist_widgets.pop(-1)
        else:
            parent.hist_widgets.pop(num)
        __class__._count -= 1


class ClusterWidget(QWidget):
    _count = 0
    close_handler = pyqtSignal()

    def __init__(self, parent):
        super().__init__()
        self.close_handler.connect(partial(self.clear, parent, __class__._count))
        __class__._count += 1
        self.parent = parent
        self.clusters, self.data_for_vis, self.ind_to_keep = self.preparing_and_clustering
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Clusters')
        self.setWindowIcon(QIcon(os.path.join('icons', 'clusters.png')))
        self.setMinimumSize(500, 500)
        self.grid = QGridLayout(self)
        self.canvas = PlotCanvas(width=8, height=7)
        self.navi_toolbar = NavigationToolbar(self.canvas, self)
        self.vbl = QVBoxLayout()
        self.vbl.addWidget(self.navi_toolbar)
        self.vbl.addWidget(self.canvas)

        self.grid.addLayout(self.vbl, 0, 0, 5, 6)
        self.setLayout(self.grid)

        self.canvas.plot(self.data_for_vis, self.parent.princ_cmp, active=self.parent.active,
                         text=' '.join([self.parent.cluster_method, '\n',
                                        str(self.parent.clusters_num), 'clusters']),
                         annotations=self.parent.add_annotations)
        matrix = self.parent.principalDf.loc[self.ind_to_keep]
        matrix['cluster'] = self.clusters
        matrix['id'] = self.parent.principalDf.index.values[self.ind_to_keep]

        cluster_table = {'name': 'cluster', 'targets': matrix.columns.values, 'matrix': matrix}
        table = MatrixWidget(cluster_table, self.parent)
        self.grid.addWidget(table, 0, 6, 5, 1)
        apr_cl_btn = QPushButton('approve clusters')
        apr_cl_btn.clicked.connect(lambda: self.aprove_func(table.matrix))
        self.grid.addWidget(apr_cl_btn, 4, 6, 1, 1, alignment=Qt.AlignRight)
        self.show()

    def aprove_func(self, matrix):
        self.data_for_vis['target'] = matrix[:, -2]
        self.canvas.plot(self.data_for_vis, self.parent.princ_cmp, active=self.parent.active,
                         text=' '.join([self.parent.cluster_method, '\n',
                                        str(self.parent.clusters_num), 'clusters']))
        self.table = ClusterDistWidget(self.parent, self.data_for_vis)
        self.table_U_Test = ClusterUTestWidget(self.parent, self.data_for_vis)

    @property
    def preparing_and_clustering(self):
        if self.parent.data_groups:
            ind_to_keep = [True if x in self.parent.data_groups else False for x in
                           self.parent.principalDf['target']]
            data_for_cluster = self.parent.principalDf[ind_to_keep]
        else:
            ind_to_keep = [x for x in range(len(self.parent.principalDf))]
            data_for_cluster = self.parent.principalDf

        if self.parent.cluster_method == 'Birch':
            cl_method = Birch
        elif self.parent.cluster_method == 'K-means':
            cl_method = KMeans
        elif self.parent.cluster_method == 'Agglomerative\nClustering':
            cl_method = AgglomerativeClustering
        data_for_vis = data_for_cluster.loc[:]
        data_for_cluster = data_for_cluster[['pc %i' % i for i in self.parent.comp_for_calc]]
        clusters = cl_method(n_clusters=self.parent.clusters_num).fit_predict(data_for_cluster)
        data_for_vis['target'] = ['cluster %i' % i for i in clusters]
        data_for_vis['Active'] = 1
        return clusters, data_for_vis, ind_to_keep

    def closeEvent(self, event):
        self.close_handler.emit()

    def clear(self, parent, num):
        if num >= len(parent.cluster_widgets):
            parent.cluster_widgets.pop(-1)
        else:
            parent.cluster_widgets.pop(num)
        __class__._count -= 1


class ClusterComboBox(QWidget):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.initUI()

    def initUI(self):
        v_box = QVBoxLayout()
        self.setLayout(v_box)
        v_box.addWidget(QLabel('num of clusters'))
        self.combo = QComboBox(self)
        v_box.addWidget(self.combo)
        self.combo.addItems(["2", '3', '4', '5', '6', '7', '8', '9', '10'])
        self.combo.setCurrentIndex(self.parent.clusters_num - 2)
        self.combo.activated[str].connect(self.on_activated)

    def on_activated(self, text):
        self.parent.clusters_num = int(text)
        print(self.parent.clusters_num)


class ClusterDistWidget(QWidget):
    def __init__(self, parent, matrix):
        super().__init__()
        self.parent = parent
        self.clusters = np.unique(matrix['target'])
        self.matrix = matrix
        self.values = self.calc_dist()
        self.initUI()

    def calc_dist(self):
        dist_dict = {}
        for group in self.clusters:
            if self.parent.active == 'all data':
                indicesToKeep = (self.matrix['target'] == group)
            elif self.parent.active == 'only train':
                indicesToKeep = (self.matrix['target'] == group) & (self.matrix['Active'] == 1)
            else:  # active == 'only test'
                indicesToKeep = (self.matrix['target'] == group) & (self.matrix['Active'] == 0)
            df_s_group = self.matrix.loc[indicesToKeep, ['pc %i' % i for i in self.parent.comp_for_calc]]
            not_df_s_group = self.matrix.loc[[not x for x in indicesToKeep],
                                             ['pc %i' % i for i in self.parent.comp_for_calc]]
            dist_dict[group] = [df_s_group, not_df_s_group]
        vals = []
        all_dist = scipy.spatial.distance.pdist(self.matrix[['pc %i' % i for i in self.parent.comp_for_calc]])
        for group, (first, second) in dist_dict.items():
            clust_dist = scipy.spatial.distance.cdist(first, second).ravel()
            vals.append(clust_dist.mean())
        vals.append(all_dist.mean())
        return np.array(vals).reshape(1, len(vals))

    def initUI(self):
        self.matrix = MatrixWidget({'name': 'distance cluster',
                                    'targets': [str(int(x)) for x in self.clusters] + ['all dist'],
                                    'matrix': self.values}, self.parent)
        self.matrix.show()


class ClusterUTestWidget(QWidget):
    def __init__(self, parent, matrix):
        super().__init__()
        self.parent = parent
        self.clusters = np.unique(matrix['target'])
        self.matrix = matrix
        self.values = self.calc_u_test()
        self.initUI()

    def calc_u_test(self):
        dist_dict = {}
        for group in self.clusters:
            if self.parent.active == 'all data':
                indicesToKeep = (self.matrix['target'] == group)
            elif self.parent.active == 'only train':
                indicesToKeep = (self.matrix['target'] == group) & (self.matrix['Active'] == 1)
            else:  # active == 'only test'
                indicesToKeep = (self.matrix['target'] == group) & (self.matrix['Active'] == 0)
            df_s_group = self.matrix.loc[indicesToKeep, ['pc %i' % i for i in self.parent.comp_for_calc]]
            not_df_s_group = self.matrix.loc[[not x for x in indicesToKeep],
                                             ['pc %i' % i for i in self.parent.comp_for_calc]]
            dist_dict[group] = [df_s_group, not_df_s_group]
        p_vals = []
        all_dist = scipy.spatial.distance.pdist(self.matrix[['pc %i' % i for i in self.parent.comp_for_calc]])
        for group, (first, second) in dist_dict.items():
            clust_dist = scipy.spatial.distance.cdist(first, second).ravel()
            p_vals.append(stats.mannwhitneyu(all_dist, clust_dist, alternative='two-sided')[1])
        return np.array(p_vals).reshape(1, len(p_vals))

    def initUI(self):
        self.matrix = MatrixWidget({'name': 'U-test cluster',
                                    'targets': [str(int(x)) for x in self.clusters],
                                    'matrix': self.values}, self.parent)
        self.matrix.show()


class UTestWidget(QWidget):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.principalDf = self.parent.principalDf
        self.data_groups = self.parent.data_groups
        if self.data_groups is None:
            self.data_groups = pd.unique(self.principalDf['target'])
        self.values = self.calc_u_test
        self.initUI()

    @property
    def calc_u_test(self):
        dist_dict = {}
        for group in self.data_groups:
            df_s_group = self.principalDf[self.principalDf['target'] == group]
            if self.parent.active == 'all data':
                indicesToKeep = (self.principalDf['target'] == group)
            elif self.parent.active == 'only train':
                indicesToKeep = (self.principalDf['target'] == group) & (self.principalDf['Active'] == 1)
            else:  # active == 'only test'
                indicesToKeep = (self.principalDf['target'] == group) & (self.principalDf['Active'] == 0)
            df_s_group = df_s_group.loc[indicesToKeep, ['pc %i' % i for i in self.parent.comp_for_calc]]
            dist_dict[group] = scipy.spatial.distance.pdist(df_s_group)
        p_vals = [stats.mannwhitneyu(first[1], second[1], alternative='two-sided')[1]
                  for first, second in combinations(dist_dict.items(), 2)]
        return scipy.spatial.distance.squareform(p_vals)

    def initUI(self):
        self.matrix = MatrixWidget({'name': 'U-test',
                                    'targets': self.data_groups,
                                    'matrix': self.values}, self.parent)
