import os.path, os
from PyQt5.QtWidgets import QToolButton
from PyQt5.QtCore import QSize, pyqtSignal
from PyQt5.QtGui import QIcon
from sklearn import decomposition, preprocessing
import sys
import numpy.random.common
import numpy.random.bounded_integers
import numpy.random.entropy
from functools import partial
import pandas as pd

from pop_menu import *
from widgets import *
from plots import PlotCanvas


class PCApp(QWidget):

    def __init__(self):
        super().__init__()
        self.active = 'all data'
        self.princ_cmp = (1, 2)
        self.data_groups = None
        self.train_data_groups = None
        self.included_features = None
        self.comp_for_calc = (1, 2, 3, 4)
        self.dist_method = 'euclidean'
        self.cluster_method = 'K-means'
        self.dist_information = ''
        self.hist_group = None
        self.hist_widgets = []
        self.cluster_widgets = []
        self.btn_dict = {}
        self.clusters_num = 2
        self.add_annotations = True
        self.initUI()

    def initUI(self):
        self.setWindowTitle('World PCA')
        self.setWindowIcon(QIcon(os.path.join('icons', 'world.png')))
        self.setMinimumSize(500, 500)
        self.grid = QGridLayout(self)
        self.canvas = PlotCanvas(width=8, height=7)
        self.navi_toolbar = NavigationToolbar(self.canvas, self)
        self.vbl = QVBoxLayout()
        self.vbl.addWidget(self.navi_toolbar)
        self.vbl.addWidget(self.canvas)
        self.grid.addLayout(self.vbl, 0, 3, 6, 1)

        names = ('load base', 'fit', 'analyze', 'distance', 'hist', 'clustering')
        functions = (self.load_click, self.fit, self.transform, self.distance, self.hist, self.cluster)
        nums = ((0, 5), (0, 6), (0, 7), (1, 7), (1, 6), (1, 5))
        for name, num, func in zip(names, nums, functions):
            btn = QPushButton(name, self)
            self.btn_dict[name] = btn
            if name != 'load base':
                btn.setEnabled(False)
            btn.clicked.connect(func)
            self.grid.addWidget(btn, *num)

        self.setLayout(self.grid)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.show()

    def load_click(self):
        try:
            self.base_name = QFileDialog.getOpenFileName(self, 'Open file', '')[0]
            self.btn_dict['fit'].setEnabled(True)
        except Exception as e: print(e)

    def distance(self):
        win = PopMenuDistance(parent=self)
        win.show()

    def show_dist(self):
        self.dist_matrix = MatrixWidget(self.dist_information, self)
        self.dist_matrix.show()
        self.U_matrix = UTestWidget(self)
        self.U_matrix.matrix.show()

    def fit(self):
        self.PCA = PCA(self.base_name)
        try:
            win = PopMenuFeatures(parent=self)
            win.show()
        except Exception as e: print(e)
        self.btn_dict['analyze'].setEnabled(True)

    def transform(self):
        win = PopMenuPlot(parent=self)
        win.show()
        for btn_name in  ('distance', 'hist','clustering'):
            self.btn_dict[btn_name].setEnabled(True)

    def hist(self):
        win = PopMenuHist(parent=self)
        win.show()

    def cluster(self):
        try:
            win = PopMenuCluster(parent=self)
            win.show()
        except Exception as e:print(e)

    def show_hist(self):
        self.hist_widgets.append(HistWidget(parent=self))
        self.hist_widgets[-1].show()

    def show_cluster(self):
        self.cluster_widgets.append(ClusterWidget(parent=self))
        self.cluster_widgets[-1].show()

    def set_visual_data(self):
        source = self.sender()
        print(source.text())
        self.active = source.text()

    def set_components(self):
        source = self.sender()
        print(source.text())
        self.princ_cmp = tuple(int(y) for y in source.text() if y.isnumeric())

    def set_hist_group(self):
        source = self.sender()
        self.hist_group = source.text()

    def set_dist_components(self):
        source = self.sender()
        print(source.text())
        if source.text() == 'components 1-4':
            self.comp_for_calc = (1, 2, 3, 4)
        else:
            self.comp_for_calc = tuple(int(y) for y in source.text() if y.isnumeric())

    def set_dist_method(self):
        source = self.sender()
        self.dist_method = source.text()

    def set_cluster_method(self):
        source = self.sender()
        self.cluster_method = source.text()

    def create_pc_table(self):
        table = QTableWidget(self)  # Создаём таблицу
        pDf = self.principalDf

        indices = list(map(lambda x: x in self.data_groups, pDf['target']))
        id = pDf.index.values[pDf.index.values[indices]]
        pDf = pDf.loc[indices]
        pDf.insert(0, 'id', id)
        if self.active == 'only train':
            pDf = pDf.loc[pDf['Active'] == 1]
        elif self.active == 'only test':
            pDf = pDf.loc[pDf['Active'] == 0]
        pDf = pDf.round(3)

        table.setColumnCount(len(pDf.columns.values))
        table.setRowCount(pDf.shape[0])
        table.setHorizontalHeaderLabels(pDf.columns.values)
        for x in range(pDf.shape[0]):
            for y in range(pDf.shape[1]):
                table.setItem(x, y, QTableWidgetItem(str(pDf.iat[x, y])))

        table.resizeColumnsToContents()  # делаем ресайз колонок по содержимому
        self.grid.addWidget(table, 2, 5, 1, 18)  # Добавляем таблицу в сетку

        save_btn = self.create_save_btn(pDf, pDf.columns.values)
        self.grid.addWidget(save_btn, 4, 5, 2, 7)

        var_btn = QPushButton('show explained variance', self)
        var_btn.clicked.connect(self.show_exp_var)
        self.grid.addWidget(var_btn, 4, 7, 2, 14)

        load_btn = QPushButton('show loadings', self)
        load_btn.clicked.connect(self.show_loadings)
        self.grid.addWidget(load_btn, 8, 7, 2, 14)

    def create_save_btn(self, matrix, targets):
        button = QToolButton(self)
        button.setIcon(QIcon(os.path.join('icons', 'save_icon.png')))
        button.setIconSize(QSize(30, 18))
        button.clicked.connect(lambda: self.save_data(matrix, targets))
       
        return button

    def show_exp_var(self):
        matrix = np.vstack([self.PCA.pca.explained_variance_, self.PCA.pca.explained_variance_ratio_]).T
        targets =('eigenvalues', 'explained_variance_ratio')
        information = {'name': 'eigenvalues', 'targets': targets, 'matrix': matrix}
        self.exp_var = MatrixWidget(information, self)
        self.exp_var.show()

    def show_loadings(self):
        matrix = (self.PCA.pca.components_)
        targets = self.included_features
        information = {'name': 'loadings', 'targets': targets, 'matrix': matrix}
        self.loadings = MatrixWidget(information, self)
        self.loadings.show()

    def save_data(self, data, targets = None):
        """
        :param data: must be pd.DataFrame or np.ndarray
        """
        if isinstance(data, np.ndarray) and targets is not None:
            data = pd.DataFrame(data, columns=targets)
        elif isinstance(data, np.ndarray) and targets is None:
            data = pd.DataFrame(data)
        elif not isinstance(data, (np.ndarray, pd.DataFrame)):
            raise ValueError("data type must be pd.DataFrame or np.ndarray")
        try:
            path, _ = QFileDialog.getSaveFileName(self, "Change name", 'untitled.xlsx', 'xlsx (*.xlsx)')
            data.to_excel(path, sheet_name='sheet1', index=False)
        except Exception as e:print(e)


class PCA:
    def __init__(self, data_name):
        df = pd.read_excel(data_name)

        self.target = df[['target']]
        self.active = df[['Active']]

        indicesToKeep = df['Active'] == 1
        self.df_train = df.loc[indicesToKeep]
        self.df = df[[x for x in df.columns.values if x not in ['Active', 'target', 'Title']]]
        self.pca = decomposition.PCA(n_components=4)

    def fit(self, train=None):
        if not train: train = self.df_train
        train = train[[x for x in train.columns.values if x not in ['Active', 'target', 'Title']]]
        train = train.to_numpy()
        normalized_train = preprocessing.scale(train, axis=0)
        self.pca.fit(normalized_train)

    def transform(self, data=None):
        if not data: data = self.df
        data = data.to_numpy()
        normalized_data = preprocessing.scale(data,axis=0)
        principalComponents = self.pca.transform(normalized_data)
        principalDf = pd.DataFrame(data=principalComponents,
                                   columns=['pc 1', 'pc 2', 'pc 3', 'pc 4'])

        principalDf['target'] = self.target
        principalDf['Active'] = self.active
        return principalDf


if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    im = PCApp()
    sys.exit(app.exec_())
