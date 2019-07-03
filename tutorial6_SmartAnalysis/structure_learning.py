import copy
import math
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.stats import stats 

import numpy as np
import scipy as sp
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid
import pdb

# Import AcrGIS python API
from arcgis import GIS
from arcgis.features import SpatialDataFrame, GeoAccessor
from arcgis.features.analysis import join_features

class StructureLearning():
    """ Object which performs structure learning on a pandas dataframe object,
        derived from ArcGIS data features. """
    def __init__(self, df, analysis_columns=None, per_capita_columns=None, 
        population_column=None, categorical_columns=None, 
        correlation_measure='mutual_information', BINS=20, inplace=False):
        """ Initialize StructureLearning object.
        Keyword arguments:
        df -- pandas data frame containing each of the analysis_columns as a column
        analysis_columns -- columns in the df to analyze
        per_captia_columns -- columns which should be treated as per captia values
                and divdied by population column
        population_column -- column containing population for each row/attribute
        BINS -- mutual information requires catgorical varaibles. Variables will
                be binned into BIN evenely spaced bins (default: 20).
        inplace -- flag for modifying the df in place or creating a copy
                within the object (default: False)
        """
        if inplace:
            self.df = df
        else:
            self.df = copy.copy(df)
            # TODO: maybe don't need to copy whole dataframe, since we only
            # use the analysis columns

        if analysis_columns is None:
            self.analysis_columns = df.columns
        else: 
            self.analysis_columns = analysis_columns

        if per_capita_columns is not None and population_column is None:
            raise ValueError('If <per_capita_columns> is specified, <population_column> must also be specified.')

        self.per_capita_columns = per_capita_columns
        self.population_column = population_column
        self.categorical_columns = categorical_columns
        self.BINS = BINS

        metric_options = ['mutual_information',
                          'spearman_rank', 
                          'pearson_rho',
                          'gaussian_correlation',
                          'fft_distance_correlation']
        if correlation_measure not in metric_options:
            raise ValueError('Correlation measure must be in:', metric_options)
        self.metric = correlation_measure

        # Initialize all graph matrices
        self._weight_matrix = None
        self._maximum_spanning_graph = None
        self._relevence_graph = None
        self.relevence_thresh = None

        # Normalize and discretize data
        self.prepare_data()

    def prepare_data(self):
        '''' Discretize data and normalize the data in per captia columns. '''
        print("Preparing data columns:")

        # self.df = self.df[self.df[self.population_column] != 0]

        # Remove features where the population equals zero
        self.df.drop(self.df[self.df[self.population_column] == 0.0].index, inplace=True)

        if self.categorical_columns is None:
            cols = self.analysis_columns
            num_cols = self.df._get_numeric_data().columns
            self.categorical_columns = list(set(cols) - set(num_cols)) 

        print("Treating the following columns as categorical data:", self.categorical_columns)

        # print(self.df[self.population_column])
        for column in self.analysis_columns:
            print(column)
            # Cannot normalize or discretize categorical data, so continue

            if column in self.categorical_columns:
                self.df[column + '_analysis'] = self.df[column] 
                continue
            # If the column should be measured in units/population, divide by population column
            if self.per_capita_columns is not None and column in self.per_capita_columns:
                self.df[column + '_analysis'] = self.df[column] * \
                    1.0 / self.df[self.population_column] * 1.0
            else:
                self.df[column + '_analysis'] = self.df[column] * 1.0

            # Mutual information requres discrete/categorical data, so bin the columns into BIN evenely spaced bins
            # if self.metric == "mutual_information":
            #     self.df[column + '_analysis'] = pd.cut(self.df[column + '_analysis'], bins=self.BINS, labels=range(
            #         self.BINS), precision=8, duplicates='raise')
            if len(self.categorical_columns) > 0:
                self.df[column + '_analysis_bin'] = pd.cut(self.df[column + '_analysis'], bins=self.BINS, labels=range(
                    self.BINS), precision=8, duplicates='raise')

    def pairwise_mutual_information(self, x_label, y_label):
        ''' Compute the mutual information between two columns in the data frame, specified by x_label and y_label.
        Keyword arguments:
        x_label: string containing the name of the first column
        y_label: string containing the name of the second column
        '''

        if any(x in x_label for x in self.categorical_columns) or any(y in y_label for y in self.categorical_columns):
            # If either attribute is a categorical variable, compute discrete entropy on binned data
            mutual_information = 0.0

            # Get unique elements for x and y variables
            try: 
                x_bins = self.df[x_label].unique()
                y_bins = self.df[y_label].unique()

                # Get the empirical counts for each value of x and y variable
                x_terms = self.df[x_label].value_counts() / self.df.shape[0]
                y_terms = self.df[y_label].value_counts() / self.df.shape[0]

                # Get the empirical pairwise counts for the combination of x and y values
                cross_terms = self.df[[x_label, y_label]].groupby(
                    [x_label, y_label]).size() / (self.df.shape[0])

            # If unique fails, can cast as a string and run unique grouping again
            except (TypeError, AttributeError) as e: 
                try:
                    x_bins = self.df[x_label].astype(str).unique()
                    y_bins = self.df[y_label].astype(str).unique()

                    # Get the empirical counts for each value of x and y variable
                    x_terms = self.df[x_label].astype(str).value_counts() / self.df.shape[0]
                    y_terms = self.df[y_label].astype(str).value_counts() / self.df.shape[0]

                    # Get the empirical pairwise counts for the combination of x and y values
                    cross_terms = self.df[[x_label, y_label]].astype(str).groupby(
                        [x_label, y_label]).size() / (self.df.shape[0])

                except (TypeError, AttributeError) as e: 
                    print("Cannot extract unique values from this dataype for comparision of:", x_label, "and", y_label)
                    print(e)


            # Compute the MI between x and y values, MI(X; Y) = H(X) - H(X | Y) = H(Y) - H(Y | X)
            for x in x_bins:
                for y in y_bins:
                    if (x, y) in cross_terms:
                        mutual_information += cross_terms[x, y] * np.log2(
                            cross_terms[x, y] * 1.0 / (x_terms[x] * y_terms[y] * 1.0))

            # Compute the entropy of attribute x
            h_x = 0.0
            for x in x_bins:
                if not np.isclose(x_terms[x], 0.0):
                    h_x += -x_terms[x] * np.log2(x_terms[x])

            # Compute the entropy of attribute y
            h_y = 0.0
            for y in y_bins:
                if not np.isclose(x_terms[x], 0.0):
                    h_y += -y_terms[y] * np.log2(y_terms[y])

            # Compute normalized mutual information by dividing by h_x + h_y
            return 2.0*mutual_information / (h_x + h_y)
        else:
            # Fit a multivaraite Gaussian distirbution to the data and compute mutual information 
            mu_hat = np.array([self.df[x_label].mean(), self.df[y_label].mean()]).reshape(1, 2)
            centered_data = self.df[[x_label, y_label]] - mu_hat
            sigma_hat = 1.0 / centered_data.shape[0] * centered_data.T.dot(centered_data)

            h_x = np.log(sigma_hat[x_label][x_label] * np.sqrt(2 * np.pi * np.e))
            h_y = np.log(sigma_hat[y_label][y_label] * np.sqrt(2 * np.pi * np.e))

            rho = sigma_hat[x_label][y_label] / (np.sqrt(sigma_hat[x_label][x_label]) * np.sqrt(sigma_hat[y_label][y_label]))
            mutual_information = -0.5 * np.log(1 - rho**2)

            # rho = sigma_hat[x_label][y_label] / (np.sqrt(sigma_hat[x_label][x_label]) * np.sqrt(sigma_hat[y_label][y_label]))
            # mutual_information = 0.5*np.log(sigma_hat[x_label][x_label] * sigma_hat[y_label][y_label] / np.linalg.det(sigma_hat))
            return mutual_information
        # print(h_x, h_y, mutual_information)

    def pairwise_gaussian_correlation(self, x_label, y_label):
        mu_hat = np.array([self.df[x_label].mean(), self.df[y_label].mean()]).reshape(1, 2)
        centered_data = self.df[[x_label, y_label]] - mu_hat
        sigma_hat = 1.0 / centered_data.shape[0] * centered_data.T.dot(centered_data)

        h_x = np.log(sigma_hat[x_label][x_label] * np.sqrt(2 * np.pi * np.e))
        h_y = np.log(sigma_hat[y_label][y_label] * np.sqrt(2 * np.pi * np.e))

        rho = sigma_hat[x_label][y_label] / (np.sqrt(sigma_hat[x_label][x_label]) * np.sqrt(sigma_hat[y_label][y_label]))

        return rho 

    def pairwise_spearman_rank_order(self, x_label, y_label):
        ''' Compute the Spearman rank-order correlation coefficient to detect monotonic relationships between variables. '''
        rho, p_value = sp.stats.stats.spearmanr(
            self.df[x_label], self.df[y_label])
        return rho  # TODO: what is the best way to incorporate/return. the p-value

    def pairwise_pearsons_rho(self, x_label, y_label):
        ''' Compute the Pearson correlation for linear relationship between variables. '''
        rho, p_value = sp.stats.stats.pearsonr(self.df[x_label], self.df[y_label])
        return rho

    def fft_distance_correlation(self, x_label, y_label, FILTER=False):
        ''' Compute the Pearson correlation for linear relationship between variables using the FFT method. '''
        x_hat = (self.df[[x_label, y_label]] - np.mean(self.df[[x_label, y_label]], axis=0)) \
                / np.std(self.df[[x_label, y_label]], axis=0) 

        Fs = 1.0 # sampling rate
        N = x_hat.shape[0] # data length
        # N_fft = (2**int(np.ceil(np.log2(N))))
        N_fft = N

        # X_hat = 1.0 / N_fft * x_hat.apply(np.fft.fft, n=N_fft, axis=0, norm=None)
        X_hat = x_hat.apply(np.fft.fft, n=N_fft, axis=0, norm='ortho')

        # Remove low magnitude components
        if FILTER:
            eps = np.max(np.abs(X_hat), axis=0) / 100
            X_hat[np.any(X_hat < eps, axis=1)] = 0.0

        # Compute the frequency bins
        bins = np.arange(N_fft) * Fs / N_fft # frequency bins
      
        # Use only CUTOFF number of Fourier coefficients 
        CUTOFF = 75
        nyquist_num = min(N_fft // 2 + 1, CUTOFF)
        X_hat = X_hat[0:nyquist_num]
        bins = bins[0:nyquist_num]

        # Compute the magnitude and phase of the complex signal
        mag = X_hat.abs()
        phase = np.arctan2(np.imag(X_hat), np.real(X_hat)) / np.pi * 180.0

        # Compute the appropriate normalization for the FFT for correlation
        # X_hat = np.sqrt(N_fft) * X_hat
        dist_freq = sp.spatial.distance.euclidean(X_hat[x_label], X_hat[y_label])
        rho_freq = 1 - (dist_freq**2 / (2 * X_hat.shape[0]))

        COMPARE = True 
        if COMPARE:
            rho, p_value = stats.pearsonr(self.df[x_label], self.df[y_label])
            print("Diffence between approx and real:", rho - rho_freq)

        return rho_freq

    def build_pairwise_weight_matrix(self):
        ''' Build a matrix containing the pairwise mutual information between each variable in self.analysis_columns. '''
        self._weight_matrix = np.zeros(
            (len(self.analysis_columns), len(self.analysis_columns)))

        for i, x_label in enumerate(self.analysis_columns):
            for j, y_label in enumerate(self.analysis_columns):
                # Only compute for the upper triangle of the matrix
                if i < j:
                    # print("Analyszing:", x_label, "and", y_label)
                    if self.metric == 'mutual_information':
                        weight = self.pairwise_mutual_information(
                            x_label + '_analysis', y_label + '_analysis')
                    elif x_label in self.categorical_columns and y_label in self.categorical_columns:
                        weight = self.pairwise_mutual_information(
                            x_label + '_analysis', y_label + '_analysis')
                    elif x_label in self.categorical_columns and y_label not in self.categorical_columns:
                        weight = self.pairwise_mutual_information(
                            x_label + '_analysis', y_label + '_analysis_bin')
                    elif x_label not in self.categorical_columns and y_label in self.categorical_columns:
                        weight = self.pairwise_mutual_information(
                            x_label + '_analysis_bin', y_label + '_analysis')
                    elif self.metric == 'spearman_rank':
                        weight = self.pairwise_spearman_rank_order(
                            x_label + '_analysis', y_label + '_analysis')
                    elif self.metric == 'pearson_rho':
                        weight = self.pairwise_pearsons_rho(
                            x_label + '_analysis', y_label + '_analysis')
                    elif self.metric == 'fft_distance_correlation':
                        weight = self.fft_distance_correlation(
                            x_label + '_analysis', y_label + '_analysis', FILTER=False)
                    elif self.metric == 'gaussian_correlation':
                        weight = self.pairwise_gaussian_correlation(
                            x_label + '_analysis', y_label + '_analysis')
                    else:
                        weight = self.pairwise_mutual_information(
                            x_label + '_analysis', y_label + '_analysis')
                    self._weight_matrix[i, j] = weight

        # print(self._weight_matrix)

    def build_maximum_spanning_tree(self):
        '''Execute the Chow Liu Tree algorithm, i.e., use the pairwise MI matrix and compute a maximum spanning tree. '''
        # self._chow_liu_graph = -1.0 * minimum_spanning_tree(-1.0 * self.weight_matrix)# Mult by -1.0 to get the maximum spanning tree
        if self.metric == "mutual_information":
            # Mult by -1.0 to get the maximum spanning tree
            maximum_spanning_graph = -1.0 * \
                minimum_spanning_tree(-1.0 * self.weight_matrix).toarray()
        # elif self.metric == "pearson_rho" or self.metric == "spearman_rank" or self.metric == "gaussian_correlation":
        #     # Mult by -1.0 to get the maximum spanning tree and take abs for pos/neg correlation
        #     maximum_spanning_graph = -1.0 * \
        #         minimum_spanning_tree(-1.0 * np.abs(self.weight_matrix)).toarray()
        else:
            maximum_spanning_graph = -1.0 * \
                minimum_spanning_tree(-1.0 * np.abs(self.weight_matrix)).toarray()
            # maximum_spanning_graph = -1.0 * \
            #     minimum_spanning_tree(-1.0 * self.weight_matrix).toarray()

        self._maximum_spanning_graph = copy.copy(self.weight_matrix)
        self._maximum_spanning_graph[maximum_spanning_graph == 0.0] = 0.0

    def build_relevence_network(self, threshhold=0.8):
        ''' Build a relevent graph, containing only edges of weight matrix within threshhold * max(weight_matrix). '''
        self._relevence_graph = copy.copy(self.weight_matrix)
        self.relevence_thresh = threshhold

        # Zero out elements that are less then threshhold times the maximum value
        self._relevence_graph[np.abs(self._relevence_graph) < np.max(
            self._relevence_graph) * threshhold] = 0.0

    def visualize_graph(self, edge_matrix, layout='circular'):
        ''' Visualize an E x E matrix as a weighted adjacency matrix for a graph using networkx. '
        Keyword arguments:
        edge_matrix -- scipy csr_matrix to be interpeted as an E x E weighted adjacency matrix.
        layout -- the graph layout. One of ['circular', 'spring', 'shell', 'kamamda'] (default: 'circular')
        '''
        if self.metric == 'mutual_information':
            cm = plt.cm.Blues
        else:
            # Choose a diverging colormap and flip weight orientation so negative is red
            midpoint = 1.0 - (np.max(self.weight_matrix) / (np.max(self.weight_matrix) + np.abs(np.min(self.weight_matrix))))
            cm = self.shiftedColorMap(plt.cm.coolwarm_r, midpoint=midpoint)  

        # g = nx.from_numpy_matrix(edge_matrix, edge_attribute='weight')
        g = nx.from_numpy_matrix(edge_matrix)

        fig, ax = plt.subplots(1, 1, figsize=(16, 12))
        edges, weights = zip(*nx.get_edge_attributes(g, 'weight').items())

        if layout == 'circular':
            pos = nx.circular_layout(g)
        elif layout == 'spring':
            pos = nx.spring_layout(g)
        elif layout == 'shell':
            pos = nx.shell_layout(g)
        elif layout == 'kamada':
            # Graph weights must be positive for the kamada layout
            # g_temp = nx.from_numpy_matrix(np.abs(edge_matrix), edge_attribute='weight')
            g_temp = nx.from_numpy_matrix(np.abs(edge_matrix))
            pos = nx.kamada_kawai_layout(g_temp)
        elif layout == 'tree':
            pos = nx.hierarchy_pos(g, 1)
        else:
            print("Layout", layout, "not recognized. Defaulting to circular layout.")
            pos = nx.circular_layout(g)

        nx.draw_networkx(G=g,
                         pos=pos,
                         ax=ax,
                         with_labels=True,
                         node_color='k',
                         edgelist=edges,
                         edge_color=(weights),
                         width=10.0,
                         edge_cmap=cm,
                         vmin=np.min(self.weight_matrix),
                         vmax=np.max(self.weight_matrix),
                         node_size=1000,
                         labels=dict(
                             zip(range(len(self.analysis_columns)), self.analysis_columns)),
                         font_size=20,
                         font_color='r')

        plt.show()

    def visualize_full_graph(self, layout='circular'):
        ''' Visualize the full weighted pairwise MI graph. '''

        # print(self.weight_matrix)
        self.visualize_graph(self.weight_matrix, layout)

    def visualize_relevence_graph(self, layout='circular', threshhold=None):
        ''' Visualize the relevence MI graph. '''
        # If visualization threshhold is differnt then the current stored graph, have to overwrite graph
        if threshhold != self.relevence_thresh:
            self._relevence_graph = None

        if threshhold is None:
            self.relevence_thresh = 0.8
        else:
            self.relevence_thresh = threshhold

        self.visualize_graph(self.relevence_graph, layout)

    def visualize_maximum_spanning_tree(self, layout='circular'):
        ''' Visualize the Chow Liu tree. '''
        self.visualize_graph(self.maximum_spanning_graph, layout)

    def visualize_scatter_plots(self, top_n=None, shape=None, use_analysis_vars=False):
        ''' Visualize scatter plots of discretized data, either for all data pairs or for the top_n MI scores.
        Arguments:
        top_n -- plot only the top_n mutual informatino scores (default: None)
        shape -- shape of the resulting subplot (default: (1, top_n))
        '''
        # If top_n is not specified, plot the full N X N matrix of pairwise variable scatter plots
        if shape is not None and (shape[0] * shape[1] != top_n):
            raise ValueError(
                'Specified shape must have the same number of plots as top_n.')

        if use_analysis_vars:
            column_list = [x + '_analysis' for x in self.analysis_columns]
        else:
            column_list = self.analysis_columns

        if top_n is None:
            top_n = len(self.analysis_columns)
            fig, ax = plt.subplots(
                top_n, top_n, figsize=(top_n * 6, top_n * 4))
            for i, x_label in enumerate(column_list):
                for j, y_label in enumerate(column_list):
                    # Only visualize the upper left triangle of the plot matrix
                    if i <= j:
                        ax[i, j].scatter(self.df[x_label],
                                         self.df[y_label], s=10)

                        # Add variable labels to the sides of the grid
                        if i == 0:
                            ax[i, j].set_title(y_label)
                        if j == i:
                            ax[i, 0].set_ylabel(x_label)
        else:
            # Create a plot figure of the specified or default size
            if shape is None:
                fig, ax = plt.subplots(1, top_n, figsize=(6 * top_n, 4))
            else:
                fig, ax = plt.subplots(
                    shape[0], shape[1], figsize=(6 * shape[1], 4 * shape[0]))

            # Sort the entires of the weight matrix in decreasing order
            sort_index = np.argsort(np.abs(self.weight_matrix), axis=None)[::-1]

            count = 0
            ind = 0
            while count < top_n :
                i = math.floor(sort_index[ind] / len(self.analysis_columns))
                j = sort_index[ind] % len(self.analysis_columns)

                # Ignore entries in the lower triangle or diagonal of the matrix
                if i >= j:
                    ind += 1
                    continue

                x_label = column_list[i]
                y_label = column_list[j]

                if shape is None or shape[0] == 1 or shape[1] == 1:
                    ax[count].scatter(self.df[x_label], self.df[y_label], s=10)
                    ax[count].set_title(
                        self.analysis_columns[i] + ' vs ' + self.analysis_columns[j])
                else:
                    ax[math.floor(count / shape[1]), count % shape[1]
                        ].scatter(self.df[x_label], self.df[y_label], s=10)
                    ax[math.floor(count / shape[1]), count % shape[1]
                        ].set_title(self.analysis_columns[i] + ' vs ' + self.analysis_columns[j])
                # Increment plot count and index
                count += 1
                ind += 1

        plt.show()

    def compare_variables(self, x_label, y_label, use_analysis_vars=False):
        if use_analysis_vars:
            x_label = x_label + '_analysis'
            y_label = y_label + '_analysis'

        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        ax.scatter(self.df[x_label], self.df[y_label], s=10)
        ax.set_title(x_label + ' vs ' + y_label)

    @property
    def weight_matrix(self):
        if self._weight_matrix is None:
            self.build_pairwise_weight_matrix()
        return self._weight_matrix

    @property
    def maximum_spanning_graph(self):
        if self._maximum_spanning_graph is None:
            self.build_maximum_spanning_tree()
        return self._maximum_spanning_graph

    @property
    def relevence_graph(self):
        if self._relevence_graph is None:
            self.build_relevence_network(threshhold=self.relevence_thresh)
        return self._relevence_graph

    def shiftedColorMap(self, cmap, start=0, midpoint=0.5, stop=1.0, name='shiftedcmap'):
        '''
        Function to offset the "center" of a colormap. Useful for
        data with a negative min and positive max and you want the
        middle of the colormap's dynamic range to be at zero.

        Input
        -----
          cmap : The matplotlib colormap to be altered
          start : Offset from lowest point in the colormap's range.
              Defaults to 0.0 (no lower offset). Should be between
              0.0 and `midpoint`.
          midpoint : The new center of the colormap. Defaults to 
              0.5 (no shift). Should be between 0.0 and 1.0. In
              general, this should be  1 - vmax / (vmax + abs(vmin))
              For example if your data range from -15.0 to +5.0 and
              you want the center of the colormap at 0.0, `midpoint`
              should be set to  1 - 5/(5 + 15)) or 0.75
          stop : Offset from highest point in the colormap's range.
              Defaults to 1.0 (no upper offset). Should be between
              `midpoint` and 1.0.

        From: phobson
        '''
        cdict = {
            'red': [],
            'green': [],
            'blue': [],
            'alpha': []
        }

        # regular index to compute the colors
        reg_index = np.linspace(start, stop, 257)

        # shifted index to match the data
        shift_index = np.hstack([
            np.linspace(0.0, midpoint, 128, endpoint=False), 
            np.linspace(midpoint, 1.0, 129, endpoint=True)
        ])

        for ri, si in zip(reg_index, shift_index):
            r, g, b, a = cmap(ri)

            cdict['red'].append((si, r, r))
            cdict['green'].append((si, g, g))
            cdict['blue'].append((si, b, b))
            cdict['alpha'].append((si, a, a))

        newcmap = matplotlib.colors.LinearSegmentedColormap(name, cdict)
        plt.register_cmap(cmap=newcmap)

        return newcmap
