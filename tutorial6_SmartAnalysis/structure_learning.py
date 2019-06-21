import pandas as pd
import numpy as np
import copy
import math
import networkx as nx
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree

class StructureLearning(object):
    """ Object which performs structure learning on a pandas dataframe object, derived from ArcGIS data features. """ 
    def __init__(self, df, analysis_columns, per_capita_columns, population_column, BINS=20, inplace=False):
        """ Initialize StructureLearning object.
        Keyword arguments:
        df -- pandas data frame containing each of the analysis_columns as a column 
        analysis_columns -- columns in the df to analyze
        per_captia_columns -- columns which should be treated as per captia values and divdied by population column
        population_column -- column containing population for each row/attribute
        BINS -- mutual information requires catgorical varaibles. Variables will be binned into BIN evenely spaced bins (default: 20).
        inplace -- flag for modifying the df in place or creating a copy within the object (defalut: False)
        """
        if inplace == True:
            self.df = df
        else:
            self.df = copy.copy(df) 
            # TODO: maybe don't need to copy whole dataframe, since we only use the analysis columns

        self.analysis_labels = analysis_columns
        self.analysis_columns = [x + '_analysis' for x in self.analysis_labels]
        self.per_capita_columns = per_capita_columns
        self.population_column = population_column
        self.BINS = BINS


        # Initialize all graph structures and sparse matrices        
        self._weight_matrix = None
        self._weight_matrix_sparse = None
        self._chow_liu_graph = None
        self._relevence_graph = None
        self._relevence_graph_sparse = None
        self.relevence_thresh = None

        # Normalize and discretize data
        self.prepare_data()


    def prepare_data(self):
        '''' Discretize data and normalize the data in per captia columns. '''
        print("Preparing data columns:")
        for column in self.analysis_labels:
            # If the column should be measured in units/population, divide by population column
            if column in self.per_capita_columns:
                self.df[column + '_analysis'] = self.df[column] * 1.0 / self.df[self.population_column] * 1.0
            else:
                self.df[column + '_analysis'] = self.df[column] * 1.0

            # MI computation requres discrete/categorical data, so bin the columns into BIN evenely spaced bins
            self.df[column + '_analysis'] = pd.cut(self.df[column + '_analysis'], bins=self.BINS, labels=range(self.BINS), precision=8, duplicates='raise')
            print(column)

    def pairwise_mutual_information(self, x_label, y_label, COMPUTE_ENTROPY = False):
        ''' Compute the mutual information between two columns in the data frame, specified by x_label and y_label.
        Keyword arguments:
        x_label: string containing the name of the first column
        y_label: string containing the name of the second column
        COMPUTE_ENTROPY: flag specifying whether to optionally compute entropy of each variable and return (default: False).
        '''
        # Initialize mutual information and entropy labels
        MI, H_x, H_y = 0.0, None, None

        # Get unique elements for x and y variables
        x_bins = self.df[x_label].unique()
        y_bins = self.df[y_label].unique()
       
        # Get the empirical counts for each value of x and y variable 
        x_terms = self.df[x_label].value_counts() / self.df.shape[0]
        y_terms = self.df[y_label].value_counts() / self.df.shape[0]

        # Get the empirical pairwise counts for the combination of x and y values
        cross_terms = self.df.groupby([x_label, y_label]).size() / (self.df.shape[0])
       
        # Compute the MI between x and y values, MI(X; Y) = H(X) - H(X | Y) = H(Y) - H(Y | X)
        for x in x_bins:
            for y in y_bins:
                if (x, y) in cross_terms:
                    MI += cross_terms[x, y] * np.log2(cross_terms[x, y]*1.0 / (x_terms[x] * y_terms[y] * 1.0))
       
        # Optionally, compute the entorpy of each individual variable H(X), H(Y) 
        if COMPUTE_ENTROPY:
            H_x = H_y = 0.0;
            # Compute the entropy of the x variable
            for x in x_bins:
                if not np.isclose(x_terms[x], 0.0):
                    H_x += - x_terms[x] * np.log2(x_terms[x])

            # Compute the entropy of the y variable
            for y in y_bins:
                if not np.isclose(y_terms[y], 0.0):
                    H_y += - y_terms[y] * np.log2(y_terms[y])            
        
        return MI, H_x, H_y

    def build_pairwise_mi_matrix(self, override=False):
        ''' Build a matrix containing the pairwise mutual information between each variable in self.analysis_columns. '''
        if self._weight_matrix is not None and override == False:
            return

        self._weight_matrix = np.zeros((len(self.analysis_columns), len(self.analysis_columns)))
        
        for i, x_label in enumerate(self.analysis_columns):
            for j, y_label in enumerate(self.analysis_columns):
                # Only compute for the upper triangle of the matrix
                if i < j:
                    mi, h_x, h_y = self.pairwise_mutual_information(x_label, y_label, COMPUTE_ENTROPY=False)
                    self._weight_matrix[i, j] = mi                

        # Sparsify weight matrix
        self._weight_matrix_sparse = csr_matrix(self._weight_matrix)

    def build_chow_liu_tree(self):
        '''Execute the Chow Liu Tree algorithm, i.e., use the pairwise MI matrix and compute a maximum spanning tree. '''
        self._chow_liu_graph = -1.0 * minimum_spanning_tree(-1.0 * self.weight_matrix)# Mult by -1.0 to get the maximum spanning tree 

    def build_relevence_network(self, threshhold=0.8):
        ''' Build a relevent graph, containing only edges of weight matrix within threshhold * max(weight_matrix). '''
        self._relevence_graph = copy.copy(self.weight_matrix)
        self.relevence_thresh = threshhold

        # Zero out elements that are less then threshhold times the maximum value
        self._relevence_graph[self._relevence_graph < np.max(self._relevence_graph) * threshhold] = 0.0
        self._relevence_graph_sparse = csr_matrix(self._relevence_graph)


    def visualize_graph(self, sparse_matrix, layout='circular'):
        ''' Visualize a sparse E x E matrix as a weighted adjacency matrix for a graph using networkx. '
        Keyword arguments:
        sparse_matrix -- scipy csr_matrix to be interpeted as an E x E weighted adjacency matrix.
        layout -- the graph layout. One of ['circular', 'spring', 'shell', 'kamamda'] (default: 'circular')
        '''
        g = nx.from_scipy_sparse_matrix(sparse_matrix, edge_attribute='weight')

        fig, ax = plt.subplots(1, 1, figsize=(16, 12))
        edges, weights = zip(*nx.get_edge_attributes(g, 'weight').items())

        if layout == 'circular':
            pos = nx.circular_layout(g)
        elif layout == 'spring':
            pos = nx.spring_layout(g)
        elif layout == 'shell':
            pos = nx.shell_layout(g)
        elif layout == 'kamada':
            pos = nx.kamada_kawai_layout(g)
        else:
            pos = nx.circular_layout(g)

        nx.draw_networkx(G=g, 
                         pos=pos,
                         ax=ax,
                         with_labels = True, 
                         node_color = 'k', 
                         edgelist=edges, 
                         edge_color=(weights), 
                         width=10.0, 
                         edge_cmap=plt.cm.Blues,
                         node_size=1000,
                         labels=dict(zip(range(len(self.analysis_labels)), self.analysis_labels)),
                         font_size=20,
                         font_color='r')

        plt.show()

    def visualize_full_graph(self, layout='circular'):
        ''' Visualize the full weighted pairwise MI graph. '''
        self.visualize_graph(self.weight_matrix_sparse, layout)

    def visualize_relevence_graph(self, layout='circular', threshhold=None):
        ''' Visualize the relevence MI graph. '''
        # If visualization threshhold is differnt then the current stored graph, have to overwrite graph
        print(threshhold, self.relevence_thresh)
        if threshhold != self.relevence_thresh:
            self._relevence_graph = None
            self._relevence_graph_sparse = None

        if threshhold is None:
            self.relevence_thresh = 0.8
        else:
            self.relevence_thresh = threshhold
            
        self.visualize_graph(self.relevence_graph_sparse, layout)

    def visualize_chow_liu_graph(self, layout='circular'):
        ''' Visualize the Chow Liu tree. '''
        self.visualize_graph(self.chow_liu_graph, layout)

    def visualize_scatter_plots(self, top_n=None, shape=None, use_analysis_vars=False):
        ''' Visualize scatter plots of discretized data, either for all data pairs or for the top_n MI scores.
        Arguments:
        top_n -- plot only the top_n mutual informatino scores (default: None)
        shape -- shape of the resulting subplot (default: (1, top_n)) 
        '''
        # If top_n is not specified, plot the full N X N matrix of pairwise variable scatter plots
        if shape is not None and (shape[0] * shape[1] != top_n):
            raise ValueError('Specified shape must have the same number of plots as top_n.')

        if use_analysis_vars:
            column_list = self.analysis_columns
        else:
            column_list = self.analysis_labels

        if top_n is None:
            top_n = len(self.analysis_columns)
            fig, ax = plt.subplots(top_n, top_n,  figsize=(top_n*6, top_n*4))
            for i, x_label in enumerate(column_list):
                for j, y_label in enumerate(column_list):
                    # Only visualize the upper left triangle of the plot matrix
                    if i <= j:
                        ax[i, j].scatter(self.df[x_label], self.df[y_label], s = 10)

                        # Add variable labels to the sides of the grid
                        if i == 0:
                            ax[i, j].set_title(y_label)
                        if j == i:
                            ax[i, 0].set_ylabel(x_label)
        else:
            # Create a plot figure of the specified or default size 
            if shape is None:
                fig, ax = plt.subplots(1, top_n, figsize=(6*top_n, 4))
            else:
                fig, ax = plt.subplots(shape[0], shape[1], figsize=(6*shape[1], 4*shape[0]))

            # Sort the entires of the weight matrix in decreasing order
            sort_index = np.argsort(-1.0*self.weight_matrix, axis=None)

            count = 0
            ind = 0
            while count < top_n:
                i = math.floor(sort_index[ind] / len(self.analysis_columns))
                j = sort_index[ind] % len(self.analysis_columns)

                # Ignore entries in the lower triangle or diagonal of the matrix
                if i >= j:
                    ind+=1
                    continue

                x_label = column_list[i]
                y_label = column_list[j]

                if shape is None or shape[0] == 1 or shape[1] == 1:
                    ax[count].scatter(self.df[x_label], self.df[y_label], s = 10)
                    ax[count].set_title(self.analysis_labels[i]+ ' vs ' + self.analysis_labels[j])
                else:
                    ax[math.floor(count/shape[1]), count%shape[1]].scatter(self.df[x_label], self.df[y_label], s = 10)
                    ax[math.floor(count/shape[1]), count%shape[1]].set_title(self.analysis_labels[i]+ ' vs ' + self.analysis_labels[j])
                # Increment plot count and index
                count+=1
                ind+=1

        plt.show()

    def compare_variables(self, x_label, y_label, use_analysis_vars=False):
        if use_analysis_vars:
            x_label = x_label + '_analysis'
            y_label = y_label + '_analysis'

        fig, ax = plt.subplots(1, 1,  figsize=(6, 4))
        ax.scatter(self.df[x_label], self.df[y_label], s = 10)
        ax.set_title(x_label + ' vs ' + y_label)

    @property
    def weight_matrix(self):
        if self._weight_matrix is None:
            self.build_pairwise_mi_matrix()
        return self._weight_matrix

    @property
    def weight_matrix_sparse(self):
        if self._weight_matrix_sparse is None:
            self.build_pairwise_mi_matrix()
        return self._weight_matrix_sparse

    @property
    def chow_liu_graph(self):
        if self._chow_liu_graph is None:
            self.build_chow_liu_tree()
        return self._chow_liu_graph

    @property
    def relevence_graph(self):
        if self._relevence_graph is None:
            self.build_relevence_network(threshhold = self.relevence_thresh)
        return self._relevence_graph

    @property
    def relevence_graph_sparse(self):
        if self._relevence_graph_sparse is None:
            self.build_relevence_network(threshhold = self.relevence_thresh)
        return self._relevence_graph_sparse

    