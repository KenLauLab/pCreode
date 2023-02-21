# p-Creode algorithm maps cell developmental trajectories
# Copyright (C) 2017  Charles A Herring

# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 2
# of the License, or (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

import sys
import matplotlib.pyplot as _plt
import pandas as _pd
import igraph as _igraph
import numpy as np
from sklearn.decomposition import PCA as _PCA
from sklearn.metrics import pairwise_distances as _pairwise_distances
from sklearn import preprocessing as _preprocessing
import os as _os
from igraph import *
from .functions import *
import matplotlib
from IPython.display import display, Image


#################################################
class PCA( object):

    def __init__( self, data):
        """
        Container class for single cell data
        :param data:  DataFrame of cells with X proteins representing expression
        """
        if not ( isinstance( data, _pd.DataFrame)):
            raise TypeError( 'data must be of pandas DataFrame')
        
        self._data          = data.values
        self._protein_list  = data.columns
        self._cell_count    = data.shape[0]
        self._protein_count = data.shape[1]
        
        #print( 'Cell count = {0}, Gene/protein count = {1}'.format( data.shape[0],data.shape[1])) 
        
    def get_pca( self): #gets a PCA of the data the object was initialized with 
        """
        Principal component analysis of data 
        """
        pca = _PCA()
        self.pca = pca.fit_transform( self._data)
        self.pca_explained_var = pca.explained_variance_ratio_ * 100 #output a percent of the variance explained within the object defined here
        return
    
    def pca_plot_explained_var( self, figsize=(6,6), xlim=(0,25)):
        """ 
        Plot the variance explained by different principal components
        :param figsize: size of plot to return
        """
        if self.pca_explained_var is None:
            raise RuntimeError('Please run get_pca() before plotting')
            
        fig = _plt.figure( figsize=figsize)
        ax = fig.add_subplot(111)
        ax.set_xlabel( 'PC#')
        ax.set_ylabel( 'Explained Var')
        ax.set_xlim( xlim)
        ax.plot( range( len( self.pca_explained_var)), self.pca_explained_var, '-o') #
        return
    
    def pca_set_components( self, n_components):
        """
        Set principal component analysis to desired set of components
        :param n_components: Number of components to keep for further analysis
        :return: All data points with selected PCA components
        """
        if self.pca_explained_var is None:
            raise RuntimeError('Please run get_pca() before selecting components')
        return( self.pca[:,:n_components])
 
################################################# 
class Density( object):

    def __init__( self, preprocessed_data, metric='euclidean'):
        """
        Container class for generating density file used to downsample data
        :param preprocessed_data: numpy array of preprocessed data
        :param metic: distance metric used by sklearn
        """
        if not ( isinstance( preprocessed_data, np.ndarray)):
            raise TypeError( 'preprocessed_data must be numpy array')
        if ( ~np.in1d( metric,  ['cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan'])):
            raise TypeError( 'metric must be one of the following cityblock, cosine, euclidean, l1, l2, or manhattan')
        
        self._data         = preprocessed_data
        self._n_pts        = preprocessed_data.shape[0]
        self._n_components = preprocessed_data.shape[1]
        self._metric       = metric
            
    def nearest_neighbor_hist( self, n_rand_pts=5000, n_bins=200, figsize=(8,6), metric='euclidean', mute=False):
        """
        Plots a histogram of distance to nearest neighbor for select number of random points 
        and returns a best guess for the radius used for density calculations
        :param n_rand_pts: Number of random pts to use to generate histogram
        :patam n_bins: Number of bins used to generate histogram
        :param figsize: size of plot to return
        :param mute: boolean operator to suppress print statements
        :return: Histograom of distances to nearest neighbors
        """
        # Save sys.stdout to return print output if muted 
        old_stdout = sys.stdout
        # Mute print statements if True
        if( mute==True):
            sys.stdout = open( os.devnull, 'w')
        
        if ( n_rand_pts>self._n_pts):
            n_rand_pts=self._n_pts
            
        r_inds     = np.random.choice( range( self._n_pts), size=n_rand_pts)
        dists      = _pairwise_distances( self._data[r_inds,:], self._data, metric=self._metric)
        dists_sort = np.sort( dists, axis=1)
        # plotting configurations
        fig = _plt.figure( figsize=figsize)
        ax = fig.add_subplot(111)
        ax.set_xlabel( 'Distance to Nearest Neighbor')
        ax.set_ylabel( 'Number of Datapoints')
        ax.hist( dists_sort[:,1], bins=n_bins)
        # plot line for best guess starting radius in downsampling
        best_guess = np.median( np.sort( dists_sort[:,1])[-20:])
        ax.axvline( best_guess, color='r')
        print( "best guess starting radius = {}".format( best_guess))
        # return to normal treatment of print statements
        sys.stdout = old_stdout
        return( best_guess)
    
    def radius_best_guess( self, n_rand_pts=5000, metric='euclidean'):
        """
        Returns a best guess for the radius based on a select number of random points
        :param n_rand_pts: Number of random pts to use to generate histogram
        :return: float numeric for best guess of radius
        """
        if ( n_rand_pts>self._n_pts):
            n_rand_pts=self._n_pts
            
        r_inds     = np.random.choice( range( self._n_pts), size=n_rand_pts)
        dists      = _pairwise_distances( self._data[r_inds,:], self._data, metric=self._metric)
        dists_sort = np.sort( dists, axis=1)
        # plotting configurations
        best_guess = np.median( np.sort( dists_sort[:,1])[-20:])
        return( best_guess)
    
    def get_density( self, radius, chunk_size=5000, mute=False):
        """
        Calculates the density of each datapoint
        :param radius: Radius around each datapoints used for density calculations
        :param chunk_size: Number of cells to consider during each iteration due to memory restrictions
        :param  mute: boolean operator to suppress print statements
        :return: Calculated densities for all datapoints
        """
        # Save sys.stdout to return print output if muted 
        old_stdout = sys.stdout
        # Mute print statements if True
        if( mute==True):
            sys.stdout = open( os.devnull, 'w')
            
        # Due to memory restrictions density assignments have to be preformed in chunks
        all_chunks = get_chunks( range( self._n_pts), chunk_size)
        # create array to hold all densities
        density = np.empty((self._n_pts), dtype=int)
        # create a nested array of indices for each cell within rad
        neighbors = np.empty((self._n_pts), dtype=object)

        for chunk in all_chunks:
            
            chunk_dist = _pairwise_distances( self._data[chunk,:], self._data, n_jobs=1, metric=self._metric)
            print( "calculating densities for datapoints: {0} -> {1}".format( chunk[0], chunk[-1]))
            
            for chunk_ind, ind in enumerate( chunk):
                neighbors[ind] = np.setdiff1d( np.ravel( np.argwhere( chunk_dist[chunk_ind]<=radius).ravel()), ind)
                density[ind] = len( neighbors[ind])
        print( "****Always check density overlay for radius fit****")
        self.density   = density
        self.neighbors = neighbors
        
        # return to normal treatment of print statements
        sys.stdout = old_stdout
        
        return( density)
    
    def density_hist( self, n_bins=200, figsize=(8,6)):
        """
        Plots a histogram of datapoints' density
        :patam n_bins: Number of bins used to generate histogram
        :param figsize: size of plot to return
        :return: Histograom of densities
        """
        if self.density is None:
            raise RuntimeError('Please run get_density() before plotting')
            
        # plotting configurations
        fig = _plt.figure( figsize=figsize)
        ax = fig.add_subplot(111)
        ax.set_xlabel( 'Density')
        ax.set_ylabel( 'Number of Datapoints')
        ax.hist( self.density, bins=n_bins)
        return
    
#################################################
class Analysis( object):

    def __init__( self, file_path, graph_id, data, density, noise, metric='euclidean'):
        """
        Container class for analyzing pCreode results
        :param file_path: path to directory where graph files are stored
        :param graph_id:  graph ID to plot in given directory
        :param data:      data used to produce pCreode graphs 
        :param density:   data point densities used to create p-Creode graph
        :param metic: distance metric used by sklearn, in this case to average the node values 
        """
        if not ( _os.path.exists( file_path)):
            raise TypeError( 'please supply a valid file path directory')
        if ( ~np.in1d( metric,  ['cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan'])):
            raise TypeError( 'metric must be one of the following cityblock, cosine, euclidean, l1, l2, or manhattan')
         
        self._file_path = file_path
        self._graph_id  = graph_id
        self._data      = data
        self._density   = density
        self._noise     = noise
        
        # cells data are above the noise threshold
        self.good_cells      = data[density>noise]
        self.good_cells_inds = np.arange( len( data))[density>noise]
        
        # node_data_indices are the node indices from the data set used to run pCreode
        self.node_data_indices = np.genfromtxt( file_path + 'ind_{}.csv'.format( self._graph_id), delimiter=',').astype( int)
        self.num_nodes = len( self.node_data_indices)
        # node_graph_indices refers to the node indices within the graph
        self.node_graph_indices = np.arange( self.num_nodes)
        
        # read adjacency matrix for graph
        adj = pd.read_csv( self._file_path + 'adj_{}.txt'.format(self._graph_id), sep=" ", header=None)
        # get distance between nodes in graph
        self.dist  = pairwise_distances( self._data[self.node_data_indices,:], self._data[self.node_data_indices,:], n_jobs=1, metric='l2')
        # calculate weighted adjacency matric
        w_adj = np.multiply( adj.values, self.dist)
        # create graph to plots
        self.graph = _igraph.Graph.Weighted_Adjacency( w_adj.tolist(), mode=ADJ_UNDIRECTED)
        self.graph.vs["label"] = range( 0, self.num_nodes+1)
        
    def plot_save_graph( self, seed, overlay, file_out, upper_range=3, node_label_size=0):
        """
        Plots a p-Creode  graph with given overlay
        :param seed:      random interger to be used to seed graph plot
        :param overlay:   characteristic to overlay on graph, likely from preprocessed data set 
        :param file_out:  name to give saved graph in file_path provided
        :param upper_range: upper range from which to normalize overlay to, this will vary with analyte
        :param node_label_size: size of node labels, when set to zero (default) no label will be plotted
        :return: A plot of selected p-Creode graph with given overlay and saves a png in file_path with file_out name
        """
        
        # normalize densities to use for nodes sizes in graph plot
        norm_dens = preprocessing.MinMaxScaler( feature_range=(8,30))
        dens      = norm_dens.fit_transform( self._density.astype( float).reshape(-1, 1))[self.node_data_indices]
        # normalize overlay to use for node overlays
        norm_ana = preprocessing.MinMaxScaler( feature_range=(0, upper_range))
        norm_ana.fit( overlay[self._density>self._noise].values.astype( np.float).reshape(-1, 1))
        old_ana  = norm_ana.transform( overlay[self._density>self._noise].values.astype( np.float).reshape(-1, 1))
        # bin the data points to each node so that an average of closest surrounding nodes is used for overlay
        bin_dist = pairwise_distances( self.good_cells, self._data[self.node_data_indices])
        bin_assignments = np.argmin( bin_dist, axis=1)
        new_ana = overlay.values[self.node_data_indices]
        for ii in range( self.num_nodes):
            new_ana[ii] = np.mean( old_ana[bin_assignments==ii])
        norm_1 = np.array( new_ana, dtype=float)
        cl_vals_1 = [[]]*self.num_nodes
        # colors to use for overlay
        get_cl = _plt.get_cmap('RdYlBu_r')
        for jj in range( self.num_nodes):
            cl_vals_1[jj] = get_cl( norm_1[jj])

        self.graph.vs["color"] = [cl_vals_1[kk] for kk in range( self.num_nodes)]
        random.seed( seed)
        layout = self.graph.layout_kamada_kawai( maxiter=2000)
        
        plot( self.graph, self._file_path + '{0}.png'.format(file_out), layout=layout, bbox=(1200,1200), 
                    vertex_size=dens, edge_width=2, vertex_label_size=node_label_size)
                    
        display( Image( filename=self._file_path + file_out + '.png', embed=True, unconfined=True, width=600,height=600))
        
        return  
    
    def get_single_trajectory_indices( self, start_id, stop_id):
        """
        Returns the node indices from the data set used to create the graph when supplied with graph indices from the plot_save_graph
        :param indices_of_interest: indices of interest plotted with the plot_save_graph function
        :return: list of indices from data set used to create the graph
        """
            
        path_ids = np.ravel( self.graph.get_shortest_paths( start_id, stop_id)[0])
        return( self.node_data_indices[path_ids])
    
    def plot_analyte_dynamics( self, overlay, root_id):
        """
        Returns bar plots of analyte dynamics over all trajectories when supplied with a root node
        :param overlay: characteristic to overlay on graph, likely from preprocessed data set
        :param root_id: graph node index for a root node (graph index is not the same as the data indexing)
        :return: bar plot of analyte dynamics for each trajectory, starting from a common root node
        """
        # get all end-state nodes, including root if also an end-state node (degree==1)
        all_end_id = self.node_graph_indices[np.transpose( self.graph.degree())==1]
        # get end-states that arn't the root node
        end_ids = all_end_id[all_end_id!=root_id]
        # return all trajectories from root node to all other end-states
        traj = np.ravel( self.graph.get_shortest_paths( root_id, end_ids))
        
        old_ana  = overlay[self._density>self._noise].values.astype( np.float)
        # bin the data points to each node so that an average of closest surrounding nodes is used for overlay
        bin_dist = pairwise_distances( self.good_cells, self._data[self.node_data_indices])
        bin_assignments = np.argmin( bin_dist, axis=1)
        new_ana = overlay.values[self.node_data_indices]
        new_std = np.zeros_like( new_ana)
        for ii in range( self.num_nodes):
            new_ana[ii] = np.mean( old_ana[bin_assignments==ii])
            # get standard error for error bars
            new_std[ii] = np.std(  old_ana[bin_assignments==ii]) / float( np.sqrt( len(old_ana[bin_assignments==ii])))
        # get branch points
        branch_id = self.node_graph_indices[np.transpose( self.graph.degree())>=3]
                        
        num_traj = len( traj)
        # get longest length of trajectories to set x-axis scale
        xlim = max([len(a) for a in traj])

        _plt.figure(1, figsize=(12,num_traj*4.0))
        for ii in range( num_traj):

            _plt.subplot(num_traj,1,ii+1)
            _plt.bar( range( len( new_ana[traj[ii]])), new_ana[traj[ii]], width=1.0, color='green', yerr=new_std[traj[ii]])
            _plt.ylim(0,max(new_ana)+max(new_std))
            _plt.xlim(0,xlim)
            # plot where trajectory ends
            _plt.axvline( x=len( new_ana[traj[ii]]), color='black', linewidth=2.5, linestyle='--')


            # plot branch points if they exist, likely always will
            br_ck = np.in1d( branch_id, traj[ii])
            if( any( br_ck)):
                vl = branch_id[br_ck]
                for kk in vl:
                    vl2 = np.argwhere( traj[ii]==kk)
                    _plt.axvline( x=vl2, color='red', linewidth=2.5, linestyle='--')
            _plt.title('Trajectory {}'.format(ii+1), fontsize=14, loc='right')     
            _plt.xlabel('p-Creode Trajectory', fontsize=12)
            _plt.ylabel('Expression', fontsize=12)
                        
        return
        

    def get_complete_analyte_gene_trajectories( self, overlay_data, root_id, file_out):
        """
        Returns csv files for dynamics of all analytes for each trajectory when supplied with a root node
        :param overlay_data: data set containing all analytes, must be a pandas dataframe
        :param root_id:      graph node index for a root node (graph index is not the same as the data indexing)
        :param file_out:  name to give saved graph in file_path provided
        :return: csv files for analyte dynamics stored in file_path with graph files, as well as a file for cluster IDs 
        """
        if not ( isinstance( overlay_data, _pd.DataFrame)):
            raise TypeError( 'data must be of pandas DataFrame')
   
        # get all end-state nodes, including root if also an end-state node (degree==1)
        all_end_id = self.node_graph_indices[np.transpose( self.graph.degree())==1]
        # get end-states that arn't the root node
        end_ids = all_end_id[all_end_id!=root_id]
        # return all trajectories from root node to all other end-states
        traj = np.ravel( self.graph.get_shortest_paths( root_id, end_ids))

        # lazy work around for when ravel is not needed with non-branching trajectories
        if( len( end_ids)==1):
            traj = [traj]
        
        num_traj = len( end_ids)
        num_ana  = overlay_data.shape[1]

        old_ana  = overlay_data.values[self._density>self._noise]
        # bin the data points to each node so that an average of closest surrounding nodes is used for overlay
        bin_dist = pairwise_distances( self.good_cells, self._data[self.node_data_indices])
        bin_assignments = np.argmin( bin_dist, axis=1)
        new_ana = old_ana
        
        #print( len( bin_assignments), len( self.good_cells_inds))
        np.savetxt( self._file_path + "{}_clust_ids.csv".format( file_out), np.vstack( (self.good_cells_inds, bin_assignments)), delimiter=',')
        
        for hh in range( num_ana):
            for ii in range( self.num_nodes):
                itr_ana = old_ana[bin_assignments==ii,hh]
                # if no cells are binned to that node
                if( itr_ana.size==0):
                    continue
                new_ana[ii,hh] = np.mean( itr_ana)
                
        for cc in range( num_traj):
            
            traj_ana = pd.DataFrame( new_ana[traj[cc]].T, index=overlay_data.columns, columns=traj[cc])
            traj_ana.to_csv( self._file_path + '{0}_traj{1}_analytes.csv'.format( file_out, cc+1))
            
        return
        
        
    def plot_save_qual_graph( self, seed, overlay, file_out):
        """
        Plots a p-Creode  graph with given overlay
        :param seed:      random interger to be used to seed graph plot
        :param overlay:   numpy string of qualitative characteristic to overlay on graph 
        :param file_out:  name to give saved graph in file_path provided
        :return: A plot of selected p-Creode graph with qualitative overlay and saves a png in file_path with file_out name
        """
        if not ( isinstance( overlay, np.ndarray)):
            raise TypeError( 'overlay variable must be numpy array')
        if not ( overlay.dtype.char == 'U'):
            raise TypeError( 'All elements in overlay variable must be in a string dtype')
        
        # get list of colors to be used for labeling
        colors   = np.array( [])
        cl_names = np.array( [])
        for name, hex in matplotlib.colors.cnames.items(): #items instead of iteritems
            colors   = np.append(   colors, hex) 
            cl_names = np.append( cl_names, name)
        
        # normalize densities to use for nodes sizes in graph plot
        norm_dens = preprocessing.MinMaxScaler( feature_range=(8,30))
        dens      = norm_dens.fit_transform( self._density.astype( float).reshape(-1, 1))[self.node_data_indices]
        
        # bin the data points to each node so that an average of closest surrounding nodes is used for overlay
        bin_dist = pairwise_distances( self.good_cells, self._data[self.node_data_indices])
        bin_assignments = np.argmin( bin_dist, axis=1)
        new_ana = overlay[self.node_data_indices]
        for ii in range( self.num_nodes):
            u_over      = np.unique( overlay[np.where( bin_assignments==ii)])
            uniqs = np.unique( overlay[np.where( bin_assignments==ii)], return_counts=True)[1]
            # skip nodes with no cells assigned to it
            if( uniqs.size==0):
                continue
            new_ana[ii] = u_over[np.argmax( uniqs)]

        ids_ana = np.zeros(self.num_nodes, dtype=int)
        zz = 0
        for ii in np.unique( overlay):
            ids_ana[new_ana==ii] = zz
            zz = zz + 1
            
        self.graph.vs["color"] = [colors[kk] for kk in ids_ana]
        
        random.seed( seed)
        layout = self.graph.layout_kamada_kawai( maxiter=2000)
        
        plot( self.graph, self._file_path + '{0}.png'.format(file_out), layout=layout, bbox=(1200,1200), 
                     vertex_size=dens, edge_width=2, vertex_label_size=0)
        display( Image( filename=self._file_path + file_out + '.png', embed=True, unconfined=True, width=600,height=600))
        
        x = np.linspace( 0, 100, len( np.unique( overlay)))
        y = [0]*len( x)
        label = np.unique( overlay)
        cls = cl_names[:len(x)]
        
        fig, ax = _plt.subplots( 1, figsize=(15,2))
        ax.scatter(x, y, s=1000, c=cls, label=label)

        for i, txt in enumerate( label):
            ax.annotate(txt, (x[i]-1.0,y[i]+0.075))
        _plt.axis( 'off')
        _plt.show()        
        
        return  
