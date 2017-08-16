import matplotlib.pyplot as _plt
import pandas as _pd
import numpy as np
from sklearn.decomposition import PCA as _PCA
from sklearn.metrics import pairwise_distances as _pairwise_distances
from sklearn import preprocessing as _preprocessing
import os as _os
from functions import *

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
        
    def get_pca( self):
        """
        Principal component analysis of data 
        """
        pca = _PCA()
        self.pca = pca.fit_transform( self._data)
        self.pca_explained_var = pca.explained_variance_ratio_ * 100
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
        ax.plot( range( len( self.pca_explained_var)), self.pca_explained_var, '-o')
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
            
    def nearest_neighbor_hist( self, n_rand_pts=5000, n_bins=200, figsize=(8,6), metric='euclidean'):
        """
        Plots a histogram of distance to nearest neighbor for
        select number of random points
        :param n_rand_pts: Number of random pts to use to generate histogram
        :patam n_bins: Number of bins used to generate histogram
        :param figsize: size of plot to return
        :return: Histograom of distances to nearest neighbors
        """
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
        # plot line for 3rd STD, can be used a starting radius in downsampling
        best_guess = np.mean( dists_sort[:,1]) + 3*np.std( dists_sort[:,1])
        ax.axvline( best_guess, color='r')
        print( "3rd STD (best guess starting radius) = {}".format( best_guess))
        return
    
    def get_density( self, radius, chunk_size=5000):
        """
        Calculates the density of each datapoint
        :param radius: Radius around each datapoints used for density calculations
        :patam chunk_size: Number of cells to consider during each iteration due to memory restrictions
        :return: Calculated densities for all datapoints
        """
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
    
