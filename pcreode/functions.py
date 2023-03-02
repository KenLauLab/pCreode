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

import numpy as np
import pandas as pd
import igraph as _igraph
import matplotlib.pyplot as _plt
import random
from igraph import *
from sklearn.metrics import pairwise_distances
from sklearn.cluster import KMeans as _KMeans
from sklearn import preprocessing
from sklearn import metrics
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
import os as _os
import sys


#################################################
def Down_Sample( data, density, noise, target, mute=False):
    ''' 
    Function for downsampling data 
    :param data:    numpy ndarray of data set
    :param density: numpy array of calculated densities for each datapoint
    :param noise:   value for noise threshold, densities below value will be removed during downsampling
    :param target:  value for target density
    :param  mute: boolean operator to suppress print statements
    :return down_sampled: array of downsampled dataset 
    :return down_ind:     array of orginal indices for selected datapoints
    '''
    # Save sys.stdout to return print output if muted 
    old_stdout = sys.stdout
    # Mute print statements if True
    if( mute==True):
        sys.stdout = open( os.devnull, 'w')
    
    # set outlier and target densities
    outlier_density = np.percentile( density, noise)
    target_density  = np.percentile( density, target)
    
    # get cell probabilities based on density
    cell_prob = np.array( [0.0]*len( density))

    for ii in range( len( density)):
        # get rid of outliers
        if( density[ii]<outlier_density):
            cell_prob[ii] = 0
        # keep data points within target range
        elif( density[ii]>=outlier_density and density[ii]<=target_density):
            cell_prob[ii] = 1
        # set probability for over represented data points
        elif( density[ii]>target_density):
            cell_prob[ii] = float( target_density)/float( density[ii])

    # create an array of random floats from which to compare cell probabilities
    test_prob = np.random.random( size=len( density))

    # select ind of cells that are kept
    down_ind = np.ravel( np.argwhere(cell_prob>=test_prob))
    
    # select which cells to keep based on density probablities
    down_sampled = data[down_ind,:]

    print( "Number of data points in downsample = {0}".format( len(down_sampled)))
        
    # return to normal treatment of print statements
    sys.stdout = old_stdout
    
    return( down_sampled, down_ind)
    
################################################# 
def get_chunks(l, n):
    ''' 
    Function used to partition data into workable chunks
    :param l: array to be broken into chunks
    :param n: value size of chunks
    :return: array of chunks
    '''
    n = max(1, n)
    return [l[i:i + n] for i in range(0, len(l), n)]
    
#################################################
def find_closest_ind( point, data):
    ''' 
    Function to return the index for the closest point to the given point 
    :param point: array for point to return closest index
    :param data:  numpy ndarray to search for closest point in
    :return: value of index for closest point from data
    '''
    dist = pairwise_distances( point.reshape(1, -1), data, n_jobs=1, metric='l2')
    closest = np.argmin( dist)
    return closest   

#################################################
def get_graph_distance( from_ind, to_ind, graph):
    ''' 
    Function to create a distance matrix from graph space
    :param from_ind: list of graph indices from which to get distance
    :param to_ind:   list of graph indices to which to get distance
    :return: ndarray distance matrix 
    '''
    T = len( to_ind)
    F = len( from_ind)
    d = np.zeros( (  F, T))
    for ii in range( F):
        d[ii,:] = graph.shortest_paths( from_ind[ii], to_ind, weights="weight")[0]
   
    return( d)

#################################################
def find_endstates( data, density, noise, target, potential_clusters=10, cls_thresh=0.0, mute=False):
    ''' 
    Function for retrieving endstates, does not run connect the endstates 
    :param data:    numpy ndarray of data set
    :param density: numpy array of calculated densities for each datapoint
    :param noise:   value for noise threshold, densities below value will be removed during downsampling
    :param target:  value for target density 
    :param potential_clusters: value for upper range of number of clusters to search for, default value is 10
    :param cls_thresh: value for closeness threshold use to separate potential endstates from transitional  cell types default value is 0.0
    :param  mute: boolean operator to suppress print statements
    :return endstates_ind: array of indices for most representative cell of identified endstates 
    :return down_ind:      ndarray with indices of datapoints selected by downsampling 
    :return clust_ids:     ndarray with cluster IDs; highest valued cluster ID is for cells with high closeness, likely transititional cell types
    :return std_cls:       array of containing the closeness values for the downsampled dataset 
    '''
    if not ( isinstance( data, np.ndarray)):
        raise TypeError( 'data variable must be numpy ndarray')
    if not ( isinstance( density, np.ndarray)):
        raise TypeError( 'data variable must be numpy array')
        
    # Save sys.stdout to return print output if muted 
    old_stdout = sys.stdout
    # Mute print statements if True
    if( mute==True):
        sys.stdout = open( os.devnull, 'w')
        
    # get downsampled dataset
    down, down_ind = Down_Sample( data, density, noise, target)

    # array for orginal density (prior to downsampling) of downsampled data points
    down_density = density[down_ind]
    n_down       = len( down)

    # get distance matrix for down sampled dataset
    Dist = np.array( pairwise_distances( down, down, n_jobs=1))

    # set upper and lower thresholds for number of neighbors to connect in density 
    # based nearest neighbor graph (d-kNN) (current fixed values are 2 and 10)
    upper_nn = 10
    lower_nn = 2

    # assign number of neighbors to connect to, to each datapoint 
    sorted_nn = np.linspace( lower_nn, upper_nn, n_down, dtype=int)
    nn = np.zeros( n_down, dtype=int)
    nn[np.argsort( down_density)] = sorted_nn

    # create adjacency matrix to hold neighbor connections for d-kNN
    knn_adj = np.zeros( ( n_down, n_down), dtype=int)
    for zz in range( n_down):
        knn_adj[zz,np.argsort( Dist[zz,:])[1:nn[zz]]] = 1
    # to make symetric add adj with transpose
    knn_adj = np.add( knn_adj, knn_adj.T)
    
    # make sure there is only one component by constructing a MST
    Dist_csr = csr_matrix( np.triu(Dist))
    Tcsr     = minimum_spanning_tree( Dist_csr)
    mst_adj  = pd.DataFrame( Tcsr.todense()).values
    mst_adj  = np.add( mst_adj, mst_adj.T)
    
    # add the two adjacency matrices
    adj = np.add( knn_adj, mst_adj)
    
    # make sure overlaping neighbors arnt double counted
    adj[adj>0] = 1.0

    # normalize the orginal densities of the downsampled data points
    norm = preprocessing.MinMaxScaler()
    dens_norm = np.ravel( norm.fit_transform( down_density.reshape( -1, 1).astype( np.float)))

    # weight edges of d-kNN by inverse of orginal densities
    den_adj = np.zeros( ( n_down, n_down), dtype=float)
    print( "Constructing density kNN")
    # get coordinates of connections from adjacency matrix
    adj_coords = np.nonzero( np.triu( adj))
    for hh, uu in zip( adj_coords[0], adj_coords[1]):
        # take the minimum density of nodes connected by the edge
        # add 0.1 so that no connection is lost (not equal to zero)
        den_adj[hh,uu] = 1.1 - ( min( [dens_norm[hh], dens_norm[uu]]))
    # make symetric 
    den_adj  = np.add( den_adj, den_adj.T)
    # final edge weights are product of density weights and distance matrix
    dist_weighted_adj = np.multiply( Dist, adj)
    dens_weighted_adj = np.multiply( Dist, den_adj)
    # create undirected igraph instance using weighted matrix
    d_knn = _igraph.Graph.Weighted_Adjacency( dist_weighted_adj.tolist(), loops=False, mode=ADJ_UNDIRECTED)

    print( "finding endstates")
    # get closeness of graph and standardize to aid in endstate identification
    cls     = np.array( d_knn.closeness( weights="weight"))
    scaler  = preprocessing.StandardScaler()
    std_cls = scaler.fit_transform( cls.reshape(-1,1)).ravel()

    # using closeness as threshold (default value = 0.0) get potential endstates
    low_cls = down[std_cls<=cls_thresh]
    # array to hold silhouette score for each cluster try
    sil_score = [0]*potential_clusters

    # prefrom K means clustering and score each attempt
    for ss in range( potential_clusters):
        kmeans_model  = _KMeans( n_clusters=ss+2, random_state=10).fit( low_cls)
        label         = kmeans_model.labels_
        sil_score[ss] = metrics.silhouette_score( low_cls, labels=label, metric='l2')

    # find most likely number of clusters from scores above and double to allow for rare cell types
    num_clusters = ( np.argmax( sil_score) + 2) * 2
    clust_model = _KMeans( n_clusters=num_clusters, random_state=10).fit( low_cls)
    label      = clust_model.labels_
    print( "Number of endstates found -> {0}".format( num_clusters))

    endstates = clust_model.cluster_centers_
    endstates_ind = np.zeros( (num_clusters, 1), dtype=int)
    for ii in range( num_clusters):
        endstates_ind[ii] = find_closest_ind( endstates[ii], data)
    
    endstates_ind = endstates_ind.ravel()
    endstates = data[endstates_ind,:]
    
    # get cluster IDs for clustered data
    clust_ids = [0]*n_down
    dd = 0
    for ii in range( n_down):
        if( std_cls[ii]<=cls_thresh):
            clust_ids[ii] = label[dd]
            dd = dd + 1
        else:
            clust_ids[ii] = num_clusters+1
            
    # return to normal treatment of print statements
    sys.stdout = old_stdout
    
    return( endstates_ind, down_ind, clust_ids, std_cls)
    
#################################################
def hierarchical_placement( graph, end_ind):
    ''' 
    Function for selecting pathways through a graph for selected nodes, revealing ancestral relationships 
    :param graph:   an igraph weighted graph
    :param end_ind: indices of selected nodes to be placed
    :return hi_pl: igraph weighted graph of hierarchical placed nodes and selected paths connecting them
    :return names: numpy array of hierarchical placed node indices (in terms of down_ind)
    '''
    n_end = len( end_ind)
    
    # create a graph object for to hold hierarchical placement (hi_pl) graph
    hi_pl = _igraph.Graph()
    # add endstate nodes to hi_pl 
    hi_pl.add_vertices( [str(ii) for ii in end_ind])
    
    # array to hold indices of nodes that are capable of being connected to, initialized with just endstates
    to_ind = end_ind
    # array to hold connected endstates 
    conn_ind = np.empty((0,1), dtype=int)
    # nested array to hold path indices for connected indices
    paths = []
    [paths.append( [ii][0]) for ii in end_ind]
    # run loop until all endstates are connected within a component
    while( np.any( np.in1d( end_ind, conn_ind, invert=True))):
        
        # get graph distance for all non connected endstates to all to_ind
        run_dist = get_graph_distance( end_ind, to_ind, graph)
        run_dist = pd.DataFrame( run_dist, index=[str(jj) for jj in end_ind], columns=[str(ii) for ii in to_ind])
        
        # set dist to used coords to unrealistic min dist
        # simply removing rows caused an accounting headache, we be fixed in future update 
        for used_itr in conn_ind:
            run_dist.loc[str( used_itr),:] = 0
            
        # find the min dist for run and the run indices for it
        run_min    = np.amin( run_dist.values[np.nonzero( run_dist.values)])
        run_coords = np.argwhere( run_dist.values==run_min)[0]
        # get the vertices within the shortest graph path
        run_path = graph.get_shortest_paths( end_ind[run_coords[0]], to_ind[run_coords[1]], weights="weight")[0]           
        unq_path = np.array(run_path)[np.in1d( run_path, to_ind, invert=True)]
        # add indices from path to possible connection indices
        to_ind   = np.append( to_ind, unq_path)
        
        # add unique path vertices to hi_pl
        hi_pl.add_vertices( [str(ii) for ii in unq_path])
        # connect edges in new graph to recreate path 
        for es_itr in range( len( run_path)-1):
            wgt = graph.shortest_paths( run_path[es_itr], run_path[es_itr+1], weights="weight")[0][0]
            hi_pl.add_edge( str(run_path[es_itr]), str(run_path[es_itr+1]), weight=wgt)
        
        for coord_itr in run_coords:
            if( coord_itr<n_end):
                conn_ind = np.append( conn_ind, end_ind[coord_itr])
                paths[coord_itr] = np.unique( np.append( paths[coord_itr], run_path))
     
    # make sure all endstate components are connected to form a complete graph
    names = np.array( hi_pl.vs['name'])
    names = names.astype( np.int)
    comp  = hi_pl.components( mode=WEAK)
    num_comp = len( comp)
    
    while( num_comp>1):
        rest = np.empty((0,1), dtype=int)
        for jj in range( 1, num_comp):
            rest = np.append( rest, names[comp[jj]])
        comp_dist   = get_graph_distance( names[comp[0]], rest, graph)
        comp_min    = np.amin( comp_dist[np.nonzero( comp_dist)])
        comp_coords = np.argwhere( comp_dist==comp_min)[0]
        comp_path = graph.get_shortest_paths( names[comp[0]][comp_coords[0]], rest[comp_coords[1]], weights="weight")[0]
        # add path vertices to hi_pl, if not already in present
        for kk in comp_path:
            if( np.in1d( kk, names)):
                continue
            else:
                hi_pl.add_vertices( str( kk))
        # connect edges in new graph to recreate path 
        for xx in range( len( comp_path)-1):
            wgt = graph.shortest_paths( comp_path[xx], comp_path[xx+1], weights="weight")[0][0]
            hi_pl.add_edge( str( comp_path[xx]),str( comp_path[xx+1]), weight=wgt)
        comp = hi_pl.components( mode=WEAK)
        num_comp = len( comp)
        names = np.array( hi_pl.vs['name'])
        names = names.astype( np.int)
        
    hi_pl.vs["label"] = hi_pl.vs["name"]
    return( hi_pl, names)
    
#################################################
def consensus_alignment( down, hi_pl_ind, data, density, noise):
    ''' 
    Function for aligning selected data points in consensus routes 
    :param down:  numpy ndarray of downsampled data set
    :param hi_pl: igraph weighted graph of hierarchical placed nodes and selected paths connecting them    :param noise:   value for noise threshold, densities below value will be removed during downsampling
    :param data:  numpy ndarray orignal data set
    :param density: numpy array of calculated densities for each datapoint in orginal dataset
    :param noise: value for noise threshold, densities below value will not be considered during alignment
    :return: ndarry of indices of new nodes in terms of location within downsampled dataset
    '''
    # remove noise from data, do not want the opinion of noise in alignment
    no_noise = data[density>np.percentile( density, noise)]
    num_cells, num_cols = no_noise.shape
    # during each iteration data point will be relocated to a data point in the downsampled
    # dataset closest to the new aligned point
    # this array will hold the indices of the closest down sampled point
    node_ind = np.zeros( len( down), dtype=int)

    # arrays to hold new node locations and indices
    new_nodes = np.zeros( ( num_cells, num_cols), dtype=float)
    new_ind   = hi_pl_ind

    # declare how many data points should be assigned each run
    run_size = 1000
    # random array to randomize assignments
    rand_ind = np.random.choice( range( num_cells), num_cells, replace=False)
    # get number of runs needed to assign all points in sets of 1000
    chunks = get_chunks( rand_ind, run_size)

    for ss in range( len( chunks)):
        # current location of points being aligned
        gp = down[hi_pl_ind]

        # get distance between chunk data points and align points
        gd_dist = pairwise_distances( no_noise[chunks[ss]], gp, n_jobs=1, metric='l2')
        
        # array to hold indices for the closest down sampled point
        node_ind = np.zeros( len( chunks[ss]), dtype=int)
        
        for ii in range( len( chunks[ss])):
            # find closest aligned indice for each point in chunk set
            node_ind[ii] = hi_pl_ind[np.argmin( gd_dist[ii])]
        for jj in range( len( hi_pl_ind)):
            # get the center of binned data points
            if( np.in1d( hi_pl_ind[jj], node_ind)):
                new_nodes[jj] = np.median( no_noise[chunks[ss][node_ind==hi_pl_ind[jj]]], axis=0)
                new_ind[jj] = find_closest_ind( new_nodes[jj], down)
            else:
                new_nodes[jj] = gp[jj]
        # reset new position of aligned nodes
        new_ind = np.unique( new_ind)
        hi_pl_ind = new_ind
        
    new_nodes = down[new_ind]

    return( new_ind)    
    
#################################################    
def graph_differences( ind_1, ind_2, dist_1, dist_2, dist, g_1, g_2):
    ''' 
    Function returns the similarity score between two graphs  
    :param ind_1:  array of indices for 1st graph
    :param ind_2:  array of indices for 2nd graph    
    :param dist_1: ndarray graph distance matrix for all nodes in 1st graph
    :param dist_2: ndarray graph distance matrix for all nodes in 2nd graph
    :param dist:   ndarray euclidean distance matrix between nodes in 1st and 2nd graphs
    :param g_1:    igraph weighted graph for 1st graph
    :param g_2:    igraph weighted graph for 2nd graph 
    :return branch_diss: value of pairwise difference in branch counts
    :return dist_diss:   value of pairwise difference in graph distance
    '''
    num_y = len( ind_1)
    branch_diff = 0.0
    dis_y = 0.0
    # variable to hold the number of interactions
    count = 0.0
    for ii in range( num_y-1):
        # find the closest node in other graph 
        min_ind_ii  = np.argmin( dist[ii,:])
        # get euclidean distance between closest nodes(transformation distance)
        min_dist_ii = dist[ii,min_ind_ii]
        for jj in range( ii+1, num_y):
            # find the other closest node in 2nd graph
            min_ind_jj  = np.argmin( dist[jj,:])
            min_dist_jj = dist[jj,min_ind_jj]
            # count number of branch points two nodes in graph one
            deg_list_1 = np.transpose( g_1.degree())[g_1.get_all_shortest_paths( ii,jj)[0]]
            g1_branches = sum( deg_list_1[deg_list_1>2]) - 2 * len( deg_list_1[deg_list_1>2])
            # count number of branch points two nodes in graph two
            deg_list_2 = np.transpose( g_2.degree())[g_2.get_all_shortest_paths( min_ind_ii, min_ind_jj)[0]]
            g2_branches = sum( deg_list_2[deg_list_2>2]) - 2 * len( deg_list_2[deg_list_2>2])
            # get difference in branch points between the two nodes being compared
            branch_diff = branch_diff + abs( g1_branches - g2_branches)
            # get differnce in graph distances plust transfromation distance bewteen two nodes in question 
            dis_y = dis_y + abs( dist_2[min_ind_ii, min_ind_jj] - dist_1[ii,jj] + min_dist_ii + min_dist_jj)
            count = count + 1
            
    del min_dist_ii
    del min_ind_ii
    del min_dist_jj
    del min_ind_jj
    # take average over all pairwise comparisons
    branch_diss = branch_diff/count
    dist_diss =    dis_y/count
    
    return( branch_diss, dist_diss)    
    
#################################################    
def pCreode_Scoring( data, file_path, num_graphs, mute=False):
    ''' 
    Function returns the similarity score between two graphs
    :param data:      numpy ndarray of data set
    :param file_path: path to directory where output files are stored
    :param num_runs:  number of graphs to be scored 
    :param  mute: boolean operator to suppress print statements
    :return branch_diss: value of pairwise difference in branch counts
    :return dist_diss:   value of pairwise difference in graph distance
    '''
    if not ( isinstance( data, np.ndarray)):
        raise TypeError( 'data variable must be numpy ndarray')        
    if not ( _os.path.exists( file_path)):
        raise TypeError( 'please supply a valid file path directory')

    # Save sys.stdout to return print output if muted 
    old_stdout = sys.stdout
    # Mute print statements if True
    if( mute==True):
        sys.stdout = open( os.devnull, 'w')
        
    # array to hold all graph differences between graphs 
    diff = np.zeros( (num_graphs, num_graphs))
    # array to hold all branch differences between graphs
    br   = np.zeros( (num_graphs, num_graphs))
    # loop through to compare all graphs, will result in lower left triangle matrix (br&diff)
    for zz in range( num_graphs-1):
        print( "scoring graph {0}".format( zz+1))
        for kk in range( zz+1, num_graphs):
            # read in arrays for indices for two graphs, in terms of original dataset
            ind_1  = np.genfromtxt( file_path + 'ind_{}.csv'.format( zz), delimiter=',').astype( int)
            ind_2  = np.genfromtxt( file_path + 'ind_{}.csv'.format( kk), delimiter=',').astype( int)
            # read adjacency matrix for graphs 
            adj_1  = pd.read_csv( file_path + 'adj_{}.txt'.format( zz), sep=" ", header=None).values
            adj_2  = pd.read_csv( file_path + 'adj_{}.txt'.format( kk), sep=" ", header=None).values
            # get euclidean distance between nodes in each graph to create weights in graph
            dist_1a = pairwise_distances( data[ind_1,:], data[ind_1,:], n_jobs=1, metric='l2')
            dist_2a = pairwise_distances( data[ind_2,:], data[ind_2,:], n_jobs=1, metric='l2')
            # create weighted adjacency matrix for graphs, 
            # original graphs are not used because edges were weighted by density
            wad_1 = np.multiply( dist_1a, adj_1)
            wad_2 = np.multiply( dist_2a, adj_2)
            # create igraph weighted graph objects
            g1 = _igraph.Graph.Weighted_Adjacency( wad_1.tolist(), mode=ADJ_UNDIRECTED)
            g2 = _igraph.Graph.Weighted_Adjacency( wad_2.tolist(), mode=ADJ_UNDIRECTED)
            # get graph distance between all nodes in each graph
            dist_1 = get_graph_distance( range( len( ind_1)), range( len( ind_1)), g1)
            dist_2 = get_graph_distance( range( len( ind_2)), range( len( ind_2)), g2)
            # get euclidean distance between node in graph 1 and graph 2 (transformation distances)
            dist = pairwise_distances( data[ind_1,:], data[ind_2,:], n_jobs=1, metric='l2')
            # get actual similarity scores, must compare graph1 to graph2 and graph2 to graph1,
            # due to the score not being inherently symmetric
            br_x, diff_x = graph_differences(  ind_1, ind_2, dist_1, dist_2, dist,   g1, g2)
            br_y, diff_y = graph_differences(  ind_2, ind_1, dist_2, dist_1, dist.T, g2, g1)
            # the max is taken to make the score symmetric
            br[zz,kk]   = max( br_x, br_y)
            diff[zz,kk] = max( diff_x, diff_y)
            # sklearn normalizer, need to normlize branch and dist diffs so that they 
            # contribute equally to final overall score 
            norm = preprocessing.MinMaxScaler( feature_range=(0,1))
            
            br_norm   = norm.fit_transform( br)
            diff_norm = norm.fit_transform( diff)
            
            br_diff = ( br_norm+br_norm.T) + ( diff_norm+diff_norm.T)
            
    np.savetxt( file_path + 'branch_diff.csv',      br+br.T,   delimiter=',')
    np.savetxt( file_path + 'graph_dist_diff.csv',    diff+diff.T, delimiter=',')
    np.savetxt( file_path + 'combined_norm_diff.csv', br_diff, delimiter=',')
    
    ranks = np.argsort( np.mean( br_diff, axis=0))
    print( "Most representative graph IDs from first to worst {}".format( ranks))
    
    # return to normal treatment of print statements
    sys.stdout = old_stdout
    
    return( ranks)
    

#################################################

def pCreode( data, density, noise, target, file_path, num_runs=100, potential_clusters=10, cls_thresh=0.0, start_id=0, mute=False):
    ''' 
    Function for running full pCreode algorithm, with the addition of principle component extremes found to be under the closeness threshold added as endstates
    :param data:    numpy ndarray of data set
    :param density: numpy array of calculated densities for each datapoint
    :param noise:   value for noise threshold, densities below value will be removed during downsampling
    :param target:  value for target density
    :param file_path: path to directory where output files will be stored
    :param num_runs:  number of independent runs to perform, default is 100 
    :param potential_clusters: value for upper range of number of clusters to search for, default value is 10
    :param cls_thresh: value for closeness threshold use to separate potential endstates from transitional cell types default value is 0.0
    :param start_id: integer at which to start labeling output graphs, allows for addition of graphs to previously ran lot
    :param  mute: boolean operator to suppress print statements
    :return: will save creode files in given directory
    '''
    if not ( isinstance( data, np.ndarray)):
        raise TypeError( 'data variable must be numpy ndarray')
    if not ( isinstance( density, np.ndarray)):
        raise TypeError( 'data variable must be numpy array')
    if not ( _os.path.exists( file_path)):
        raise TypeError( 'please supply a valid directory')
        
    # Save sys.stdout to return print output if muted 
    old_stdout = sys.stdout
    # Mute print statements if True
    if( mute==True):
        sys.stdout = open( os.devnull, 'w')
        
    print( "Performing {0} independent runs, may take some time".format( num_runs))
  
    for run_itr in range( num_runs):
        
        # get downsampled dataset
        down, down_ind = Down_Sample( data, density, noise, target)

        # array for orginal density (prior to downsampling) of downsampled data points
        down_density = density[down_ind]
        n_down       = len( down)

        # get distance matrix for down sampled dataset
        Dist = np.array( pairwise_distances( down, down, n_jobs=1))

        # set upper and lower thresholds for number of neighbors to connect in density 
        # based nearest neighbor graph (d-kNN) (current fixed values are 2 and 10)
        upper_nn = 10
        lower_nn = 2

        # assign number of neighbors to connect to, to each datapoint 
        sorted_nn = np.linspace( lower_nn, upper_nn, n_down, dtype=int)
        nn = np.zeros( n_down, dtype=int)
        nn[np.argsort( down_density)] = sorted_nn

        # create adjacency matrix to hold neighbor connections for d-kNN
        knn_adj = np.zeros( ( n_down, n_down), dtype=int)
        for zz in range( n_down):
            knn_adj[zz,np.argsort( Dist[zz,:])[1:nn[zz]]] = 1
        # to make symetric add adj with transpose
        knn_adj = np.add( knn_adj, knn_adj.T)
        
        # make sure there is only one component by constructing a MST
        Dist_csr = csr_matrix( np.triu(Dist))
        Tcsr     = minimum_spanning_tree( Dist_csr)
        mst_adj  = pd.DataFrame( Tcsr.todense()).values
        mst_adj  = np.add( mst_adj, mst_adj.T)
        
        # add the two adjacency matrices
        adj = np.add( knn_adj, mst_adj)
        
        # make sure overlaping neighbors arnt double counted
        adj[adj>0] = 1.0

        # normalize the orginal densities of the downsampled data points
        norm = preprocessing.MinMaxScaler()
        dens_norm = np.ravel( norm.fit_transform( down_density.reshape( -1, 1).astype( np.float)))

        # weight edges of d-kNN by inverse of orginal densities
        den_adj = np.zeros( ( n_down, n_down), dtype=float)
        print( "Constructing density kNN")
        # get coordinates of connections from adjacency matrix
        adj_coords = np.nonzero( np.triu( adj))
        for hh, uu in zip( adj_coords[0], adj_coords[1]):
            # take the minimum density of nodes connected by the edge
            # add 0.1 so that no connection is lost (not equal to zero)
            den_adj[hh,uu] = 1.1 - ( min( [dens_norm[hh], dens_norm[uu]]))
        # make symetric 
        den_adj  = np.add( den_adj, den_adj.T)
        # final edge weights are product of density weights and distance matrix
        dist_weighted_adj = np.multiply( Dist, adj)
        dens_weighted_adj = np.multiply( Dist, den_adj)
        # create undirected igraph instance using weighted matrix
        d_knn = _igraph.Graph.Weighted_Adjacency( dist_weighted_adj.tolist(), loops=False, mode=ADJ_UNDIRECTED)

        print( "finding endstates")
        # get closeness of graph and standardize to aid in endstate identification
        cls     = np.array( d_knn.closeness( weights="weight"))
        scaler  = preprocessing.StandardScaler()
        std_cls = scaler.fit_transform( cls.reshape(-1,1)).ravel()

        # using closeness as threshold (default value = 0.0) get potential endstates
        low_cls = down[std_cls<=cls_thresh]
        # array to hold silhouette score for each cluster try
        sil_score = [0]*potential_clusters

        # prefrom K means clustering and score each attempt
        for ss in range( potential_clusters):
            kmeans_model  = _KMeans( n_clusters=ss+2, random_state=10).fit( low_cls)
            label         = kmeans_model.labels_
            sil_score[ss] = metrics.silhouette_score( low_cls, labels=label, metric='l2')

        # find most likely number of clusters from scores above and double to allow for rare cell types
        num_clusters = ( np.argmax( sil_score) + 2) * 2
        clust_model = _KMeans( n_clusters=num_clusters, random_state=10).fit( low_cls)
        label      = clust_model.labels_
        print( "Number of endstates found -> {0}".format( num_clusters))

        endstates = clust_model.cluster_centers_
        endstates_ind = np.zeros( (num_clusters, 1), dtype=int)
        for ii in range( num_clusters):
            endstates_ind[ii] = find_closest_ind( endstates[ii], data)
        
        endstates_ind = endstates_ind.ravel()
        endstates = data[endstates_ind,:]
        num_clusters = len( endstates_ind)
        
        # Endstate data points were picked from full data set, so need to be appended to down and down_ind
        # Create array to hold where end_states are located within the downsampled dataset
        cen_ind = np.zeros( num_clusters, dtype=int)
        ind = n_down
        for es in range( num_clusters):
            # first need to check if they are already in the graph, if not:
            if( ~np.in1d( endstates_ind[es], down_ind)):
                down     = np.vstack( ( down, endstates[es]))
                down_ind = np.append( down_ind, endstates_ind[es])
                cen_ind[es] = ind
                ind = ind + 1
            # if data point is already in down
            else:
                cen_ind[es] = np.argwhere( endstates_ind[es]==down_ind).ravel()[0]
                continue
        
        # re-initialize using density and distance weighted edges         
        dens_knn = _igraph.Graph.Weighted_Adjacency( dens_weighted_adj.tolist(), loops=False, mode=ADJ_UNDIRECTED)

        # add endstate data points to the already constructed dens_knn graph, connecting to 2 closest neighbors
        # future update will so that number of edges is based on density of data point
        knn_num = 2
        # add nodes to graph that will represent endstates
        dens_knn.add_vertices( num_clusters)
        # get distance matrix to be used for finding closeset neighbors in graph
        end_dist = np.array( pairwise_distances( endstates, down[:-num_clusters], n_jobs=1))
        for kk in range( num_clusters):
            edg_wts = np.sort( end_dist[kk,:])[1:knn_num+1]
            edg_ids = np.argsort( end_dist[kk,:])[1:knn_num+1]
            for jj in range( knn_num):
                # no need to connect if connection is already present
                if( edg_wts[jj]<2.0e-06):
                    continue
                # if not present add edge with distance/density weight
                else:
                    dens_knn.add_edge( cen_ind[kk], edg_ids[jj], weight=edg_wts[jj]*(1-dens_norm[edg_ids[jj]]))
        
        print( "hierarchical placing")
        # perform hierarchical placement of endstates (find shortest paths connecting them within d_knn)
        hi_pl, hi_pl_ind = hierarchical_placement( dens_knn, cen_ind)
        print( "consensus aligning")
        # perform consensus alignment of hierarchical placement data points
        aligned_ind = consensus_alignment( down, hi_pl_ind.copy(), data, density, noise)
        # add orginal endstates back into aligned list of indices 
        al_es_ind = np.append( cen_ind, np.unique( aligned_ind[~np.in1d( aligned_ind, cen_ind)]))
        # perform hierarchical placement of of newly aligned data points
        al_hi_pl, al_hi_pl_ind = hierarchical_placement( dens_knn, al_es_ind)
        # rerun hierarchical placement on the aligned placement graph to eliminate superfluous edges
        # by re-feeding it the orginal endstate indices
        creode_graph, creode_ind = hierarchical_placement( al_hi_pl, range( len( cen_ind)))
        creode_graph.simplify( combine_edges="mean")
        print( "saving files for run_num {0}".format( run_itr + 1))
        np.savetxt( file_path + "ind_{0}.csv".format( start_id), down_ind[al_hi_pl_ind[creode_ind]], delimiter=',')
        creode_graph.save( file_path + "adj_{0}.txt".format( start_id), format="adjacency" )
        
        start_id = start_id + 1
        
    # return to normal treatment of print statements
    sys.stdout = old_stdout
        
    return( creode_graph, down_ind[al_hi_pl_ind[creode_ind]])
    
#################################################
    
def pCreode_pca_extremes( data, density, noise, target, file_path, num_runs=100, potential_clusters=10, cls_thresh=0.0, mute=False):
    ''' 
    Function for running full pCreode algorithm, with the addition of principle component extremes found to be under the closeness threshold added as endstates
    :param data:    numpy ndarray of data set
    :param density: numpy array of calculated densities for each datapoint
    :param noise:   value for noise threshold, densities below value will be removed during downsampling
    :param target:  value for target density
    :param file_path: path to directory where output files will be stored
    :param num_runs:  number of independent runs to perform, default is 100 
    :param potential_clusters: value for upper range of number of clusters to search for, default value is 10
    :param cls_thresh: value for closeness threshold use to separate potential endstates from transitional cell types default value is 0.0
    :param  mute: boolean operator to suppress print statements
    :return: will save creode files in given directory
    '''
    if not ( isinstance( data, np.ndarray)):
        raise TypeError( 'data variable must be numpy ndarray')
    if not ( isinstance( density, np.ndarray)):
        raise TypeError( 'data variable must be numpy array')
    if not ( _os.path.exists( file_path)):
        raise TypeError( 'please supply a valid directory')
        
    # Save sys.stdout to return print output if muted 
    old_stdout = sys.stdout
    # Mute print statements if True
    if( mute==True):
        sys.stdout = open( os.devnull, 'w')   
        
    print( "Performing {0} independent runs, may take some time".format( num_runs))
  
    for run_itr in range( num_runs):
        
        # get downsampled dataset
        down, down_ind = Down_Sample( data, density, noise, target)

        # array for orginal density (prior to downsampling) of downsampled data points
        down_density = density[down_ind]
        n_down       = len( down)

        # get distance matrix for down sampled dataset
        Dist = np.array( pairwise_distances( down, down, n_jobs=1))

        # set upper and lower thresholds for number of neighbors to connect in density 
        # based nearest neighbor graph (d-kNN) (current fixed values are 2 and 10)
        upper_nn = 10
        lower_nn = 2

        # assign number of neighbors to connect to, to each datapoint 
        sorted_nn = np.linspace( lower_nn, upper_nn, n_down, dtype=int)
        nn = np.zeros( n_down, dtype=int)
        nn[np.argsort( down_density)] = sorted_nn

        # create adjacency matrix to hold neighbor connections for d-kNN
        knn_adj = np.zeros( ( n_down, n_down), dtype=int)
        for zz in range( n_down):
            knn_adj[zz,np.argsort( Dist[zz,:])[1:nn[zz]]] = 1
        # to make symetric add adj with transpose
        knn_adj = np.add( knn_adj, knn_adj.T)
        
        # make sure there is only one component by constructing a MST
        Dist_csr = csr_matrix( np.triu(Dist))
        Tcsr     = minimum_spanning_tree( Dist_csr)
        mst_adj  = pd.DataFrame( Tcsr.todense()).values
        mst_adj  = np.add( mst_adj, mst_adj.T)
        
        # add the two adjacency matrices
        adj = np.add( knn_adj, mst_adj)
        
        # make sure overlaping neighbors arnt double counted
        adj[adj>0] = 1.0

        # normalize the orginal densities of the downsampled data points
        norm = preprocessing.MinMaxScaler()
        dens_norm = np.ravel( norm.fit_transform( down_density.reshape( -1, 1).astype( np.float)))

        # weight edges of d-kNN by inverse of orginal densities
        den_adj = np.zeros( ( n_down, n_down), dtype=float)
        print( "Constructing density kNN")
        # get coordinates of connections from adjacency matrix
        adj_coords = np.nonzero( np.triu( adj))
        for hh, uu in zip( adj_coords[0], adj_coords[1]):
            # take the minimum density of nodes connected by the edge
            # add 0.1 so that no connection is lost (not equal to zero)
            den_adj[hh,uu] = 1.1 - ( min( [dens_norm[hh], dens_norm[uu]]))
        # make symetric 
        den_adj  = np.add( den_adj, den_adj.T)
        # final edge weights are product of density weights and distance matrix
        dist_weighted_adj = np.multiply( Dist, adj)
        dens_weighted_adj = np.multiply( Dist, den_adj)
        # create undirected igraph instance using weighted matrix
        d_knn = _igraph.Graph.Weighted_Adjacency( dist_weighted_adj.tolist(), loops=False, mode=ADJ_UNDIRECTED)

        print( "finding endstates")
        # get closeness of graph and standardize to aid in endstate identification
        cls     = np.array( d_knn.closeness( weights="weight"))
        scaler  = preprocessing.StandardScaler()
        std_cls = scaler.fit_transform( cls.reshape(-1,1)).ravel()

        # using closeness as threshold (default value = 0.0) get potential endstates
        low_cls = down[std_cls<=cls_thresh]
        # array to hold silhouette score for each cluster try
        sil_score = [0]*potential_clusters

        # prefrom K means clustering and score each attempt
        for ss in range( potential_clusters):
            kmeans_model  = _KMeans( n_clusters=ss+2, random_state=10).fit( low_cls)
            label         = kmeans_model.labels_
            sil_score[ss] = metrics.silhouette_score( low_cls, labels=label, metric='l2')

        # find most likely number of clusters from scores above and double to allow for rare cell types
        num_clusters = ( np.argmax( sil_score) + 2) * 2
        clust_model = _KMeans( n_clusters=num_clusters, random_state=10).fit( low_cls)
        label      = clust_model.labels_
        print( "Number of endstates found -> {0}".format( num_clusters))

        endstates = clust_model.cluster_centers_
        endstates_ind = np.zeros( (num_clusters, 1), dtype=int)
        for ii in range( num_clusters):
            endstates_ind[ii] = find_closest_ind( endstates[ii], data)
        
        extrs = np.argsort( down, axis=0)[0]
        extrs = np.append( extrs, np.argsort( down, axis=0)[-1])
        
        endstates_ind = endstates_ind.ravel()
        endstates = data[endstates_ind,:]
        num_clusters = len( endstates_ind)
        
        # Endstate data points were picked from full data set, so need to be appended to down and down_ind
        # Create array to hold where end_states are located within the downsampled dataset
        cen_ind = np.zeros( num_clusters, dtype=int)
        ind = n_down
        for es in range( num_clusters):
            # first need to check if they are already in the graph, if not:
            if( ~np.in1d( endstates_ind[es], down_ind)):
                down     = np.vstack( ( down, endstates[es]))
                down_ind = np.append( down_ind, endstates_ind[es])
                cen_ind[es] = ind
                ind = ind + 1
            # if data point is already in down
            else:
                cen_ind[es] = np.argwhere( endstates_ind[es]==down_ind).ravel()[0]
                continue
        
        # re-initialize using density and distance weighted edges         
        dens_knn = _igraph.Graph.Weighted_Adjacency( dens_weighted_adj.tolist(), loops=False, mode=ADJ_UNDIRECTED)

        # add endstate data points to the already constructed dens_knn graph, connecting to 2 closest neighbors
        # future update will so that number of edges is based on density of data point
        knn_num = 2
        # add nodes to graph that will represent endstates
        dens_knn.add_vertices( num_clusters)
        # get distance matrix to be used for finding closeset neighbors in graph
        end_dist = np.array( pairwise_distances( endstates, down[:-num_clusters], n_jobs=1))
        for kk in range( num_clusters):
            edg_wts = np.sort( end_dist[kk,:])[1:knn_num+1]
            edg_ids = np.argsort( end_dist[kk,:])[1:knn_num+1]
            for jj in range( knn_num):
                # no need to connect if connection is already present
                if( edg_wts[jj]<2.0e-06):
                    continue
                # if not present add edge with distance/density weight
                else:
                    dens_knn.add_edge( cen_ind[kk], edg_ids[jj], weight=edg_wts[jj]*(1-dens_norm[edg_ids[jj]]))
        
        cen_ind = np.append( cen_ind, extrs[std_cls[extrs]>cls_thresh])
        
        print( "hierarchical placing")
        # perform hierarchical placement of endstates (find shortest paths connecting them within d_knn)
        hi_pl, hi_pl_ind = hierarchical_placement( dens_knn, cen_ind)
        print( "consensus aligning")
        # perform consensus alignment of hierarchical placement data points
        aligned_ind = consensus_alignment( down, hi_pl_ind.copy(), data, density, noise)
        # add orginal endstates back into aligned list of indices 
        al_es_ind = np.append( cen_ind, np.unique( aligned_ind[~np.in1d( aligned_ind, cen_ind)]))
        # perform hierarchical placement of of newly aligned data points
        al_hi_pl, al_hi_pl_ind = hierarchical_placement( dens_knn, al_es_ind)
        # rerun hierarchical placement on the aligned placement graph to eliminate superfluous edges
        # by re-feeding it the orginal endstate indices
        creode_graph, creode_ind = hierarchical_placement( al_hi_pl, range( len( cen_ind)))
        creode_graph.simplify( combine_edges="mean")
        print( "saving files for run_num {0}".format( run_itr + 1))
        np.savetxt( file_path + "ind_{0}.csv".format( run_itr), down_ind[al_hi_pl_ind[creode_ind]], delimiter=',')
        creode_graph.save( file_path + "adj_{0}.txt".format( run_itr), format="adjacency" )
    
    # return to normal treatment of print statements
    sys.stdout = old_stdout
    
    return( creode_graph, down_ind[al_hi_pl_ind[creode_ind]])
    
#################################################

def return_weighted_adj( data, file_path, graph_id):
    ''' 
    Function returns the similarity score between two graphs
    :param data:      numpy ndarray of data set
    :param graph_id:  graph ID to plot in given directory
    :param file_path: path to directory where output files are stored
    :return w_adj:    a numpy ndarray representing weighted adjacency matrix
    '''
    if not ( isinstance( data, np.ndarray)):
        raise TypeError( 'data variable must be numpy ndarray')        
    if not ( _os.path.exists( file_path)):
        raise TypeError( 'please supply a valid file path directory')

    # read in arrays for indices for two graphs, in terms of original dataset
    ind_1  = np.genfromtxt( file_path + 'ind_{}.csv'.format( graph_id), delimiter=',').astype( int)
    # read adjacency matrix for graphs 
    adj_1  = pd.read_csv( file_path + 'adj_{}.txt'.format( graph_id), sep=" ", header=None).values
    # get euclidean distance between nodes in each graph to create weights in graph
    dist_1a = pairwise_distances( data[ind_1,:], data[ind_1,:], n_jobs=1, metric='l2')
    # create weighted adjacency matrix for graphs, 
    # original graphs are not used because edges were weighted by density
    wad_1 = np.multiply( dist_1a, adj_1)
    
    return( wad_1)
    
#################################################

def pCreode_rare_cells( data, density, noise, target, file_path, rare_clusters, num_runs=100, potential_clusters=10, cls_thresh=0.0, start_id=0, mute=False):
    ''' 
    Function for running full pCreode algorithm, but allows user to add rare cell clusters using the indices of the cells.
    :param density: numpy array of calculated densities for each datapoint
    :param noise:   value for noise threshold, densities below value will be removed during downsampling
    :param target:  value for target density
    :param file_path: path to directory where output files will be stored
    :param rare_clusters: nested array consisting of indices for clusters to be added, format of [[clust1 indices],[clust2 indices],[clust3 indices],...]
    :param num_runs:  number of independent runs to perform, default is 100 
    :param potential_clusters: value for upper range of number of clusters to search for, default value is 10
    :param cls_thresh: value for closeness threshold use to separate potential endstates from transitional cell types default value is 0.0
    :param start_id: integer at which to start labeling output graphs, allows for addition of graphs to previously ran lot
    :param  mute: boolean operator to suppress print statements
    :return: will save creode files in given directory
    '''
    if not ( isinstance( data, np.ndarray)):
        raise TypeError( 'data variable must be numpy ndarray')
    if not ( isinstance( density, np.ndarray)):
        raise TypeError( 'data variable must be numpy array')
    if not ( _os.path.exists( file_path)):
        raise TypeError( 'please supply a valid directory')
        
    # Save sys.stdout to return print output if muted 
    old_stdout = sys.stdout
    # Mute print statements if True
    if( mute==True):
        sys.stdout = open( os.devnull, 'w')    
        
    print( "Performing {0} independent runs, may take some time".format( num_runs))
  
    for run_itr in range( num_runs):
        
        # get downsampled dataset
        down, down_ind = Down_Sample( data, density, noise, target)
        
        # add manually selected clusters to downsampled data, if not already
        for clust_itr in range( len( rare_clusters)):
            for cell_itr in range( len( rare_clusters[clust_itr])):
                cell_ind = rare_clusters[clust_itr][cell_itr]
                if( ~np.in1d( cell_ind, down_ind)):
                    down     = np.vstack( ( down, data[cell_ind]))
                    down_ind = np.append( down_ind, cell_ind)
        
        # array for orginal density (prior to downsampling) of downsampled data points
        down_density = density[down_ind]
        n_down       = len( down)

        # get distance matrix for down sampled dataset
        Dist = np.array( pairwise_distances( down, down, n_jobs=1))

        # set upper and lower thresholds for number of neighbors to connect in density 
        # based nearest neighbor graph (d-kNN) (current fixed values are 2 and 10)
        upper_nn = 10
        lower_nn = 2

        # assign number of neighbors to connect to, to each datapoint 
        sorted_nn = np.linspace( lower_nn, upper_nn, n_down, dtype=int)
        nn = np.zeros( n_down, dtype=int)
        nn[np.argsort( down_density)] = sorted_nn

        # create adjacency matrix to hold neighbor connections for d-kNN
        knn_adj = np.zeros( ( n_down, n_down), dtype=int)
        for zz in range( n_down):
            knn_adj[zz,np.argsort( Dist[zz,:])[1:nn[zz]]] = 1
        # to make symetric add adj with transpose
        knn_adj = np.add( knn_adj, knn_adj.T)
        
        # make sure there is only one component by constructing a MST
        Dist_csr = csr_matrix( np.triu(Dist))
        Tcsr     = minimum_spanning_tree( Dist_csr)
        mst_adj  = pd.DataFrame( Tcsr.todense()).values
        mst_adj  = np.add( mst_adj, mst_adj.T)
        
        # add the two adjacency matrices
        adj = np.add( knn_adj, mst_adj)
        
        # make sure overlaping neighbors arnt double counted
        adj[adj>0] = 1.0

        # normalize the orginal densities of the downsampled data points
        norm = preprocessing.MinMaxScaler()
        dens_norm = np.ravel( norm.fit_transform( down_density.reshape( -1, 1).astype( np.float)))

        # weight edges of d-kNN by inverse of orginal densities
        den_adj = np.zeros( ( n_down, n_down), dtype=float)
        print( "Constructing density kNN")
        # get coordinates of connections from adjacency matrix
        adj_coords = np.nonzero( np.triu( adj))
        for hh, uu in zip( adj_coords[0], adj_coords[1]):
            # take the minimum density of nodes connected by the edge
            # add 0.1 so that no connection is lost (not equal to zero)
            den_adj[hh,uu] = 1.1 - ( min( [dens_norm[hh], dens_norm[uu]]))
        # make symetric 
        den_adj  = np.add( den_adj, den_adj.T)
        # final edge weights are product of density weights and distance matrix
        dist_weighted_adj = np.multiply( Dist, adj)
        dens_weighted_adj = np.multiply( Dist, den_adj)
        # create undirected igraph instance using weighted matrix
        d_knn = _igraph.Graph.Weighted_Adjacency( dist_weighted_adj.tolist(), loops=False, mode=ADJ_UNDIRECTED)

        print( "finding endstates")
        # get closeness of graph and standardize to aid in endstate identification
        cls     = np.array( d_knn.closeness( weights="weight"))
        scaler  = preprocessing.StandardScaler()
        std_cls = scaler.fit_transform( cls.reshape(-1,1)).ravel()

        # using closeness as threshold (default value = 0.0) get potential endstates
        low_cls = down[std_cls<=cls_thresh]
        # array to hold silhouette score for each cluster try
        sil_score = [0]*potential_clusters

        # prefrom K means clustering and score each attempt
        for ss in range( potential_clusters):
            kmeans_model  = _KMeans( n_clusters=ss+2, random_state=10).fit( low_cls)
            label         = kmeans_model.labels_
            sil_score[ss] = metrics.silhouette_score( low_cls, labels=label, metric='l2')

        # find most likely number of clusters from scores above and double to allow for rare cell types
        num_clusters = ( np.argmax( sil_score) + 2) * 2
        clust_model = _KMeans( n_clusters=num_clusters, random_state=10).fit( low_cls)
        label      = clust_model.labels_
        print( "Number of endstates found -> {0}".format( num_clusters))

        endstates = clust_model.cluster_centers_
        
        # add manually selected clusters to endstates
        for clust_itr in range( len( rare_clusters)):
            rare_clusters_center = np.median( data[rare_clusters[clust_itr]], axis=0)
            endstates = np.vstack( ( endstates, rare_clusters_center))

        # update number of clusters
        num_clusters = num_clusters + len( rare_clusters)
        
        endstates_ind = np.zeros( (num_clusters, 1), dtype=int)
        for ii in range( num_clusters):
            endstates_ind[ii] = find_closest_ind( endstates[ii], data)
        
        endstates_ind = np.unique( endstates_ind.ravel())
        endstates = data[endstates_ind,:]
        num_clusters = len( endstates_ind)
        
        # Endstate data points were picked from full data set, so need to be appended to down and down_ind
        # Create array to hold where end_states are located within the downsampled dataset
        cen_ind = np.zeros( num_clusters, dtype=int)
        ind = n_down
        for es in range( num_clusters):
            # first need to check if they are already in the graph, if not:
            if( ~np.in1d( endstates_ind[es], down_ind)):
                down     = np.vstack( ( down, endstates[es]))
                down_ind = np.append( down_ind, endstates_ind[es])
                cen_ind[es] = ind
                ind = ind + 1
            # if data point is already in down
            else:
                cen_ind[es] = np.argwhere( endstates_ind[es]==down_ind).ravel()[0]
                continue
        
        # re-initialize using density and distance weighted edges         
        dens_knn = _igraph.Graph.Weighted_Adjacency( dens_weighted_adj.tolist(), loops=False, mode=ADJ_UNDIRECTED)

        # add endstate data points to the already constructed dens_knn graph, connecting to 2 closest neighbors
        # future update will so that number of edges is based on density of data point
        knn_num = 2
        # add nodes to graph that will represent endstates
        dens_knn.add_vertices( num_clusters)
        # get distance matrix to be used for finding closeset neighbors in graph
        end_dist = np.array( pairwise_distances( endstates, down[:-num_clusters], n_jobs=1))
        for kk in range( num_clusters):
            edg_wts = np.sort( end_dist[kk,:])[1:knn_num+1]
            edg_ids = np.argsort( end_dist[kk,:])[1:knn_num+1]
            for jj in range( knn_num):
                # no need to connect if connection is already present
                if( edg_wts[jj]<2.0e-06):
                    continue
                # if not present add edge with distance/density weight
                else:
                    dens_knn.add_edge( cen_ind[kk], edg_ids[jj], weight=edg_wts[jj]*(1-dens_norm[edg_ids[jj]]))
        
        print( "hierarchical placing")
        # perform hierarchical placement of endstates (find shortest paths connecting them within d_knn)
        hi_pl, hi_pl_ind = hierarchical_placement( dens_knn, cen_ind)
        print( "consensus aligning")
        # perform consensus alignment of hierarchical placement data points
        aligned_ind = consensus_alignment( down, hi_pl_ind.copy(), data, density, noise)
        # add orginal endstates back into aligned list of indices 
        al_es_ind = np.append( cen_ind, np.unique( aligned_ind[~np.in1d( aligned_ind, cen_ind)]))
        # perform hierarchical placement of of newly aligned data points
        al_hi_pl, al_hi_pl_ind = hierarchical_placement( dens_knn, al_es_ind)
        # rerun hierarchical placement on the aligned placement graph to eliminate superfluous edges
        # by re-feeding it the orginal endstate indices
        creode_graph, creode_ind = hierarchical_placement( al_hi_pl, range( len( cen_ind)))
        creode_graph.simplify( combine_edges="mean")
        print( "saving files for run_num {0}".format( run_itr + 1))
        np.savetxt( file_path + "ind_{0}.csv".format( start_id), down_ind[al_hi_pl_ind[creode_ind]], delimiter=',')
        creode_graph.save( file_path + "adj_{0}.txt".format( start_id), format="adjacency" )
        
        start_id = start_id + 1
    
    # return to normal treatment of print statements
    sys.stdout = old_stdout
    
    return( creode_graph, down_ind[al_hi_pl_ind[creode_ind]])
    

#################################################

def pCreode_supervised( data, density, noise, target, file_path, man_clust, num_runs=100, start_id=0, mute=False):
    ''' 
    Function for running full pCreode algorithm, without endstate identification. Relies on user supplied endstates. 
    :param data:    numpy ndarray of data set
    :param density: numpy array of calculated densities for each datapoint
    :param noise:   value for noise threshold, densities below value will be removed during downsampling
    :param target:  value for target density
    :param file_path: path to directory where output files will be stored
    :param man_clust: nest array consisting of indices for clusters to be added, format of [[clust1 indices],[clust2 indices],[clust3 indices],...]
    :param num_runs:  number of independent runs to perform, default is 100 
    :param start_id: integer at which to start labeling output graphs, allows for addition of graphs to previously ran lot
    :param  mute: boolean operator to suppress print statements
    :return: will save creode files in given directory
    '''
    if not ( isinstance( data, np.ndarray)):
        raise TypeError( 'data variable must be numpy ndarray')
    if not ( isinstance( density, np.ndarray)):
        raise TypeError( 'data variable must be numpy array')
    if not ( _os.path.exists( file_path)):
        raise TypeError( 'please supply a valid directory')
    
    # Save sys.stdout to return print output if muted 
    old_stdout = sys.stdout
    # Mute print statements if True
    if( mute==True):
        sys.stdout = open( os.devnull, 'w')
    
    print( "Performing {0} independent runs, may take some time".format( num_runs))
  
    for run_itr in range( num_runs):
        
        # get downsampled dataset
        down, down_ind = Down_Sample( data, density, noise, target)
        
        # add manually selected clusters to downsampled data, if not already
        for clust_itr in range( len( man_clust)):
            for cell_itr in range( len( man_clust[clust_itr])):
                cell_ind = man_clust[clust_itr][cell_itr]
                if( ~np.in1d( cell_ind, down_ind)):
                    down     = np.vstack( ( down, data[cell_ind]))
                    down_ind = np.append( down_ind, cell_ind)
        
        # array for orginal density (prior to downsampling) of downsampled data points
        down_density = density[down_ind]
        n_down       = len( down)

        # get distance matrix for down sampled dataset
        Dist = np.array( pairwise_distances( down, down, n_jobs=1))

        # set upper and lower thresholds for number of neighbors to connect in density 
        # based nearest neighbor graph (d-kNN) (current fixed values are 2 and 10)
        upper_nn = 10
        lower_nn = 2

        # assign number of neighbors to connect to, to each datapoint 
        sorted_nn = np.linspace( lower_nn, upper_nn, n_down, dtype=int)
        nn = np.zeros( n_down, dtype=int)
        nn[np.argsort( down_density)] = sorted_nn

        # create adjacency matrix to hold neighbor connections for d-kNN
        knn_adj = np.zeros( ( n_down, n_down), dtype=int)
        for zz in range( n_down):
            knn_adj[zz,np.argsort( Dist[zz,:])[1:nn[zz]]] = 1
        # to make symetric add adj with transpose
        knn_adj = np.add( knn_adj, knn_adj.T)
        
        # make sure there is only one component by constructing a MST
        Dist_csr = csr_matrix( np.triu(Dist))
        Tcsr     = minimum_spanning_tree( Dist_csr)
        mst_adj  = pd.DataFrame( Tcsr.todense()).values
        mst_adj  = np.add( mst_adj, mst_adj.T)
        
        # add the two adjacency matrices
        adj = np.add( knn_adj, mst_adj)
        
        # make sure overlaping neighbors arnt double counted
        adj[adj>0] = 1.0

        # normalize the orginal densities of the downsampled data points
        norm = preprocessing.MinMaxScaler()
        dens_norm = np.ravel( norm.fit_transform( down_density.reshape( -1, 1).astype( np.float)))

        # weight edges of d-kNN by inverse of orginal densities
        den_adj = np.zeros( ( n_down, n_down), dtype=float)
        print( "Constructing density kNN")
        # get coordinates of connections from adjacency matrix
        adj_coords = np.nonzero( np.triu( adj))
        for hh, uu in zip( adj_coords[0], adj_coords[1]):
            # take the minimum density of nodes connected by the edge
            # add 0.1 so that no connection is lost (not equal to zero)
            den_adj[hh,uu] = 1.1 - ( min( [dens_norm[hh], dens_norm[uu]]))
        # make symetric 
        den_adj  = np.add( den_adj, den_adj.T)
        # final edge weights are product of density weights and distance matrix
        dist_weighted_adj = np.multiply( Dist, adj)
        dens_weighted_adj = np.multiply( Dist, den_adj)
        # create undirected igraph instance using weighted matrix
        d_knn = _igraph.Graph.Weighted_Adjacency( dist_weighted_adj.tolist(), loops=False, mode=ADJ_UNDIRECTED)


        # add manually selected clusters to endstates
        num_clusters = len( man_clust)
        endstates = np.zeros( ( num_clusters, data.shape[1]))
        for clust_itr in range( num_clusters):
            man_clust_center = np.median( data[man_clust[clust_itr]], axis=0)
            endstates[clust_itr] = man_clust_center
        
        endstates_ind = np.zeros( (num_clusters, 1), dtype=int)
        for ii in range( num_clusters):
            endstates_ind[ii] = find_closest_ind( endstates[ii], data)
        
        endstates_ind = np.unique( endstates_ind.ravel())
        endstates = data[endstates_ind,:]
        num_clusters = len( endstates_ind)
        
        # Endstate data points were picked from full data set, so need to be appended to down and down_ind
        # Create array to hold where end_states are located within the downsampled dataset
        cen_ind = np.zeros( num_clusters, dtype=int)
        ind = n_down
        for es in range( num_clusters):
            # first need to check if they are already in the graph, if not:
            if( ~np.in1d( endstates_ind[es], down_ind)):
                down     = np.vstack( ( down, endstates[es]))
                down_ind = np.append( down_ind, endstates_ind[es])
                cen_ind[es] = ind
                ind = ind + 1
            # if data point is already in down
            else:
                cen_ind[es] = np.argwhere( endstates_ind[es]==down_ind).ravel()[0]
                continue
        
        # re-initialize using density and distance weighted edges         
        dens_knn = _igraph.Graph.Weighted_Adjacency( dens_weighted_adj.tolist(), loops=False, mode=ADJ_UNDIRECTED)

        # add endstate data points to the already constructed dens_knn graph, connecting to 2 closest neighbors
        # future update will so that number of edges is based on density of data point
        knn_num = 2
        # add nodes to graph that will represent endstates
        dens_knn.add_vertices( num_clusters)
        # get distance matrix to be used for finding closeset neighbors in graph
        end_dist = np.array( pairwise_distances( endstates, down[:-num_clusters], n_jobs=1))
        for kk in range( num_clusters):
            edg_wts = np.sort( end_dist[kk,:])[1:knn_num+1]
            edg_ids = np.argsort( end_dist[kk,:])[1:knn_num+1]
            for jj in range( knn_num):
                # no need to connect if connection is already present
                if( edg_wts[jj]<2.0e-06):
                    continue
                # if not present add edge with distance/density weight
                else:
                    dens_knn.add_edge( cen_ind[kk], edg_ids[jj], weight=edg_wts[jj]*(1-dens_norm[edg_ids[jj]]))
        
        print( "hierarchical placing")
        # perform hierarchical placement of endstates (find shortest paths connecting them within d_knn)
        hi_pl, hi_pl_ind = hierarchical_placement( dens_knn, cen_ind)
        print( "consensus aligning")
        # perform consensus alignment of hierarchical placement data points
        aligned_ind = consensus_alignment( down, hi_pl_ind.copy(), data, density, noise)
        # add orginal endstates back into aligned list of indices 
        al_es_ind = np.append( cen_ind, np.unique( aligned_ind[~np.in1d( aligned_ind, cen_ind)]))
        # perform hierarchical placement of of newly aligned data points
        al_hi_pl, al_hi_pl_ind = hierarchical_placement( dens_knn, al_es_ind)
        # rerun hierarchical placement on the aligned placement graph to eliminate superfluous edges
        # by re-feeding it the orginal endstate indices
        creode_graph, creode_ind = hierarchical_placement( al_hi_pl, range( len( cen_ind)))
        creode_graph.simplify( combine_edges="mean")
        print( "saving files for run_num {0}".format( run_itr + 1))
        np.savetxt( file_path + "ind_{0}.csv".format( start_id), down_ind[al_hi_pl_ind[creode_ind]], delimiter=',')
        creode_graph.save( file_path + "adj_{0}.txt".format( start_id), format="adjacency" )
        
        start_id = start_id + 1
    
    # return to normal treatment of print statements
    sys.stdout = old_stdout
    
    return( creode_graph, down_ind[al_hi_pl_ind[creode_ind]])


#################################################

def get_thresholds( data):
    ''' 
    Function for unsupervised selection of density thresholds for downsampling. 
    :param    data: numpy ndarray of data set
    :return  noise: noise float threshold 
    :return target: target float threshold
    '''
    num_cells = len( data)

    if( num_cells<1000):
        target = 80.0
        noise  = 1.0
    else:
        target = 50.0
        noise = 5.0

    return( noise, target)

#################################################

def pCreode_sparse( data, density, noise, target, file_path, num_runs=100, potential_clusters=10, cls_thresh=1.0, start_id=0, mute=False):
    ''' 
    Function for running full pCreode algorithm, with the addition of principle component extremes found to be under the closeness threshold added as endstates
    :param data:    numpy ndarray of data set
    :param density: numpy array of calculated densities for each datapoint
    :param noise:   value for noise threshold, densities below value will be removed during downsampling
    :param target:  value for target density
    :param file_path: path to directory where output files will be stored
    :param num_runs:  number of independent runs to perform, default is 100 
    :param potential_clusters: value for upper range of number of clusters to search for, default value is 10
    :param cls_thresh: value for closeness threshold use to separate potential endstates from transitional cell types default value is 1.0
    :param start_id: integer at which to start labeling output graphs, allows for addition of graphs to previously ran lot
    :param  mute: boolean operator to suppress print statements
    :return: will save creode files in given directory
    '''
    if not ( isinstance( data, np.ndarray)):
        raise TypeError( 'data variable must be numpy ndarray')
    if not ( isinstance( density, np.ndarray)):
        raise TypeError( 'data variable must be numpy array')
    if not ( _os.path.exists( file_path)):
        raise TypeError( 'please supply a valid directory')
        
    # Save sys.stdout to return print output if muted 
    old_stdout = sys.stdout
    # Mute print statements if True
    if( mute==True):
        sys.stdout = open( os.devnull, 'w')
        
    print( "Performing {0} independent runs, may take some time".format( num_runs))
  
    for run_itr in range( num_runs):
        
        # get downsampled dataset
        down, down_ind = Down_Sample( data, density, noise, target)

        # array for orginal density (prior to downsampling) of downsampled data points
        down_density = density[down_ind]
        n_down       = len( down)

        # get distance matrix for down sampled dataset
        Dist = np.array( pairwise_distances( down, down, n_jobs=1))

        # set upper and lower thresholds for number of neighbors to connect in density 
        # based nearest neighbor graph (d-kNN) (current fixed values are 2 and 10)
        upper_nn = 10
        lower_nn = 2

        # assign number of neighbors to connect to, to each datapoint 
        sorted_nn = np.linspace( lower_nn, upper_nn, n_down, dtype=int)
        nn = np.zeros( n_down, dtype=int)
        nn[np.argsort( down_density)] = sorted_nn

        # create adjacency matrix to hold neighbor connections for d-kNN
        knn_adj = np.zeros( ( n_down, n_down), dtype=int)
        for zz in range( n_down):
            knn_adj[zz,np.argsort( Dist[zz,:])[1:nn[zz]]] = 1
        # to make symetric add adj with transpose
        knn_adj = np.add( knn_adj, knn_adj.T)
        
        # make sure there is only one component by constructing a MST
        Dist_csr = csr_matrix( np.triu(Dist))
        Tcsr     = minimum_spanning_tree( Dist_csr)
        mst_adj  = pd.DataFrame( Tcsr.todense()).values
        mst_adj  = np.add( mst_adj, mst_adj.T)
        
        # add the two adjacency matrices
        adj = np.add( knn_adj, mst_adj)
        
        # make sure overlaping neighbors arnt double counted
        adj[adj>0] = 1.0

        # normalize the orginal densities of the downsampled data points
        norm = preprocessing.MinMaxScaler()
        dens_norm = np.ravel( norm.fit_transform( down_density.reshape( -1, 1).astype( np.float)))

        # weight edges of d-kNN by inverse of orginal densities
        den_adj = np.zeros( ( n_down, n_down), dtype=float)
        print( "Constructing density kNN")
        # get coordinates of connections from adjacency matrix
        adj_coords = np.nonzero( np.triu( adj))
        for hh, uu in zip( adj_coords[0], adj_coords[1]):
            # take the minimum density of nodes connected by the edge
            # add 0.1 so that no connection is lost (not equal to zero)
            den_adj[hh,uu] = 1.1 - ( min( [dens_norm[hh], dens_norm[uu]]))
        # make symetric 
        den_adj  = np.add( den_adj, den_adj.T)
        # final edge weights are product of density weights and distance matrix
        dist_weighted_adj = np.multiply( Dist, adj)
        dens_weighted_adj = np.multiply( Dist, den_adj)
        # create undirected igraph instance using weighted matrix
        d_knn = _igraph.Graph.Weighted_Adjacency( dist_weighted_adj.tolist(), loops=False, mode=ADJ_UNDIRECTED)

        print( "finding endstates")
        # get closeness of graph and standardize to aid in endstate identification
        cls     = np.array( d_knn.closeness( weights="weight"))
        scaler  = preprocessing.StandardScaler()
        std_cls = scaler.fit_transform( cls.reshape(-1,1)).ravel()

        # using closeness as threshold (default value = 0.0) get potential endstates
        low_cls = down[std_cls<=cls_thresh]
        # array to hold silhouette score for each cluster try
        sil_score = [0]*potential_clusters

        # prefrom K means clustering and score each attempt
        for ss in range( potential_clusters):
            kmeans_model  = _KMeans( n_clusters=ss+2, random_state=10).fit( low_cls)
            label         = kmeans_model.labels_
            sil_score[ss] = metrics.silhouette_score( low_cls, labels=label, metric='l2')

        # find most likely number of clusters from scores above and double to allow for rare cell types
        num_clusters = ( np.argmax( sil_score) + 2)
        clust_model = _KMeans( n_clusters=num_clusters, random_state=10).fit( low_cls)
        label      = clust_model.labels_
        print( "Number of endstates found -> {0}".format( num_clusters))

        endstates = clust_model.cluster_centers_
        endstates_ind = np.zeros( (num_clusters, 1), dtype=int)
        for ii in range( num_clusters):
            endstates_ind[ii] = find_closest_ind( endstates[ii], data)
        
        endstates_ind = endstates_ind.ravel()
        endstates = data[endstates_ind,:]
        num_clusters = len( endstates_ind)
        
        # Endstate data points were picked from full data set, so need to be appended to down and down_ind
        # Create array to hold where end_states are located within the downsampled dataset
        cen_ind = np.zeros( num_clusters, dtype=int)
        ind = n_down
        for es in range( num_clusters):
            # first need to check if they are already in the graph, if not:
            if( ~np.in1d( endstates_ind[es], down_ind)):
                down     = np.vstack( ( down, endstates[es]))
                down_ind = np.append( down_ind, endstates_ind[es])
                cen_ind[es] = ind
                ind = ind + 1
            # if data point is already in down
            else:
                cen_ind[es] = np.argwhere( endstates_ind[es]==down_ind).ravel()[0]
                continue
        
        # re-initialize using density and distance weighted edges         
        dens_knn = _igraph.Graph.Weighted_Adjacency( dens_weighted_adj.tolist(), loops=False, mode=ADJ_UNDIRECTED)

        # add endstate data points to the already constructed dens_knn graph, connecting to 2 closest neighbors
        # future update will so that number of edges is based on density of data point
        knn_num = 2
        # add nodes to graph that will represent endstates
        dens_knn.add_vertices( num_clusters)
        # get distance matrix to be used for finding closeset neighbors in graph
        end_dist = np.array( pairwise_distances( endstates, down[:-num_clusters], n_jobs=1))
        for kk in range( num_clusters):
            edg_wts = np.sort( end_dist[kk,:])[1:knn_num+1]
            edg_ids = np.argsort( end_dist[kk,:])[1:knn_num+1]
            for jj in range( knn_num):
                # no need to connect if connection is already present
                if( edg_wts[jj]<2.0e-06):
                    continue
                # if not present add edge with distance/density weight
                else:
                    dens_knn.add_edge( cen_ind[kk], edg_ids[jj], weight=edg_wts[jj]*(1-dens_norm[edg_ids[jj]]))
        
        print( "hierarchical placing")
        # perform hierarchical placement of endstates (find shortest paths connecting them within d_knn)
        hi_pl, hi_pl_ind = hierarchical_placement( dens_knn, cen_ind)
        print( "consensus aligning")
        # perform consensus alignment of hierarchical placement data points
        aligned_ind = consensus_alignment( down, hi_pl_ind.copy(), data, density, noise)
        # add orginal endstates back into aligned list of indices 
        al_es_ind = np.append( cen_ind, np.unique( aligned_ind[~np.in1d( aligned_ind, cen_ind)]))
        # perform hierarchical placement of of newly aligned data points
        al_hi_pl, al_hi_pl_ind = hierarchical_placement( dens_knn, al_es_ind)
        # rerun hierarchical placement on the aligned placement graph to eliminate superfluous edges
        # by re-feeding it the orginal endstate indices
        creode_graph, creode_ind = hierarchical_placement( al_hi_pl, range( len( cen_ind)))
        creode_graph.simplify( combine_edges="mean")
        print( "saving files for run_num {0}".format( run_itr + 1))
        np.savetxt( file_path + "ind_{0}.csv".format( start_id), down_ind[al_hi_pl_ind[creode_ind]], delimiter=',')
        creode_graph.save( file_path + "adj_{0}.txt".format( start_id), format="adjacency" )
        
        start_id = start_id + 1
        
    # return to normal treatment of print statements
    sys.stdout = old_stdout
        
    return( creode_graph, down_ind[al_hi_pl_ind[creode_ind]])
    
#################################################

def pCreode_extended_trajectories( data, density, noise, target, file_path, num_runs=100, potential_clusters=10, cls_thresh=0.0, start_id=0, clust_scaler=2, mute=False):
    ''' 
    Function for running full pCreode algorithm, with the addition of principle component extremes found to be under the closeness threshold added as endstates
    :param data:    numpy ndarray of data set
    :param density: numpy array of calculated densities for each datapoint
    :param noise:   value for noise threshold, densities below value will be removed during downsampling
    :param target:  value for target density
    :param file_path: path to directory where output files will be stored
    :param num_runs:  number of independent runs to perform, default is 100 
    :param potential_clusters: value for upper range of number of clusters to search for, default value is 10
    :param cls_thresh: value for closeness threshold use to separate potential endstates from transitional cell types default value is 0.0
    :param start_id: integer at which to start labeling output graphs, allows for addition of graphs to previously ran lot
    :param  mute: boolean operator to suppress print statements
    :param clust_scaler: scaling factor for number of clusters, default is 2 to help identify rare cell types
    :return: will save creode files in given directory
    '''
    if not ( isinstance( data, np.ndarray)):
        raise TypeError( 'data variable must be numpy ndarray')
    if not ( isinstance( density, np.ndarray)):
        raise TypeError( 'data variable must be numpy array')
    if not ( _os.path.exists( file_path)):
        raise TypeError( 'please supply a valid directory')
        
    # Save sys.stdout to return print output if muted 
    old_stdout = sys.stdout
    # Mute print statements if True
    if( mute==True):
        sys.stdout = open( os.devnull, 'w')
        
    print( "Performing {0} independent runs, may take some time".format( num_runs))
  
    for run_itr in range( num_runs):
        
        # get downsampled dataset
        down, down_ind = Down_Sample( data, density, noise, target)

        # array for orginal density (prior to downsampling) of downsampled data points
        down_density = density[down_ind]
        n_down       = len( down)

        # get distance matrix for down sampled dataset
        Dist = np.array( pairwise_distances( down, down, n_jobs=1))

        # set upper and lower thresholds for number of neighbors to connect in density 
        # based nearest neighbor graph (d-kNN) (current fixed values are 2 and 10)
        upper_nn = 10
        lower_nn = 2

        # assign number of neighbors to connect to, to each datapoint 
        sorted_nn = np.linspace( lower_nn, upper_nn, n_down, dtype=int)
        nn = np.zeros( n_down, dtype=int)
        nn[np.argsort( down_density)] = sorted_nn

        # create adjacency matrix to hold neighbor connections for d-kNN
        knn_adj = np.zeros( ( n_down, n_down), dtype=int)
        for zz in range( n_down):
            knn_adj[zz,np.argsort( Dist[zz,:])[1:nn[zz]]] = 1
        # to make symetric add adj with transpose
        knn_adj = np.add( knn_adj, knn_adj.T)
        
        # make sure there is only one component by constructing a MST
        Dist_csr = csr_matrix( np.triu(Dist))
        Tcsr     = minimum_spanning_tree( Dist_csr)
        mst_adj  = pd.DataFrame( Tcsr.todense()).values
        mst_adj  = np.add( mst_adj, mst_adj.T)
        
        # add the two adjacency matrices
        adj = np.add( knn_adj, mst_adj)
        
        # make sure overlaping neighbors arnt double counted
        adj[adj>0] = 1.0

        # normalize the orginal densities of the downsampled data points
        norm = preprocessing.MinMaxScaler()
        dens_norm = np.ravel( norm.fit_transform( down_density.reshape( -1, 1).astype( np.float)))

        # weight edges of d-kNN by inverse of orginal densities
        den_adj = np.zeros( ( n_down, n_down), dtype=float)
        print( "Constructing density kNN")
        # get coordinates of connections from adjacency matrix
        adj_coords = np.nonzero( np.triu( adj))
        for hh, uu in zip( adj_coords[0], adj_coords[1]):
            # take the minimum density of nodes connected by the edge
            # add 0.1 so that no connection is lost (not equal to zero)
            den_adj[hh,uu] = 1.1 - ( min( [dens_norm[hh], dens_norm[uu]]))
        # make symetric 
        den_adj  = np.add( den_adj, den_adj.T)
        # final edge weights are product of density weights and distance matrix
        dist_weighted_adj = np.multiply( Dist, adj)
        dens_weighted_adj = np.multiply( Dist, den_adj)
        # create undirected igraph instance using weighted matrix
        d_knn = _igraph.Graph.Weighted_Adjacency( dist_weighted_adj.tolist(), loops=False, mode=ADJ_UNDIRECTED)

        print( "finding endstates")
        # get closeness of graph and standardize to aid in endstate identification
        cls     = np.array( d_knn.closeness( weights="weight"))
        scaler  = preprocessing.StandardScaler()
        std_cls = scaler.fit_transform( cls.reshape(-1,1)).ravel()

        # using closeness as threshold (default value = 0.0) get potential endstates
        cls_mask    = std_cls<=cls_thresh
        low_cls     = down[cls_mask]
        low_cls_ind = down_ind[cls_mask]
        # array to hold silhouette score for each cluster try
        sil_score = [0]*potential_clusters

        # prefrom K means clustering and score each attempt
        for ss in range( potential_clusters):
            kmeans_model  = _KMeans( n_clusters=ss+2, random_state=10).fit( low_cls)
            label         = kmeans_model.labels_
            sil_score[ss] = metrics.silhouette_score( low_cls, labels=label, metric='l2')

        # find most likely number of clusters from scores above and scale by clust_fact
        num_clusters = ( np.argmax( sil_score) + 2) * int( clust_scaler)
        clust_model  = _KMeans( n_clusters=num_clusters, random_state=10).fit( low_cls)
        label        = clust_model.labels_
        print( "Number of endstates found -> {0}".format( num_clusters))

        endstates_ind = np.zeros( [], dtype=int)
        for ii in range( num_clusters):
            # get the coorindates for each cell in cluster ii
            clust_inds = np.argwhere(ii==label)
            # throughout small clusters
            if( len( clust_inds)<=10):
                print( "Cluster with less than 10 cells found and removed")
                continue
            
            clust_data = low_cls[clust_inds]
            
            # pull and sort closeness values, pulling 10 highest for finding center
            high_cls_args = np.argsort( std_cls[cls_mask][clust_inds].ravel())[:min(len(clust_inds),10)]
            # median for highest ten data points are the new centers
            endstate = np.mean( clust_data[high_cls_args], axis=0)
            endstates_ind = np.append( endstates_ind, find_closest_ind( endstate, data))
        
        endstates_ind = endstates_ind.ravel()
        endstates = data[endstates_ind,:]
        num_clusters = len( endstates_ind)
        
        # Endstate data points were picked from full data set, so need to be appended to down and down_ind
        # Create array to hold where end_states are located within the downsampled dataset
        cen_ind = np.zeros( num_clusters, dtype=int)
        ind = n_down
        for es in range( num_clusters):
            # first need to check if they are already in the graph, if not:
            if( ~np.in1d( endstates_ind[es], down_ind)):
                down     = np.vstack( ( down, endstates[es]))
                down_ind = np.append( down_ind, endstates_ind[es])
                cen_ind[es] = ind
                ind = ind + 1
            # if data point is already in down
            else:
                cen_ind[es] = np.argwhere( endstates_ind[es]==down_ind).ravel()[0]
                continue
        
        # re-initialize using density and distance weighted edges         
        dens_knn = _igraph.Graph.Weighted_Adjacency( dens_weighted_adj.tolist(), loops=False, mode=ADJ_UNDIRECTED)

        # add endstate data points to the already constructed dens_knn graph, connecting to 2 closest neighbors
        # future update will so that number of edges is based on density of data point
        knn_num = 2
        # add nodes to graph that will represent endstates
        dens_knn.add_vertices( num_clusters)
        # get distance matrix to be used for finding closeset neighbors in graph
        end_dist = np.array( pairwise_distances( endstates, down[:-num_clusters], n_jobs=1))
        for kk in range( num_clusters):
            edg_wts = np.sort( end_dist[kk,:])[1:knn_num+1]
            edg_ids = np.argsort( end_dist[kk,:])[1:knn_num+1]
            for jj in range( knn_num):
                # no need to connect if connection is already present
                if( edg_wts[jj]<2.0e-06):
                    continue
                # if not present add edge with distance/density weight
                else:
                    dens_knn.add_edge( cen_ind[kk], edg_ids[jj], weight=edg_wts[jj]*(1-dens_norm[edg_ids[jj]]))
        
        print( "hierarchical placing")
        # perform hierarchical placement of endstates (find shortest paths connecting them within d_knn)
        hi_pl, hi_pl_ind = hierarchical_placement( dens_knn, cen_ind)
        print( "consensus aligning")
        # perform consensus alignment of hierarchical placement data points
        aligned_ind = consensus_alignment( down, hi_pl_ind.copy(), data, density, noise)
        # add orginal endstates back into aligned list of indices 
        al_es_ind = np.append( cen_ind, np.unique( aligned_ind[~np.in1d( aligned_ind, cen_ind)]))
        # perform hierarchical placement of of newly aligned data points
        al_hi_pl, al_hi_pl_ind = hierarchical_placement( dens_knn, al_es_ind)
        # rerun hierarchical placement on the aligned placement graph to eliminate superfluous edges
        # by re-feeding it the orginal endstate indices
        creode_graph, creode_ind = hierarchical_placement( al_hi_pl, range( len( cen_ind)))
        creode_graph.simplify( combine_edges="mean")
        print( "saving files for run_num {0}".format( run_itr + 1))
        np.savetxt( file_path + "ind_{0}.csv".format( start_id), down_ind[al_hi_pl_ind[creode_ind]], delimiter=',')
        creode_graph.save( file_path + "adj_{0}.txt".format( start_id), format="adjacency" )
        
        start_id = start_id + 1
        
    # return to normal treatment of print statements
    sys.stdout = old_stdout
        
    return( creode_graph, down_ind[al_hi_pl_ind[creode_ind]])
   
    
