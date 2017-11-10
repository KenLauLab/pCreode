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
import os as _os

#################################################   
def plot_save_graph( seed, file_path, graph_id, data, overlay, density, file_out, upper_range=3):
    """
    Plots a p-Creode  graph with given overlay
    :param seed:      random interger to be used to seed graph plot
    :param file_path: path to directory where graph files are stored
    :param graph_id:  graph ID to plot in given directory
    :param data:      data used to find p-Creode graph (processed data set)
    :param density:   data point densities used to create p-Creode graph
    :patam overlay:   characteristic to overlay on graph, likely from preprocessed data set 
    :param file_out:  name to give saved graph
    :param up_range:  to upper range from which to normalize overlay to, this will vary with overlay
    :return: A plot of selected p-Creode graph with given overlay and saves a png in file_path with file_out name
    """
    # read adjacency matrix for graph
    adj   = pd.read_table( file_path + 'adj_{}.txt'.format(graph_id), sep=" ", header=None)
    # read in indices for nodes in terms of original data set
    ind   = np.genfromtxt( file_path + 'ind_{}.csv'.format(graph_id), delimiter=',').astype( int)
    # get distance between nodes in graph
    dist  = pairwise_distances( data[ind,:], data[ind,:], n_jobs=1, metric='l2')
    # calculate weighted adjacency matric
    w_adj = np.multiply( adj.values, dist)
    # create graph to plots
    graph = _igraph.Graph.Weighted_Adjacency( w_adj.tolist(), mode=ADJ_UNDIRECTED)
    
    # normalize densities to use for nodes sizes in graph plot
    norm_dens = preprocessing.MinMaxScaler( feature_range=(4,25))
    dens      = norm_dens.fit_transform( density.astype( float).reshape(-1, 1))[ind]
    # normalize overlay to use for node overlays
    norm_ana = preprocessing.MinMaxScaler( feature_range=(0, upper_range))
    old_ana  = norm_ana.fit_transform( overlay.values.astype( np.float).reshape(-1, 1))
    # bin the data points to each node so that an average of closest surrounding nodes is used for overlay
    bin_dist = pairwise_distances( data, data[ind])
    bin_assignments = np.argmin( bin_dist, axis=1)
    new_ana = overlay.values[ind]
    for ii in range( len( ind)):
        new_ana[ii] = np.mean( old_ana[bin_assignments==ii])
    norm_1 = np.array( new_ana, dtype=float)
    cl_vals_1 = [[]]*len( ind)
    # colors to use for overlay
    get_cl = _plt.get_cmap('RdYlBu_r')
    for jj in range( len( ind)):
        cl_vals_1[jj] = get_cl( norm_1[jj])

    graph.vs["color"] = [cl_vals_1[kk] for kk in range( len( ind))]
    random.seed( seed)
    layout = graph.layout_kamada_kawai( maxiter=2000, sigma=1000.0)
    #graph.vs["label"] = range( 1, len( ind)+1)      
    return( plot( graph, file_path + '{0}.png'.format(file_out), layout=layout, bbox=(500,500), 
                 vertex_size=dens, edge_width=2))  
    
#################################################
def Down_Sample( data, density, noise, target):
    ''' 
    Function for downsampling data 
    :param data:    numpy ndarray of data set
    :param density: numpy array of calculated densities for each datapoint
    :param noise:   value for noise threshold, densities below value will be removed during downsampling
    :param target:  value for target density 
    :return down_sampled: array of downsampled dataset 
    :return down_ind:     array of orginal indices for selected datapoints
    '''
    # set outlier and target densities
    outlier_density = np.percentile( density, noise)
    target_density  = np.percentile( density, target)
    
    # get cell probabilities based on density
    cell_prob = np.array( [0.0]*len( density))

    for ii in range( len( density)):
        # get rid of outliers
        if( density[ii]<=outlier_density):
            cell_prob[ii] = 0
        # keep data points within target range
        elif( density[ii]>outlier_density and density[ii]<=target_density):
            cell_prob[ii] = 1
        # set probability for over represented data points
        elif( density[ii]>target_density):
            cell_prob[ii] = float( target_density)/float( density[ii])

    # create an array of random floats from which to compare cell probabilities
    test_prob = np.random.random( size=len( density))

    # select ind of cells that are kept
    down_ind = np.ravel( np.argwhere(cell_prob>test_prob))
    
    # select which cells to keep based on density probablities
    down_sampled = data[down_ind,:]

    print( "Number of data points in downsample = {0}".format( len(down_sampled)))
        
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
def find_endstates( data, density, noise, target, potential_clusters=10, cls_thresh=0.0):
    ''' 
    Function for retrieving endstates, does not run connect the endstates 
    :param data:    numpy ndarray of data set
    :param density: numpy array of calculated densities for each datapoint
    :param noise:   value for noise threshold, densities below value will be removed during downsampling
    :param target:  value for target density 
    :param potential_clusters: value for upper range of number of clusters to search for, default value is 10
    :param cls_thresh: value for closeness threshold use to separate potential endstates from transitional  cell types default value is 0.0
    :return endstates_ind: array of indices for most representative cell of identified endstates 
    :return down_ind:      ndarray with indices of datapoints selected by downsampling 
    :return clust_ids:     ndarray with cluster IDs; highest valued cluster ID is for cells with high closeness, likely transititional cell types
    :return std_cls:       array of containing the closeness values for the downsampled dataset 
    '''
    if not ( isinstance( data, np.ndarray)):
        raise TypeError( 'data variable must be numpy ndarray')
    if not ( isinstance( density, np.ndarray)):
        raise TypeError( 'data variable must be numpy array')
        
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
    adj = np.zeros( ( n_down, n_down), dtype=int)
    for zz in range( n_down):
        adj[zz,np.argsort( Dist[zz,:])[1:nn[zz]]] = 1
    # to make symetric add adj with transpose
    adj = np.add( adj, adj.T)
    # make sure overlaping neighbors arnt double counted
    adj[adj>0] = 1.0

    # normalize the orginal densities of the downsampled data points
    norm = preprocessing.MinMaxScaler()
    dens_norm = np.ravel( norm.fit_transform( down_density.reshape( -1, 1).astype( np.float)))

    # weight edges of d-kNN by inverse of orginal densities
    den_adj = np.zeros( ( n_down, n_down), dtype=float)
    print( "weighting edges")
    # get coordinates of connections from adjacency matrix
    adj_coords = np.nonzero( np.triu( adj))
    for ss, tt in zip( adj_coords[0], adj_coords[1]):
        # take the minimum density of nodes connected by the edge
        # add 0.1 so that no connection is lost (not equal to zero)
        den_adj[ss,tt] = 1.1 - ( min( [dens_norm[ss], dens_norm[tt]]))
    # make symetric 
    den_adj  = np.add( den_adj, den_adj.T)
    # final edge weights are product of density weights and distance matrix
    weighted_adj = np.multiply( Dist, den_adj) 
    # create undirected igraph instance using weighted matrix
    d_knn = _igraph.Graph.Weighted_Adjacency( weighted_adj.tolist(), loops=False, mode=ADJ_UNDIRECTED)

    # need to make sure all graph components in d-kNN are connected (d-kNN is a complete graph)
    # components() returns nested array with each row containing the indices for each component
    # the largest component is listed first.
    comp   = d_knn.components( mode=WEAK)
    n_comp = len( comp)
    print( "connecting components")
    while( n_comp>1):
        # find graph component that is closest to the largest component
        rest = np.empty( (0,1), dtype=int)
        rc   = np.zeros( n_comp, dtype=int)
        for zz in range( 1, n_comp):
            # rest hold the indices for all data points not connected to largest component
            rest   = np.append( rest, comp[zz])
            # rc is an accounting variable used to identify which components the data points are in 
            rc[zz] = len( rest)
        # find the location of the closest unconnected data point
        rest_dist   = np.array( pairwise_distances( down[comp[0],:], down[rest,:], n_jobs=1))
        rest_min    = np.amin( rest_dist[np.nonzero( rest_dist)])
        rest_coords = np.argwhere( rest_dist==rest_min)[0]

        for jj in range( 1, n_comp):
            if( rc[jj]>rest_coords[1]):
                min_comp  = jj 
                comp_dist = rest_dist[:,rc[jj-1]:rc[jj]]
                break

        conn_dist = np.sort( comp_dist[np.nonzero( comp_dist)].tolist())[:1]
        for conn in conn_dist:
            comp_coords = np.argwhere( comp_dist==conn)[0]
            combined    = conn*(1 - (min( dens_norm[comp[0][comp_coords[0]]], dens_norm[comp[min_comp][comp_coords[1]]])))
            d_knn.add_edge( comp[0][comp_coords[0]], comp[min_comp][comp_coords[1]], weight=combined)
        comp    = d_knn.components( mode=WEAK)
        n_comp = len( comp)

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
    for ii in range( potential_clusters):
        print ii
        kmeans_model  = _KMeans( n_clusters=ii+2, random_state=10).fit( low_cls)
        label         = kmeans_model.labels_
        sil_score[ii] = metrics.silhouette_score( low_cls, labels=label, metric='l2')
        
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
        # add path vertices to hi_pl
        hi_pl.add_vertices( [str(ii) for ii in comp_path[1:-1]])
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
    num_y = len( ind_2)
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
            dis_y = dis_y + abs( dist_2[ii,jj] - dist_1[min_ind_ii,min_ind_jj] + min_dist_ii + min_dist_jj)
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
def pCreode_Scoring( data, file_path, num_graphs):
    ''' 
    Function returns the similarity score between two graphs
    :param data:      numpy ndarray of data set
    :param file_path: path to directory where output files are stored
    :param num_runs:  number of graphs to be scored 
    :return branch_diss: value of pairwise difference in branch counts
    :return dist_diss:   value of pairwise difference in graph distance
    '''
    if not ( isinstance( data, np.ndarray)):
        raise TypeError( 'data variable must be numpy ndarray')        
    if not ( _os.path.exists( file_path)):
        raise TypeError( 'please supply a valid directory')

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
            adj_1  = pd.read_table( file_path + 'adj_{}.txt'.format( zz), sep=" ", header=None).values
            adj_2  = pd.read_table( file_path + 'adj_{}.txt'.format( kk), sep=" ", header=None).values
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
    
#################################################
#### add option for endstates                ####
#################################################

def pCreode( data, density, noise, target, file_path, num_runs=100, potential_clusters=10, cls_thresh=0.0):
    ''' 
    Function for retrieving endstates, does not run connect the endstates 
    :param data:    numpy ndarray of data set
    :param density: numpy array of calculated densities for each datapoint
    :param noise:   value for noise threshold, densities below value will be removed during downsampling
    :param target:  value for target density
    :param file_path: path to directory where output files will be stored
    :param num_runs:  number of independent runs to perform, default is 100 
    :param potential_clusters: value for upper range of number of clusters to search for, default value is 10
    :param cls_thresh: value for closeness threshold use to separate potential endstates from transitional cell types default value is 0.0
    :return: will save creode files in given directory
    '''
    if not ( isinstance( data, np.ndarray)):
        raise TypeError( 'data variable must be numpy ndarray')
    if not ( isinstance( density, np.ndarray)):
        raise TypeError( 'data variable must be numpy array')
    if not ( _os.path.exists( file_path)):
        raise TypeError( 'please supply a valid directory')
        
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
        adj = np.zeros( ( n_down, n_down), dtype=int)
        for zz in range( n_down):
            adj[zz,np.argsort( Dist[zz,:])[1:nn[zz]]] = 1
        # to make symetric add adj with transpose
        adj = np.add( adj, adj.T)
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
        weighted_adj = np.multiply( Dist, den_adj) 
        # create undirected igraph instance using weighted matrix
        d_knn = _igraph.Graph.Weighted_Adjacency( weighted_adj.tolist(), loops=False, mode=ADJ_UNDIRECTED)

        # need to make sure all graph components in d-kNN are connected (d-kNN is a complete graph)
        # components() returns nested array with each row containing the indices for each component
        # the largest component is listed first.
        comp   = d_knn.components( mode=WEAK)
        n_comp = len( comp)
        while( n_comp>1):
            # find graph component that is closest to the largest component
            rest = np.empty( (0,1), dtype=int)
            rc   = np.zeros( n_comp, dtype=int)
            for zz in range( 1, n_comp):
                # rest hold the indices for all data points not connected to largest component
                rest   = np.append( rest, comp[zz])
                # rc is an accounting variable used to identify which components the data points are in 
                rc[zz] = len( rest)
            # find the location of the closest unconnected data point
            rest_dist   = np.array( pairwise_distances( down[comp[0],:], down[rest,:], n_jobs=1))
            rest_min    = np.amin( rest_dist[np.nonzero( rest_dist)])
            rest_coords = np.argwhere( rest_dist==rest_min)[0]

            for jj in range( 1, n_comp):
                if( rc[jj]>rest_coords[1]):
                    min_comp  = jj 
                    comp_dist = rest_dist[:,rc[jj-1]:rc[jj]]
                    break

            conn_dist = np.sort( comp_dist[np.nonzero( comp_dist)].tolist())[:1]
            for conn in conn_dist:
                comp_coords = np.argwhere( comp_dist==conn)[0]
                combined    = conn*(1 - (min( dens_norm[comp[0][comp_coords[0]]], dens_norm[comp[min_comp][comp_coords[1]]])))
                d_knn.add_edge( comp[0][comp_coords[0]], comp[min_comp][comp_coords[1]], weight=combined)
            comp    = d_knn.components( mode=WEAK)
            n_comp = len( comp)

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

        # add endstate data points to the already constructed d_knn graph, connecting to 2 closest neighbors
        # future update will so that number of edges is based on density of data point
        knn_num = 2
        # add nodes to graph that will represent endstates
        d_knn.add_vertices( num_clusters)
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
                    d_knn.add_edge( cen_ind[kk], edg_ids[jj], weight=edg_wts[jj]*(1-dens_norm[edg_ids[jj]]))
        print( "hierarchical placing")
        # perform hierarchical placement of endstates (find shortest paths connecting them within d_knn)
        hi_pl, hi_pl_ind = hierarchical_placement( d_knn, cen_ind)
        print( "consensus aligning")
        # perform consensus alignment of hierarchical placement data points
        aligned_ind = consensus_alignment( down, hi_pl_ind.copy(), data, density, noise)
        # add orginal endstates back into aligned list of indices 
        al_es_ind = np.append( cen_ind, np.unique( aligned_ind[~np.in1d( aligned_ind, cen_ind)]))
        # perform hierarchical placement of of newly aligned data points    
        al_hi_pl, al_hi_pl_ind = hierarchical_placement( d_knn, al_es_ind)
        # rerun hierarchical placement on the aligned placement graph to eliminate superfluous edges
        # by re-feeding it the orginal endstate indices
        creode_graph, creode_ind = hierarchical_placement( al_hi_pl, range( num_clusters))
        creode_graph.simplify( combine_edges="mean")
        print( "saving files for run_num {0}".format( run_itr + 1))
        np.savetxt( file_path + "ind_{0}.csv".format( run_itr), down_ind[al_hi_pl_ind[creode_ind]], delimiter=',')
        creode_graph.save( file_path + "adj_{0}.txt".format( run_itr), format="adjacency" )

    return( creode_graph, down_ind[al_hi_pl_ind[creode_ind]])

