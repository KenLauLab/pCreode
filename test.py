import numpy as np
import pandas as pd
import pcreode

import matplotlib
matplotlib.use('Agg')

file_nm = "data/synthetic.csv"
expression = pd.read_csv( file_nm, skiprows=[0])

# test pca functions
data_pca = pcreode.PCA( expression)
data_pca.get_pca()

pca_reduced_data = data_pca.pca_set_components( min( 3, expression.shape[1]))
data_pca.pca_plot_explained_var()

# calculate density
dens = pcreode.Density( pca_reduced_data)
best_guess = dens.radius_best_guess()
density = dens.get_density( radius=best_guess, mute=True)
best_guess_2 = dens.nearest_neighbor_hist()
dens.density_hist()

# get downsampling parameters
noise, target = pcreode.get_thresholds( pca_reduced_data)

num_runs = 2
file_path = "test/"

# run pCreode
out_graph, out_ids = pcreode.pCreode(
  data=pca_reduced_data,
  density=density,
  noise=noise,
  target=target,
  file_path=file_path,
  num_runs=num_runs,
  mute=True
)
# find endstates
endstates_ind, down_ind, clust_ids, std_cls = pcreode.find_endstates( pca_reduced_data, density, noise, target, mute=True)
# pca extremes
out_graph, out_ids = pcreode.pCreode_pca_extremes( pca_reduced_data, density, noise, target, file_path, num_runs=num_runs, mute=True)
# rare cell types
out_graph, out_ids = pcreode.pCreode_rare_cells( pca_reduced_data, density, noise, target, file_path, rare_clusters=[[0,1]], num_runs=num_runs, mute=True)
# supervised cell types
out_graph, out_ids = pcreode.pCreode_supervised( pca_reduced_data, density, noise, target, file_path, man_clust=[[0,1],[50,51]], num_runs=num_runs, mute=True)
# sparse cell types
out_graph, out_ids = pcreode.pCreode_sparse( pca_reduced_data, density, noise, target, file_path, num_runs=num_runs, mute=True)


# score graphs, returns a vector of ranks by similarity
graph_ranks = pcreode.pCreode_Scoring( data=pca_reduced_data, file_path=file_path, num_graphs=num_runs, mute=True)
# select most representative graph
gid = graph_ranks[0]

# extract cell graph
analysis = pcreode.Analysis(
  file_path=file_path,
  graph_id=gid,
  data=pca_reduced_data,
  density=density,
  noise=noise
)

analysis.plot_save_graph( seed=0, overlay=expression["X"], file_out=file_path, upper_range=3, node_label_size=0)





print "all good"
