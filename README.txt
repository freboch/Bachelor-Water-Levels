
Notebooks
data_dict_build:  builds the data dictionary from the datafolders and gets the names of lakes from the csv
data_exploration:  plots and computaitons for data section
data_vis_wlts: plots and computations for water level time series. individual lakes, boxplot and heatmap

dirichlet synthetic torch: plots of dirichlet samples in 2D and 3D

eval_model_lake: selects lake data for run and runs model on lake data
eval_model_lake_vis: visualisation for the lake model results, saves csv for map

eval_model_synth: runs model evaluation on the synthetic data 
eval_model_synth_vis: visualisation for the synthetic data model evaluation

folderscripts: creates the txt files with folder and file names needed to load the data

plot_datasamples: old code for plotting the spheres with only the samples




Python scripts
AA_model:  has the model structure and loss functions built on torch nn structure for both AA and DAA models
dirichlet_func:  function to pull samples from the dirichlet distribution
eval_model_plots: functions to plot model evaluation: loss, loss vs epochs, NMI, spheres
model_loop: function to run the model training/evaluation loop
NMI_func:  functions to compute NMI between two S matrices
plot_sphere_func:  functions to plot sphere as well as samples and archetypes on sphere
project_func:  function to project datapoint(s) to unit hypersphere
synth_data_samples:  generates the 4 synthetic data samples


