# Bachelor project: 
## Characterizing Lake Water Levels using Clustering and Archetypal Analysis Approaches

This is the code repository for my bachelor project. 

# Contents

## Notebooks
folderscripts: creates the txt files with folder and file names needed to load the data

data_dict_build:  builds the data dictionary from the datafolders and gets the names of lakes from the csv
data_exploration:  plots and computaitons for data section
data_vis_wlts: plots and computations for water level time series. individual lakes, boxplot and heatmap
data_prep_steps: steps for running the data_prep and interpolation
data_lake_plots: plots of all time series subject to various criteria and interpolation

dirichlet synthetic torch: plots of dirichlet samples in 2D and 3D
plot_datasamples: plot the spheres with only the samples

evaluate: run model evaluation on a single data set
evaluate loop: run model evaluation on several data set and several models 
evaluate_combine_dicts: combine evaluation results (when you have only had to re-run a single model)
evaluate_vis: visualise evaluation results


## Python scripts

dirichlet_func:  function to pull samples from the dirichlet distribution
project_func:  function to project datapoint(s) to unit hypersphere

synth_data_samples:  generates the synthetic data samples
prep_lake_data: clear the lake data and get it on the right format

models:  has the model structure and loss functions built on torch nn structure for AA, DAA and NMF models
model_loop4: function to run the model training/evaluation loop
model_train: sub function for the model_loop to do the model training

NMI_func:  functions to compute NMI between two S matrices

eval_model_plots: functions to plot model evaluation and save archetypes for GIS visualisations
plot_sphere_func:  functions to plot sphere as well as samples and archetypes on sphere


## Folders
saves: has the data dictionary and results from model evaluations
plots: has all plots

