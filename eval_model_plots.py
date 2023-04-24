import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
import os

colors = {'AA': 'blue', 
         'DAA': 'red',
         'NMF': 'green',
         'GMM': 'magenta'}



def makefolder(path):
    if not os.path.exists(path):
        os.makedirs(path)


# plot the data
def plot_data(X, data_set, projected=False, X_time = None, arc=False, lake=False):
    fig, ax = plt.subplots(1, 1,figsize=(8, 5))
    if projected:
        X = proj_to_sphere(X) 
    if arc:
        for i,archetype in enumerate(X):
            ax.plot(archetype, label=f'Archetype {i+1}')
        fig.legend(loc='lower center', ncol=5)
        plt.subplots_adjust(bottom=0.15)
    else:
        ax.plot(X.T)
    title = f'Data: {data_set}'
    if lake:
        ax.set_xlabel('time (decimal year)')
    else:
        ax.set_xlabel('Dimension')
    if X_time is not None:
        x = np.arange(X.shape[1])
        x = [i for i in x if i%12==0]
        x_time = np.around(X_time[x])
        ax.set_xticks(x,labels=x_time)
    if projected:
        title = title + ', projected'
        if lake:
            ax.set_ylabel('normalised water level')
    elif lake:
        ax.set_ylabel('water level (m)')
    else:
        ax.set_ylabel('value')
    ax.set_title(title)
    plt.savefig(f"plots/data{data_set}_X_proj{projected}_size{X.shape[0]}_dim{X.shape[1]}.png", 
                    bbox_inches='tight', facecolor ="w", dpi=200)
    plt.show()

# from https://stackoverflow.com/questions/10481990/matplotlib-axis-with-two-scales-shared-origin
def align_yaxis(ax1, ax2):
    y_lims = np.array([ax.get_ylim() for ax in [ax1, ax2]])

    # force 0 to appear on both axes, comment if don't need
    y_lims[:, 0] = y_lims[:, 0].clip(None, 0)
    y_lims[:, 1] = y_lims[:, 1].clip(0, None)

    # normalize both axes
    y_mags = (y_lims[:,1] - y_lims[:,0]).reshape(len(y_lims),1)
    y_lims_normalized = y_lims / y_mags

    # find combined range
    y_new_lims_normalized = np.array([np.min(y_lims_normalized), np.max(y_lims_normalized)])

    # denormalize combined range to get new axes
    new_lim1, new_lim2 = y_new_lims_normalized * y_mags
    ax1.set_ylim(new_lim1)
    ax2.set_ylim(new_lim2)

def plot_loss(res, synth=True):
    # get some variables from the results
    models = ['DAA', 'AA', 'NMF']
    models = [model for model in models if model in list(res.keys())]
    size = res['data']['X'].shape[0]
    [nK, J, L, n_epoch] = list(res[models[0]]['all_loss_t'].shape)
    K = res['data']['K']
    data = res['data']['data_id']
    # plot
    fig, ax = plt.subplots(1, 1,figsize=(8, 5))
    ax1 = ax.twinx() 
    ymax = 0
    ymax1 = 0
    
    
    #colors = ['tab:blue','tab:orange']
    #colors = ['red','blue', 'green', 'magenta']
    for i,model in enumerate(models):
        
        min_loss = torch.min(res[model]['final_loss'],dim=2) # the best model for each K for each set
        mean_loss_k = torch.mean(min_loss.values,dim=1).detach()
        error = torch.std(min_loss.values,dim=1).detach()
        # plot
        if model == 'DAA':
            ax.errorbar(x=K, y=mean_loss_k, yerr=error, capsize=4, color=colors[model],label=f"{model}")
            ymax = max(ymax, max(mean_loss_k+error))
        if model == 'AA' or model == 'NMF':
            ax1.errorbar(x=K, y=mean_loss_k, yerr=error, capsize=4, color=colors[model],label=f"{model}")
            ymax1= max(ymax1, max(mean_loss_k+error))
    fig.suptitle(f'Loss Curves', fontsize=16)
    test=''
    if isinstance(data, int):
        test='test'
    ax.set_title(f'Data set: {test} {data}, epochs: {n_epoch}, sets: {J}, init: {L}, n: {size}')
    ax.set_ylabel('Cosine loss')
    ax1.set_ylabel('Least squares loss')
    align_yaxis(ax, ax1)
    ax.set_xlabel('Number of archetypes K')
    ax.grid(True)
    if isinstance(data, int):
        ax.set_ylim(-0.0025,0.085)
        ax1.set_ylim(-0.005,0.16)
    elif 'random' == data or 'synth lakes' == data:
        ax.set_ylim(-0.0025,0.025)
        ax1.set_ylim(-5,50)
    else:
        ax.set_ylim(-0.0025,0.025)
        ax1.set_ylim(-5,50)
        #ax.set_ylim(top=ymax+0.05*ymax)
        #ax1.set_ylim(top=ymax1+0.05*ymax1)
    fig.legend()
    #ax.set_xticks(np.arange(9),labels=K)
    folder = f"plots/{data}_epochs{n_epoch}_sets{J}_iter{L}"
    makefolder(folder)
    filename = folder +'/' + f"loss_v_archetypes_dataset{data}.png"
    plt.savefig(filename, bbox_inches='tight', facecolor ="w", dpi=200)
    plt.show()
    
def plot_loss_gmm(res, synth=True):
    if 'GMM' not in res.keys():
        print('No GMM, so no GMM loss plot')
        return
    # get some variables from the results
    model ='GMM'
    size = res['data']['X'].shape[0]
    [nK, J, L] = list(res[model]['all_lb'].shape)
    n_epoch = res['data']['n_epoch']
    K = res['data']['K']
    data = res['data']['data_id']
    # plot
    fig, ax = plt.subplots(1, 1,figsize=(8, 5))
    
    #colors = ['tab:blue','tab:orange']
    #colors = ['red','blue', 'green', 'magenta']     
    max_loss = torch.max(res[model]['all_lb'],dim=2) # the best model for each K for each set
    mean_loss_k = torch.mean(max_loss.values,dim=1).detach()
    error = torch.std(max_loss.values,dim=1).detach()
    # plot
    ax.errorbar(x=K, y=mean_loss_k, yerr=error, capsize=4, color=colors[model],label=f"{model}")
    
    fig.suptitle(f'Lower bound LL GMM', fontsize=16)
    test=''
    if isinstance(data, int):
        test='test'
    ax.set_title(f'Data set: {test} {data}, epochs: {n_epoch}, sets: {J}, init: {L}, n: {size}')
    ax.set_ylabel('Log likelihood')
    #align_yaxis(ax, ax1)
    ax.set_xlabel('Number of components K')
    ax.grid(True)
    ax.set_ylim([-70,270])
    #if synth:
    #    ax.set_ylim(-0.005,0.1)
    #    ax1.set_ylim(-8.5,165)
    fig.legend()
    #ax.set_xticks(np.arange(9),labels=K)
    folder = f"plots/{data}_epochs{n_epoch}_sets{J}_iter{L}"
    makefolder(folder)
    filename = folder +'/' + f"loss_v_archetypes_dataset{data}_GMM.png"
    plt.savefig(filename, bbox_inches='tight', facecolor ="w", dpi=200)
    plt.show()
        
        
def plot_loss_v_epoch(res, synth=True):
    models = ['DAA', 'AA', 'NMF']
    models = [model for model in models if model in list(res.keys())]
    size = res['data']['X'].shape[0]
    [nK, J, L, n_epoch] = list(res[models[0]]['all_loss_t'].shape)
    K = res['data']['K']
    data = res['data']['data_id']
    test=''
    if isinstance(data, int):
        test='test'
    
    colors_k = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
    for model in models:
        fig, ax = plt.subplots(1, 1,figsize=(8, 5))
        fig.suptitle(f'Loss vs epochs model {model}', fontsize=16)
        
        for i, k in enumerate(K):
            all_loss_k = res[model]['all_loss_t'][i,:,:,:]
            # Get the run that ends in the lowest final loss
            mint = torch.min(all_loss_k[:,:,-1],1)
            set_idx = torch.argmin(mint[0])
            it_idx = mint[1][set_idx]
            ax.plot(all_loss_k[set_idx,it_idx,:],'-', color=colors_k[i],label=f'k={k} min')
            # get the run that ends in the highest final loss
            maxt = torch.max(all_loss_k[:,:,-1],1)
            set_idx = torch.argmax(maxt[0])
            it_idx = maxt[1][set_idx]
            ax.plot(all_loss_k[set_idx,it_idx,:],'--', color=colors_k[i],label=f'k={k} max')
        ax.set_title(f'Data set: {test} {data}, epochs: {n_epoch}, sets: {J}, init: {L}, n: {size}')    
        ax.set_ylabel('Loss')
        ax.set_xlabel('Epochs')
        if isinstance(data, int):
            if model == 'DAA':
                ax.set_ylim([-0.01,0.18])
            if model == 'AA':
                ax.set_ylim([-0.02,0.330])
        elif 'random' == data or 'synth lakes' == data:
            if model == 'DAA':
                ax.set_ylim([-0.001,0.0240])
            if model == 'AA':
                ax.set_ylim([-0.2,48])
        else: 
            if model == 'DAA':
                ax.set_ylim([-0.001,0.0240])
            if model == 'AA':
                ax.set_ylim([-0.2,48])
        #else:
            #ax.set_ylim(bottom=0)
        if model == 'NMF':
            ax.set_ylim([-0.02,1000])
        ax.grid(True)
        fig.legend(loc='center right')
        fig.tight_layout(pad=1.0)
        plt.subplots_adjust(right=0.85)
        
        folder = f"plots/{data}_epochs{n_epoch}_sets{J}_iter{L}"
        makefolder(folder)
        filename = folder +'/' + f"loss_v_epoch_dataset{data}_model{model}.png"
        plt.savefig(filename, bbox_inches='tight', facecolor ="w", dpi=200)
        plt.show()

        
        
def plot_GMM_converge(res):
    if 'GMM' not in res.keys():
        return
    model = 'GMM'
    size = res['data']['X'].shape[0]
    [nK, J, L] = list(res[model]['all_lb'].shape)
    n_epoch = res['data']['n_epoch']
    K = res['data']['K']
    data = res['data']['data_id']
    
    test=''
    if isinstance(data, int):
        test='test'
    
    conv = res['GMM']['conv']
    conv_flat  =torch.flatten(conv, start_dim=1, end_dim=2)
    
    fig, ax = plt.subplots(1, 1,figsize=(8, 5))
    ax.bar(res['data']['K'],conv_flat.sum(1), color=colors[model])
    fig.suptitle(f"Number of converged {model} models feature count", fontsize=16)
    ax.set_title(f'Data set: {test} {data}, epochs: {n_epoch}, sets: {J}, init: {L}, n: {size}')
    ax.set_xlabel("Features")
    ax.set_ylabel("Converged models")
    ax.set_xticks(res['data']['K'])
    ax.set_ylim([0,260])

    folder = f"plots/{data}_epochs{n_epoch}_sets{J}_iter{L}"
    makefolder(folder)
    filename = folder +'/' + f"conv_means_dataset{data}_model{model}.png"
    plt.savefig(filename, bbox_inches='tight', facecolor ="w", dpi=200)
    plt.show()
            
from NMI_func import *    

def plot_NMI_truth(res):
    
    models = ['DAA', 'AA']
    models = [model for model in models if model in list(res.keys())]
    #colors = ['red','blue', 'green']
    if models == []:
        return
    
    size = res['data']['X'].shape[0]
    [nK, J, L, n_epoch] = list(res[models[0]]['all_loss_t'].shape)
    K = res['data']['K']
    data = res['data']['data_id']
    X = res['data']['X']
    Strue = res['data']['Strue']
    if Strue == None:
        print('no true S, no NMI truth computed')
        return
    test=''
    if isinstance(data, int):
        test='test'
       
    fig, ax = plt.subplots(1, 1,figsize=(8, 5))
    for m,model in enumerate(models):
        NMI_list = torch.zeros(len(K),J)
        for i,k in enumerate(K): # for each number of archetypes
            for j in range(J):
                S = res[model]['set_S'][i][j]
                NMI_list[i,j] = NMI(Strue, S)
        NMI_k = torch.mean(NMI_list,dim=1).detach()
        NMI_error = torch.std(NMI_list,dim=1).detach()
        ax.errorbar(x=K, y=NMI_k, yerr=NMI_error, capsize=4, color=colors[model],label=f"{model}")
    fig.suptitle(f'Model Consistency compared to ground truth', fontsize=16)
    ax.set_title(f'Data set: {test} {data}, epochs: {n_epoch}, sets: {J}, init: {L}, n: {size}')
    ax.set_ylabel('Normalised Mutual Information')
    ax.set_xlabel('Number of components K')
    ax.grid(True)
    #if isinstance(data, int):
    ax.set_ylim(-0.02,1.02)
        
    #ax.set_ylim(0,1.02)
    fig.legend()
    #ax.set_xticks(np.arange(9),labels=K)
    folder = f"plots/{data}_epochs{n_epoch}_sets{J}_iter{L}"
    makefolder(folder)
    filename = folder +'/' + f"NMItruth_v_archetypes_dataset{data}.png"
    plt.savefig(filename, bbox_inches='tight', facecolor ="w", dpi=200)
    plt.show()

    
def plot_NMI_sets(res, synth=True):
    fig, ax = plt.subplots(1, 1,figsize=(8, 5))
    
    models = ['DAA', 'AA', 'GMM']
    models = [model for model in models if model in list(res.keys())]
    if models == []:
        return
    #colors = ['red','blue', 'green', 'magenta']
    
    size = res['data']['X'].shape[0]
    [nK, J, L, n_epoch] = list(res[models[0]]['all_loss_t'].shape)
    K = res['data']['K']
    data = res['data']['data_id']
    test=''
    if isinstance(data, int):
        test='test'
    
    sets_to_compare = [(i,i+1) for i in range(J-1)]
    sets_to_compare.append((J-1,0))
    for m, model in enumerate(models):
        NMI_list = torch.zeros(len(K),len(sets_to_compare))
        for i,k in enumerate(K): # for each number of archetypes
            for j,(r,rm) in enumerate(sets_to_compare):
                if model == 'DAA' or model == 'AA':
                    Sr = res[model]['set_S'][i][r]
                    Srm = res[model]['set_S'][i][rm]
                elif model == 'NMF':
                    Sr = res[model]['set_W'][i][r]
                    Srm = res[model]['set_W'][i][rm]
                elif model == 'GMM':
                    Sr = res[model]['set_resp'][i][r]
                    Srm = res[model]['set_resp'][i][rm]
                NMI_list[i,j] = NMI(Sr, Srm)
        NMI_k = torch.mean(NMI_list,dim=1).detach()
        NMI_error = torch.std(NMI_list,dim=1).detach()
        ax.errorbar(x=K, y=NMI_k, yerr=NMI_error, capsize=4, color=colors[model], label=f"{model}")
    fig.suptitle(f'Model Consistency comparing sets', fontsize=16)
    ax.set_title(f'Data set: {test} {data}, epochs: {n_epoch}, sets: {J}, init: {L}, n: {size}')
    ax.set_ylabel('Normalised Mutual Information')
    ax.set_xlabel('Number of components K')
    ax.grid(True)
    #if isinstance(data, int):
    ax.set_ylim(-0.02,1.1)
    fig.legend()
    #ax.set_xticks(np.arange(9),labels=K)
    folder = f"plots/{data}_epochs{n_epoch}_sets{J}_iter{L}"
    makefolder(folder)
    filename = folder +'/' + f"NMI_v_archetypes_dataset{data}.png"
    plt.savefig(filename, bbox_inches='tight', facecolor ="w", dpi=200)
    plt.show()
    
    
# visualise test data on spheres
from plot_sphere_func import *
from PIL import Image

def crop_sphere(filename):
    im = Image.open(filename)
    # Size of the image in pixels (size of original image)
    width, height = im.size
    # Setting the points for cropped image
    left = 350
    top = 0
    right = width - 350
    bottom = height - 450
    # Cropped image of above dimension
    im1 = im.crop((left, top, right, bottom))
    # save cropped image over the original
    im1.save(filename)
    

# for data examples, ot to show archetypes
def plot_data_sphere(X, archetypes, data):
    if X.shape[1] != 3:
        print('data is not 3d, no 3d-sphere plots created')
        return
    
    size = X.shape[0]
    D = archetypes.shape[0]
    angles = ['front', 'side']
    for angle in angles:
        [fig, ax] = plot_sphere(shape=(8, 8),view=angle)
        plot_samples(X, ax, color='k', label='Data')
        if data == 2 or data == 3:
            plot_archetypes(archetypes, ax, color='k', label='True convex-hull')
        elif data == 1 or data == 4 or data == 5 or data == 6:
            plot_archetypes_on_sphere(archetypes, ax, color='k', label='True convex-hull')
            
        
        #fig.suptitle(f'Test data {data}', fontsize=16)
        ax.set_title(f'Test data {data}', fontsize=16)
        
        ax.legend(loc="upper center", ncol=2)
        folder = f"plots/sphere_data_samples"
        makefolder(folder)
        filename = folder +'/' + f"sphere_{angle}_dataset{data}.png"
        plt.savefig(filename, bbox_inches='tight', facecolor ="w", dpi=300)
        plt.show()
        # crop image
        crop_sphere(filename)

def plot_test_data_sphere(res, archetypes):
    X = res['data']['X']
    if X.shape[1] != 3:
        print('data is not 3d, no 3d-sphere plots created')
        return
    
    models = list(res.keys())[1:]
    #colors = ['red','blue','green', 'magenta']
    
    size = res['data']['X'].shape[0]
    [nK, J, L, n_epoch] = list(res[models[0]]['all_loss_t'].shape)
    K = res['data']['K']
    data = res['data']['data_id']
    
    D = archetypes.shape[0]
    angles = ['front', 'side']
    for angle in angles:
        [fig, ax] = plot_sphere(shape=(8, 8),view=angle)
        plot_samples(X, ax, color='k', label='Data')
        if data == 2 or data == 3:
            plot_archetypes(archetypes, ax, color='k', label='True convex-hull')
        elif data == 1 or data == 4 or data == 5 or data == 6:
            plot_archetypes_on_sphere(archetypes, ax, color='k', label='True convex-hull')
        if 'DAA' in models:
            plot_archetypes_on_sphere(res['DAA']['best_archetypes'][D-2].detach(), ax, color=colors['DAA'], label=f'DAA {D} archetypes')
        if 'AA' in models:
            plot_archetypes(res['AA']['best_archetypes'][D-2].detach(), ax, color=colors['AA'], label=f'AA {D} archetypes')
        if 'GMM' in models:
            plot_samples(res['GMM']['best_means'][D-2],ax,color=colors['GMM'],label=f'GMM means {D} components', means=True)

        
        fig.suptitle(f'Test data {data}', fontsize=16)
        ax.set_title(f'Epochs: {n_epoch}, sets: {J}, init: {L}, n: {size}')
        
        #ax.set_title(f'Test data {data}', fontsize=16) # : \n Original Archetypes \n and data samples'
        ax.legend(loc="upper center", ncol=2)
        folder = f"plots/{data}_epochs{n_epoch}_sets{J}_iter{L}"
        makefolder(folder)
        filename = folder +'/' + f"sphere_w_arc_{angle}_dataset{data}.png"
        plt.savefig(filename, bbox_inches='tight', facecolor ="w", dpi=300)
        plt.show()
        # crop image
        crop_sphere(filename)
        

# plot lake archetypes like a time series
def plot_lake_archetypes(res, archetypes=None):
    
    models = ['DAA', 'AA']
    models = [model for model in models if model in list(res.keys())]
    
    if models == []:
        return
    
    X = res['data']['X']
    size = res['data']['X'].shape[0]
    [nK, J, L, n_epoch] = list(res[models[0]]['all_loss_t'].shape)
    K = res['data']['K']
    data = res['data']['data_id']
    
    for model in models:
        for a, archetype in enumerate(K):
    
            fig, ax = plt.subplots(1, 1,figsize=(10, 6))
            for i, arc in enumerate(res[model]['best_archetypes'][a]): 
                if res['data']['new_time'] is not None:
                    ax.plot(res['data']['new_time'], arc.detach(), ".-",linewidth=1, label=f"Archetype {i+1}")
                else:
                    ax.plot(arc.detach(), ".-",linewidth=1, label=f"Archetype {i+1}")
            if archetypes is not None:
                for i, arc in enumerate(archetypes):
                    ax.plot(arc.detach(), "k-",linewidth=1, label=f"True Archetype {i+1}")
            fig.suptitle(f"{archetype} Archetypes by model {model}", fontsize=16)
            ax.set_title(f'Data set: {data}, epochs: {n_epoch}, sets: {J}, init: {L}, n: {size}')
            ax.set_xlabel("Time")
            #ax.set_ylabel("Normalised Water level (m)")
            ax.grid(True)
            #if model == 'AA':
            #    ax.set_ylim([0,5000])
            #if model == 'DAA':
            #    ax.set_ylim([-0.2,0.45])
            ncol=5
            fig.legend(loc='lower center', ncol=ncol)
            plt.subplots_adjust(bottom=0.15)
            if archetype >5:
                plt.subplots_adjust(bottom=0.20)
            #fig.tight_layout(rect=[0, 0.03, 1, 0.99])
            folder = f"plots/{data}_epochs{n_epoch}_sets{J}_iter{L}"
            makefolder(folder)
            filename = folder +'/' + f"archetypes_lake_dataset{data}_model{model}_arc{archetype}.png"
            plt.savefig(filename, bbox_inches='tight', facecolor ="w", dpi=200)
            plt.show()
      
    
def plot_lake_means(res, archetypes=None):
    X = res['data']['X']
    if 'GMM' not in res.keys():
        return
    model = 'GMM'
    
    size = res['data']['X'].shape[0]
    [nK, J, L] = list(res[model]['all_lb'].shape)
    n_epoch = res['data']['n_epoch']
    K = res['data']['K']
    data = res['data']['data_id']
    
    for a, components in enumerate(K):

        fig, ax = plt.subplots(1, 1,figsize=(10, 6))
        for i, arc in enumerate(res[model]['best_means'][a]): 
            if res['data']['new_time'] is not None:
                ax.plot(res['data']['new_time'], arc.detach(), ".-",linewidth=1, label=f"Mean {i+1}")
            else:
                ax.plot(arc.detach(), ".-",linewidth=1, label=f"Mean {i+1}")
        if archetypes is not None:
            for i, arc in enumerate(archetypes):
                ax.plot(arc.detach(), "k-",linewidth=1, label=f"True Archetype {i+1}")
        fig.suptitle(f"{components} Means by model {model}", fontsize=16)
        ax.set_title(f'Data set: {data}, epochs: {n_epoch}, sets: {J}, init: {L}, n: {size}')
        ax.set_xlabel("Time")
        #ax.set_ylabel("Normalised Water level (m)")
        ax.grid(True)
        ncol=5
        fig.legend(loc='lower center', ncol=ncol)
        plt.subplots_adjust(bottom=0.15)
        if components >5:
            plt.subplots_adjust(bottom=0.20)
        #fig.tight_layout(rect=[0, 0.03, 1, 0.99])
        folder = f"plots/{data}_epochs{n_epoch}_sets{J}_iter{L}"
        makefolder(folder)
        filename = folder +'/' + f"means_lake_dataset{data}_model{model}_components{components}.png"
        plt.savefig(filename, bbox_inches='tight', facecolor ="w", dpi=200)
        plt.show()
        
def plot_lake_NMF(res, archetypes=None):
    X = res['data']['X']
    if 'NMF' not in res.keys():
        return
    model = 'NMF'
    size = res['data']['X'].shape[0]
    [nK, J, L, n_epoch] = list(res[model]['all_loss_t'].shape)
    n_epoch = res['data']['n_epoch']
    K = res['data']['K']
    data = res['data']['data_id']
    
    for a, components in enumerate(K):

        fig, ax = plt.subplots(1, 1,figsize=(10, 6))
        for i, arc in enumerate(res[model]['best_W'][a].T): 
            if res['data']['new_time'] is not None:
                ax.plot(res['data']['new_time'], arc.detach(), ".-",linewidth=1, label=f"W feature {i+1}")
            else:
                ax.plot(arc.detach(), ".-",linewidth=1, label=f"W feature {i+1}")
        if archetypes is not None:
            for i, arc in enumerate(archetypes):
                ax.plot(arc.detach(), "k-",linewidth=1, label=f"True Archetype {i+1}")
        fig.suptitle(f"{model} model, W with {components} features", fontsize=16)
        ax.set_title(f'Data set: {data}, epochs: {n_epoch}, sets: {J}, init: {L}, n: {size}')
        ax.set_xlabel("Time")
        #ax.set_ylabel("Normalised Water level (m)")
        ax.grid(True)
        ncol=5
        fig.legend(loc='lower center', ncol=ncol)
        plt.subplots_adjust(bottom=0.15)
        if components >5:
            plt.subplots_adjust(bottom=0.20)
        #fig.tight_layout(rect=[0, 0.03, 1, 0.99])
        folder = f"plots/{data}_epochs{n_epoch}_sets{J}_iter{L}"
        makefolder(folder)
        filename = folder +'/' + f"W_lake_dataset{data}_model{model}_components{components}.png"
        plt.savefig(filename, bbox_inches='tight', facecolor ="w", dpi=200)
        plt.show()
    
# save archetypes to csv for use in QGIS
def save_arc_csv(res):
    if res['data']['X_ids'] is None:
        print('not lake data, archetypes not saved')
        return
    
    models = ['DAA', 'AA']
    models = [model for model in models if model in list(res.keys())]

    K = res['data']['K']
    [nK, J, L, n_epoch] = list(res[models[0]]['all_loss_t'].shape)
    data = res['data']['data_id']
    print('saving')
    for model in models:
        print(f'model {model}', end=' ')
        for i,arc in enumerate(K):
            print(f'arc {i+1}/{len(K)}', end=' ')
            # the unique lakes represented
            X_ids = torch.tensor(res['data']['X_ids'])
            X_ids = X_ids[:,0]
            lakes_rep = torch.unique(X_ids) 
            #combine S-values of lakes with multiple so there is one per lake
            new_S = []
            for lake in lakes_rep:
                idx = [X_ids == lake]
                w = res['data']['W'][idx]
                s = res[model]['best_S'][i][idx]
                ws = (w*s.T).T
                new_s = torch.sum(ws,dim=0)
                new_S.append(new_s)
            new_S = torch.stack(new_S)

            dat = {'lake_id': lakes_rep.detach()}
            for d in range(new_S.shape[1]):
                dat[f'arc{d+1}'] = new_S[:,d].detach()
            df = pd.DataFrame(data = dat)
            folder = f"saves/{data}_epochs{n_epoch}_sets{J}_iter{L}"
            makefolder(folder)
            filename = folder +'/' + f"S_lake_model{model}_arc{arc}.csv"
            df.to_csv(filename,index=False)
        print(' ')
    print('done')
    
def save_H_csv(res):
    if res['data']['X_ids'] is None:
        print('not lake data, archetypes not saved')
        return

    model = 'NMF'    
    if model not in list(res.keys()):
        print('No NMF')
        return

    K = res['data']['K']
    [nK, J, L, n_epoch] = list(res[model]['all_loss_t'].shape)
    data = res['data']['data_id']
    print('saving')
    print(f'model {model}', end=' ')
    for i,arc in enumerate(K):
        print(f'NMF comp {i+1}/{len(K)}', end=' ')
        # the unique lakes represented
        X_ids = torch.tensor(res['data']['X_ids'])
        X_ids = X_ids[:,0]
        lakes_rep = torch.unique(X_ids) 
        #combine S-values of lakes with multiple so there is one per lake
        new_H = []
        for lake in lakes_rep:
            idx = X_ids == lake
            w = res['data']['W'][idx]
            h = res[model]['best_H'][i].T[idx]
            wh = (w*h.T).T
            new_h = torch.sum(wh,dim=0)
            new_H.append(new_h)
        new_H = torch.stack(new_H)

        dat = {'lake_id': lakes_rep.detach()}
        for d in range(new_H.shape[1]):
            dat[f'comp{d+1}'] = new_H[:,d].detach()
        df = pd.DataFrame(data = dat)
        folder = f"saves/{data}_epochs{n_epoch}_sets{J}_iter{L}"
        makefolder(folder)
        filename = folder +'/' + f"H_lake_model{model}_comp{arc}.csv"
        df.to_csv(filename,index=False)
    print('done')
    
def save_GMM_csv(res):
    if res['data']['X_ids'] is None:
        print('not lake data, archetypes not saved')
        return

    model = 'GMM'    
    if model not in list(res.keys()):
        print('No GMM')
        return

    K = res['data']['K']
    [nK, J, L] = list(res[model]['all_lb'].shape)
    data = res['data']['data_id']
    n_epoch = res['data']['n_epoch']
    
    print('saving')
    print(f'model {model}', end=' ')
    for i,arc in enumerate(K):
        print(f'comp {i+1}/{len(K)}', end=' ')
        # the unique lakes represented
        X_ids = torch.tensor(res['data']['X_ids'])
        X_ids = X_ids[:,0]
        lakes_rep = torch.unique(X_ids) 
        #combine S-values of lakes with multiple so there is one per lake
        new_RESP = []
        for lake in lakes_rep:
            idx = [X_ids == lake]
            w = res['data']['W'][idx]
            resp = res[model]['best_resp'][i][idx]
            w_resp = (w*resp.T).T
            new_resp = torch.sum(w_resp,dim=0)
            new_RESP.append(new_resp)
        new_RESP = torch.stack(new_RESP)

        dat = {'lake_id': lakes_rep.detach()}
        for d in range(new_RESP.shape[1]):
            dat[f'comp{d+1}'] = new_RESP[:,d].detach()
        df = pd.DataFrame(data = dat)
        folder = f"saves/{data}_epochs{n_epoch}_sets{J}_iter{L}"
        makefolder(folder)
        filename = folder +'/' + f"RESP_lake_model{model}_comp{arc}.csv"
        df.to_csv(filename,index=False)
    print(' ')
    print('done')
    return(df)