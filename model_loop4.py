import torch
import pickle 
from sklearn.mixture import GaussianMixture
import warnings


from models import AA, DAA, NMF, my_COSINEloss, my_LSloss
from model_train import train_model
from eval_model_plots import makefolder



def model_eval(K, J, L, n_epoch, data, X, X_ids=None, W=None, archetypes=None, Strue=None, new_time=None, models = ['DAA','AA', 'NMF'], tol=1e-5, save=True):
    size = X.shape[0]
    # learning rate scaled to data
    learning_rates= {'DAA': 0.05, 'AA': 0.01, 'NMF': 0.1, 'GMM':None} 
  
    # dict for data info and results
    res = {'data': {'X': X, 
                    'X_ids': X_ids, 
                    'K': K, 
                    'W': W, 
                    'data_id':data, 
                    'archetypes':archetypes, 
                    'Strue': Strue, 
                    'new_time': new_time,
                    'lr': learning_rates,
                    'n_epoch': n_epoch,
                    'tol': tol}}
 
    # model evaluation
    for model in models:
        print(f'Model {model}')
        if 'AA' in model:
            res[model] = {'all_loss_t': torch.zeros(len(K), J, L, n_epoch),
                          'final_loss': torch.zeros(len(K), J, L),
                          'end_epoch': torch.zeros(len(K), J, L),
                          'set_archetypes': [], # best from each set
                          'set_S': [], # best from each set
                          'best_archetypes': [], # best from each number of archetypes
                          'best_S': []}  # best from each number of archetypes
        if 'NMF' in model:
            res[model] = {'all_loss_t': torch.zeros(len(K), J, L, n_epoch),
                          'final_loss': torch.zeros(len(K), J, L),
                          'end_epoch': torch.zeros(len(K), J, L),
                          'set_H': [], 
                          'set_W': [], 
                          'best_H': [],
                          'best_W': []}     
        if 'GMM' in model:
            res[model] = {'all_lb': torch.zeros(len(K), J, L),
                          'conv': torch.zeros(len(K), J, L),
                          'end_epoch': torch.zeros(len(K), J, L),
                          'set_means': [], 
                          'set_weights': [],
                          'set_covs': [],
                          'set_resp': [],
                          'best_means': [], 
                          'best_weights': [],
                          'best_covs': [],
                          'best_resp': []}
        
        for i,k in enumerate(K): # try different number of archetypes/components
            print(f"Archetype/component {i+1}/{len(K)}:", end =" ")
            # initiate lists to save archetype and S matrices
            K_S = []
            K_arc = []
            K_H = []
            K_W = []
            K_means = []
            K_weights = []
            K_covs = []
            K_resp = []
            K_final_loss = torch.zeros(J)
            for j in range(J):  #  J sets
                print(f"Set {j+1}/{J},", end =" ")
                # initiate lists to save archetype and S matrices 
                set_S = []
                set_arc = []
                set_H = []
                set_W = []
                set_means = []
                set_weights = []
                set_covs = []
                set_resp = []
                set_final_loss = torch.zeros(L)
                for l in range(L):   # L random initialisations of C and S
                    # model training
                    if 'GMM' in model:
                        gmm = GaussianMixture(n_components=k, covariance_type='full', max_iter=n_epoch, tol=tol, init_params='random',verbose=False)
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            gmm.fit(X)
                        res[model]['all_lb'][i,j,l] = gmm.lower_bound_
                        res[model]['conv'][i,j,l] = gmm.converged_
                        res[model]['end_epoch'][i,j,l] = gmm.n_iter_
                        set_means.append(torch.tensor(gmm.means_))
                        set_weights.append(torch.tensor(gmm.weights_))
                        set_covs.append(torch.tensor(gmm.covariances_))
                        set_resp.append(torch.tensor(gmm.predict_proba(X)))
                        
                    else:    # AA, DAA and NMF
                        train_results = train_model(model,X,k,learning_rates[model], n_epoch,W,tol)
                        if 'AA' in model: # DAA and AA
                            loss_all, final_archetypes, final_S, end_epoch = train_results
                            set_arc.append(final_archetypes) 
                            set_S.append(final_S) 
                        elif 'NMF' in model:
                            loss_all, final_H, final_W, end_epoch = train_results
                            set_H.append(final_H)
                            set_W.append(final_W)
                        res[model]['all_loss_t'][i, j, l] = loss_all
                        res[model]['end_epoch'][i, j, l] = end_epoch
                        res[model]['final_loss'][i, j, l] = loss_all[end_epoch]
                        set_final_loss[l] = loss_all[end_epoch]

                # for set j, find best model instance out of the L initialisations   
                # keep the parameters for this instance only
                if 'GMM' in model:
                    lb_max = torch.max(res[model]['all_lb'][i,j], dim=0)
                    max_idx = lb_max.indices
                    K_final_loss[j] = lb_max.values
                    K_means.append(set_means[max_idx])
                    K_weights.append(set_weights[max_idx])
                    K_covs.append(set_covs[max_idx])
                    K_resp.append(set_resp[max_idx])
                else:
                    loss_min = torch.min(set_final_loss, dim=0)
                    min_idx = loss_min.indices
                    K_final_loss[j] = loss_min.values
                    if 'AA' in model:
                        K_S.append(set_S[min_idx])
                        K_arc.append(set_arc[min_idx])
                    elif 'NMF' in model:
                        K_H.append(set_H[min_idx])
                        K_W.append(set_W[min_idx])
            # for number of archetypes trial i, find best model instance out of the J sets
            # keep the para,eters for this instance
            
            if 'GMM' in model:
                loss_max = torch.max(K_final_loss, dim=0)
                max_idx = loss_max.indices
                # save set means, weights, covs for this K to dict
                res[model]['set_means'].append(K_means)
                res[model]['set_weights'].append(K_weights)
                res[model]['set_covs'].append(K_covs)
                res[model]['set_resp'].append(K_resp)
                # save best means, weights, covs for this K to dict
                res[model]['best_means'].append(K_means[max_idx])
                res[model]['best_weights'].append(K_weights[max_idx])
                res[model]['best_covs'].append(K_covs[max_idx])
                res[model]['best_resp'].append(K_resp[max_idx])
            else:
                loss_min = torch.min(K_final_loss, dim=0)
                min_idx = loss_min.indices
                if 'AA' in model:
                    # save set archetypes and S matrices for this K to dict
                    res[model]['set_S'].append(K_S)
                    res[model]['set_archetypes'].append(K_arc)
                    # save best archetype and S for this K to dict
                    res[model]['best_S'].append(K_S[min_idx])
                    res[model]['best_archetypes'].append(K_arc[min_idx])
                elif 'NMF' in model:
                    # save set H and W for this K to dict
                    res[model]['set_H'].append(K_H)
                    res[model]['set_W'].append(K_W)
                    # save best H and W for this K to dict
                    res[model]['best_H'].append(K_H[min_idx])
                    res[model]['best_W'].append(K_W[min_idx])         

            print(" ")

        # save model dict
        if save:
            res_model = dict()
            res_model['data'] = res['data']
            res_model[model] = res[model]
            folder = f"saves/data{data}_size{size}_arc{len(K)}_sets{J}_init{L}_epoch{n_epoch}"
            makefolder(folder)
            filename = folder + f'/res_dict_model{model}_data{data}_size{size}_arc{len(K)}_sets{J}_init{L}_epoch{n_epoch}.pkl'
            with open(filename, 'wb') as f:
                pickle.dump(res_model, f)
            print(f'saved {model} dict')
    
    # save full dict
    if save:
        folder = f"saves/data{data}_size{size}_arc{len(K)}_sets{J}_init{L}_epoch{n_epoch}"
        makefolder(folder)
        filename = folder + f'/res_dict_data{data}_size{size}_arc{len(K)}_sets{J}_init{L}_epoch{n_epoch}.pkl'
        with open(filename, 'wb') as f:
            pickle.dump(res, f)
        print('saved full dict')
        
    print('Done')
    return res