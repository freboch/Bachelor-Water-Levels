import pandas as pd
import numpy as np
import torch
from scipy import interpolate




def nn_lake_dict(data):
    count_wlts = 0
    count_lakes = 0
    count_wlts_nn = 0
    count_lakes_nn = 0
    for lake in list(data.keys()):
        data[lake]['nn'] = []
        for wlts in data[lake]['wlts']:
            count_wlts +=1
            if np.any(np.asarray(wlts.wl) < 0):
                data[lake]['nn'].append(False)
            else:
                data[lake]['nn'].append(True)
                count_wlts_nn +=1
        if len(data[lake]['nn']) >=1:
            count_lakes += 1
            if np.all(data[lake]['nn']):
                count_lakes_nn += 1
    print(f'Non negative wlts: {count_wlts_nn}/{count_wlts}, lakes with NN wlts: {count_lakes_nn}/{count_lakes}')
    return data



def interpolate_n_clip(X, data):
    """
    X is the new X to align the data points
    data is the data dictionary
    """
    clip = min(X)-0.2 
    lakes = list(data.keys())
    
    counts = np.zeros(6)
    counts_lakes = 0
    total = 0
    for lake in lakes:
        data[lake]['interp wl'] = []
        data[lake]['interp good'] = []
        data[lake]['clip wlts'] = []
        data[lake]['interp wlts'] = []
        for i, wlts in enumerate(data[lake]['wlts']):
            # only use wlts if it is non negative
            if not data[lake]['nn'][i]:
                data[lake]['interp good'].append(False)
                data[lake]['clip wlts'].append(np.nan)
                data[lake]['interp wlts'].append(np.nan)
            else:
                total +=1
                idx = wlts.time > clip # index for observations that are in the time frame of interest
                wlts_clipped = list(wlts.time[idx])
                # look at gaps in the cycle list
                extr = data[lake]['extr'][i]
                cycles = np.asarray(list(extr.cycle[idx]))
                comp_cycles = np.arange(min(cycles),min(cycles)+len(cycles),1)
                diff = cycles - comp_cycles
                uni_cycles = np.unique(diff)
                jumps = np.asarray([uni_cycles[j+1]-gap for j, gap in enumerate(uni_cycles[:-1])])
                if wlts_clipped[0] > X[0] or wlts_clipped[-1] < X[-1]:
                    # since we will interpolate, the time series need to have data on the outside of the new X range
                    counts[3]+=1
                    data[lake]['interp good'].append(False)
                    data[lake]['clip wlts'].append(np.nan)
                    data[lake]['interp wlts'].append(np.nan)
                    continue
                elif sum(idx) < 20:
                    # if there are less than 20 points in the time series
                    counts[2]+=1
                    data[lake]['interp good'].append(False)
                    data[lake]['clip wlts'].append(np.nan)
                    data[lake]['interp wlts'].append(np.nan)
                    continue
                elif len(uni_cycles) > 5:
                    # more than 5 holes in the time series
                    counts[4]+=1
                    data[lake]['interp good'].append(False)
                    data[lake]['clip wlts'].append(np.nan)
                    data[lake]['interp wlts'].append(np.nan)
                    continue
                elif np.any(jumps >3):
                    # more than 3 cycles in a row are missing
                    counts[5]+=1
                    data[lake]['interp good'].append(False)
                    data[lake]['clip wlts'].append(np.nan)
                    data[lake]['interp wlts'].append(np.nan)
                    continue
                elif data[lake]['conv'][i].SigmaObs[0] > 1:
                    # the standard deviation in Karinas model was more than 1 meter
                    counts[0]+=1
                    data[lake]['interp good'].append(False)
                    data[lake]['clip wlts'].append(np.nan)
                    data[lake]['interp wlts'].append(np.nan)
                    continue
                # the time series is useable
                data[lake]['interp good'].append(True)
                # interpolation
                # clipped time and water level
                x = list(wlts.time[idx])
                y = list(wlts.wl[idx])
                # interpolate
                f = interpolate.interp1d(x, y, kind='linear')
                Y = f(X)
                # save 
                data[lake]['clip wlts'].append((x,y))
                data[lake]['interp wlts'].append(Y)
        if len(data[lake]['interp good']) >=1 and np.any(data[lake]['interp good']):
            counts_lakes +=1
    print(f'wlts removed total: {np.sum(counts)}/{total} for not whole range: {counts[3]}, too short: {counts[2]},  many gaps in cycle: {counts[4]}, big holes in cycle: {counts[5]}, SigmaObs: {counts[0]}')
    print(f'lakes: {counts_lakes}')
    return data



def prep_lake(data):
    """
    Go from data dictionary to X tensor and weights
    """
    wls = []
    X_ids = []
    for lake in list(data.keys()):
        if 'interp good' in data[lake]:
            for i, good in enumerate(data[lake]['interp good']):
                if good:
                    wl_l = list(data[lake]['interp wlts'][i])
                    wls.append(torch.tensor(wl_l).float())
                    X_ids.append([lake, i])
    X = torch.stack(wls)
    # weights for each lake so they sum to 1 based on the number of timeseries per lake 
    xi = torch.tensor(X_ids)
    xi = xi[:,0]
    (xi_u, xi_u_count) = torch.unique(xi, return_counts=True)
    xi_u_weight = 1/xi_u_count

    W = torch.tensor([xi_u_weight[torch.where(xi_u == i)] for i in xi])
    return X, X_ids, W





def select_lakes(data):
    lakes = list(data.keys())
    # lake table from GIS
    folder = r"C:\Users\frede\OneDrive\Dokumenter\DTU\bachelor\GIS"
    file = r"\great_lakes.csv"
    path = folder + file
    select_lakes = pd.read_csv(path,
                    engine='python') 
    new_lakes = [lake for lake in list(select_lakes.lake_id) if lake in lakes]
    new_data = dict()
    
    for lake in new_lakes:
        new_data[lake] = data[lake]
    return new_data






def full_lake_data_prep(data, new_time, select=False):
    if select:
        data = select_lakes(data)
    data_nn = nn_lake_dict(data) 
    data_nn_interpolated = interpolate_n_clip(new_time, data_nn)
    X, X_ids, W = prep_lake(data_nn_interpolated)
    print(f'X shape: {X.shape}')
    return X, X_ids, W








