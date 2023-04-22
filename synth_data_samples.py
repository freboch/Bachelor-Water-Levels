import torch 
import numpy as np
from dirichlet_func import my_dirichlet_torch
from project_func import proj_to_sphere

def get_synth_data_sample(sample):
    size = 1000

    if sample == 1: # axis corners of sphere, data on sphere, 3 archetypes
        archetypes = torch.tensor([[1,0,0],[0,0,1],[0,1,0]]).float() 
        alpha =(1,1,1)
        X, Strue = my_dirichlet_torch(alpha,size,archetypes, return_S = True)
        X = proj_to_sphere(X)
    elif sample == 2:  # axis corners of sphere, data non spherical, 3 archetypes
        archetypes = torch.tensor([[1,0,0],[0,0,1],[0,1,0]]).float()
        alpha =(1,1,1)
        X, Strue = my_dirichlet_torch(alpha,size,archetypes, return_S = True) 
    elif sample == 3: # one archetype at origin, data non spherical, 3 archetypes
        archetypes = torch.tensor([[0,0,0],[0,0,1],[0,1,0]]).float()
        alpha =(1,1,1)
        X, Strue = my_dirichlet_torch(alpha,size,archetypes, return_S = True) 
    elif sample == 4: # archetypes on sphere, data on sphere, 4 archetypes
        archetypes = torch.tensor([[0,0.75,0.65],[0.2,1,0],[1,0.2,0],[0.75,0,0.65]]).float()
        archetypes = proj_to_sphere(archetypes) # make sure archetypes are fully on sphere
        alpha =(1,1,1,1)
        X, Strue = my_dirichlet_torch(alpha,size,archetypes, return_S = True)
        X = proj_to_sphere(X)
    elif sample == 5: # sample 1 but with higher alpha
        archetypes = torch.tensor([[1,0,0],[0,0,1],[0,1,0]]).float() 
        alpha =(10,10,10)
        X, Strue = my_dirichlet_torch(alpha,size,archetypes, return_S = True)
        X = proj_to_sphere(X)
    elif sample == 6: # sample 1 but with fewer datapoints
        archetypes = torch.tensor([[1,0,0],[0,0,1],[0,1,0]]).float() 
        alpha =(1,1,1)
        size = 200
        X, Strue = my_dirichlet_torch(alpha,size,archetypes, return_S = True)
        X = proj_to_sphere(X)
    return archetypes, X, Strue, size, alpha


def get_synth_lake(n,d):
    size = n
    # x-dimension 
    x = np.arange(0,d,1)
    # some fake lake archetypes
    y1 = 0.1*x**0.9*np.sin(100*x +30) 
    y2 = np.cos(50*x)
    y3 = 0.5 * np.sin(25*x)
    y4 = -0.1*x+2
    Y = np.asarray([y1,y2,y3,y4])
    # save to tensor
    Yt = torch.tensor(Y).float()
    Yt = Yt - Yt.min()
    
    archetypes = Yt
    alpha =(1,1,1,1)
    X, Strue = my_dirichlet_torch(alpha,size,archetypes, return_S = True) 
    return archetypes, X, Strue, size, alpha

def get_random_data(n,d):
    # uniform distribution random
    #X = torch.rand(n,d)
    # normal distribution random
    X = torch.randn(n,d)
    X = X - X.min()
    return X
