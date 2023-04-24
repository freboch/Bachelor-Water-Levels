import torch
import numpy as np

def my_dirichlet_torch(alpha, size, corners, return_S=False):
    """
    input:
    alpha: concentration parameters alpha_i > 0. Tuble with as many elements as there are corners in the convex hull
    size: skalar number of samples to draw - will be the number of points in the sample
    corners: tensor matrix of coordinates for the convex hull, shape: (number of corners, dimension)
    output:
    sample: tensor matrix, shape (size, dimension) 
    """
    ds = np.random.dirichlet(alpha=alpha, size=size)
    ds_t = torch.from_numpy(ds).float()
    sample = torch.matmul(ds_t, corners)
    if return_S:
        return sample, ds_t
    return sample


# How to use it
# corners of the convex hull
corners = torch.tensor([[3,4], [5,8], [7,3]]).float()
#print(f'corners shape: {corners.shape}')
# number of samples
size=500
#print(f'size: {size}')
# alpha values
alpha = (1,1,1)
# Get sample
X = my_dirichlet_torch(alpha,size,corners) 
#print(f'sample shape: {X.shape}')