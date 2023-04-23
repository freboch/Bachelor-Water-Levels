import torch

def proj_to_sphere(points):
    # (1/norm(P))*P
    ones = torch.ones(points.shape[0])
    norm = torch.linalg.norm(points, dim=1)
    points_proj = ((ones/norm)*points.T).T
    return points_proj