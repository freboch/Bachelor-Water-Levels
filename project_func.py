import torch

def proj_to_sphere(points):
    # (1/norm(P))*P
    ones = torch.ones(points.shape[0])
    points_n = torch.linalg.norm(points, dim=1)
    points_proj = ((ones/points_n)*points.T).T
    return points_proj