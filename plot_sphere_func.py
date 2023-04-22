import torch
import numpy as np

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from itertools import product, combinations
from project_func import proj_to_sphere



def plot_sphere(shape, view='front'):
    fig = plt.figure(figsize=shape)#, layout="constrained")
    ax = fig.add_subplot(projection='3d')
    ax.set_box_aspect((1,1,1))
    ax.grid(False)
    ax.axis('off')
    ax.margins(x=0,y=0, z=0)
    if view=='front':
        ax.view_init(elev=30., azim=45)
    elif view=='side':
        ax.view_init(elev=4., azim=330)

    # draw sphere
    u, v = np.mgrid[0:2*np.pi:25j, 0:np.pi:15j]
    x = np.cos(u)*np.sin(v)
    y = np.sin(u)*np.sin(v)
    z = np.cos(v)
    ax.plot_wireframe(x, y, z, color="grey", alpha=.3)

    # axis lines
    ax.plot([-1,1], [0,0], zs=[0,0], color='k',linewidth=0.7)
    ax.plot([0,0], [-1,1], zs=[0,0], color='k',linewidth=0.7)
    ax.plot([0,0], [0,0], zs=[-1,1], color='k',linewidth=0.7)
    return [fig, ax]



def plot_archetypes(archetypes, ax, label, color):
    # corner points
    ax.scatter(archetypes[:,0], archetypes[:,1], archetypes[:,2], marker='o', color=color, s=60)
    # lines between corners
    corner_pairs = [(i,i+1) for i in range(archetypes.shape[0]-1)]
    corner_pairs.append((archetypes.shape[0]-1,0))
    if archetypes.shape[0] == 4:
        corner_pairs = [(0, 2), (2, 3), (3, 1), (1, 0)]
    a=archetypes
    for (c1,c2) in corner_pairs:
        if (c1,c2) == corner_pairs[-1]:
            ax.plot([a[c1,0],a[c2,0]], [a[c1,1],a[c2,1]], zs=[a[c1,2],a[c2,2]], color=color,linewidth=2, label=label)
        else:
            ax.plot([a[c1,0],a[c2,0]], [a[c1,1],a[c2,1]], zs=[a[c1,2],a[c2,2]], color=color,linewidth=2)

        
def proj_line_to_sphere(endpoint1, endpoint2):
    lines = ()
    for i in range(3):
        lines += (torch.linspace(endpoint1[i],endpoint2[i],100),)
    ls = torch.stack(tensors=lines, dim=1)
    ls_proj = proj_to_sphere(ls)
    ls_proj = ls_proj.numpy()
    return ls_proj
    
    
def plot_archetypes_on_sphere(archetypes, ax, label, color):
    # archetype coordinates on sphere
    archetypes_sphere = proj_to_sphere(archetypes)
    archetypes_sphere = archetypes_sphere.numpy()
    # plot archetype corners
    ax.scatter(archetypes_sphere[:,0], archetypes_sphere[:,1], archetypes_sphere[:,2], 
               marker='o', color=color, s=60)
    # create lines
    corner_pairs = [(i,i+1) for i in range(archetypes.shape[0]-1)]
    corner_pairs.append((archetypes.shape[0]-1,0))
    lines = []
    for (c1,c2) in corner_pairs:   
        line1 = proj_line_to_sphere(archetypes_sphere[c1], archetypes_sphere[c2])
        lines.append(line1)
        
    # plot lines on sphere
    for line in lines:
        if np.all(line == lines[-1]):  # to only get 1 label
            ax.plot(line[:,0],line[:,1],line[:,2],'-',color=color,linewidth=2, label=label)
        else:
            ax.plot(line[:,0],line[:,1],line[:,2],'-', color=color,linewidth=2)
    
    
def plot_samples(points, ax, label, color,means=False):
    size = 2
    if means:
        size=60
    ax.scatter(points[:,0], points[:,1], points[:,2], 
               marker='o', color=color, s=size, label=label)
    
    
def plot_samples_on_sphere(points, ax, label, color):
    points_sphere = proj_to_sphere(points)
    points_sphere = points_sphere.numpy()
    # sample points on sphere
    ax.scatter(points_sphere[:,0], points_sphere[:,1], points_sphere[:,2], 
               marker='o', color=color, s=2, label=label)