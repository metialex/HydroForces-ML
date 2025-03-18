import math
import pyvista as pv
import numpy as np

def neighbors_3D(i_0,j_0,k_0,nx,ny,nz,n_neighbors):
    """
    Returns a 1D array of i,j,k indexes of the subset of mesh
    that surrounds the i_0,j_0,k_0 cell
    Arguments:
        i_0,j_0,k_0 - indexes of the cell of interest
        nx,ny,nz - the number of cells within each direction
        n_neighbors - number of closest neighbors
    Return:
         1D arrays of i,j,k indexes of neighbors
    """
    res_size = 1+2*n_neighbors
    i_idx = np.zeros((res_size,res_size,res_size))
    j_idx = np.zeros((res_size,res_size,res_size))
    k_idx = np.zeros((res_size,res_size,res_size))
    
    for i in range(res_size):
        for j in range(res_size):
            for k in range(res_size):
                i_idx[i][j][k] = i_0-n_neighbors+i
                j_idx[i][j][k] = j_0-n_neighbors+j
                k_idx[i][j][k] = k_0-n_neighbors+k

                #Apply periodic BC
                if (i_idx[i][j][k] < 0): i_idx[i][j][k] += nx
                elif (i_idx[i][j][k] > nx-1): i_idx[i][j][k] -= nx
                if (j_idx[i][j][k] < 0): j_idx[i][j][k] += ny
                elif (j_idx[i][j][k] > ny-1): j_idx[i][j][k] -= ny
                if (k_idx[i][j][k] < 0): k_idx[i][j][k] += nz
                elif (k_idx[i][j][k] > nz-1): k_idx[i][j][k] -= nz
    return i_idx.ravel(),j_idx.ravel(),k_idx.ravel()

def D1_to_3D_idx(idx,nx,ny):
    """
    Converts 1D index to 3D index for uniform grid
    Arguments:
        idx - 1D index
        nx,ny - x,y width of the grid
    Return:
        x,y,z -components of 3D index
    """
    i = math.floor(idx%nx)
    j = math.floor((idx / nx) % ny)
    k = math.floor(idx / (nx * ny))
    return i,j,k

def D3_to_1D_idx(i,j,k,nx,ny):
    """
    Converts 3D index to 1D index for uniform grid
    Arguments:
        x,y,z -components of 3D index
        nx,ny - x,y width of the grid
    Return:
        idx - 1D index
    """
    idx = nx*ny*k+nx*j+i
    return int(idx)

def return_neighbours_1D(grid,px,py,pz,n_neighbors):
    """
    Takes the position of a particle and returns the index of subdomain
    Arguments:
        x,y,z - position of particle
        grid (pyvista object) - grid that contain info about the subdaomains
    Return:
        idx - index of the subdomain that corres
    """
    #Grid parameters
    nx,ny,nz = grid.dimensions
    nx,ny,nz = nx-1,ny-1,nz-1
    #Find the cell where particle is located
    cell = grid.find_closest_cell([px,py,pz])
    i,j,k = D1_to_3D_idx(cell,nx,ny)
    
    
    #Find neighbouring cells
    i_idx,j_idx,k_idx = neighbors_3D(i,j,k,nx,ny,nz,n_neighbors)
    idx_1D=[]
    for idx,i in enumerate(i_idx):
        idx_1D.append(D3_to_1D_idx(i_idx[idx],
                                 j_idx[idx],
                                 k_idx[idx],
                                 nx,
                                 ny))
    

    return idx_1D

def return_cell_center(grid,px,py,pz):
    """
    Takes the position of a particle and returns the index of subdomain
    Arguments:
        x,y,z - position of particle
        grid (pyvista object) - grid that contain info about the subdaomains
    Return:
        idx - index of the subdomain that corres
    """
    #Grid parameters
    nx,ny,nz = grid.dimensions
    nx,ny,nz = nx-1,ny-1,nz-1
    #Find the cell where particle is located
    cell = grid.find_closest_cell([px,py,pz])
    X_c = (grid.cell_points(cell)[0][0]+grid.cell_points(cell)[-1][0])/2
    Y_c = (grid.cell_points(cell)[0][1]+grid.cell_points(cell)[-1][1])/2
    Z_c = (grid.cell_points(cell)[0][2]+grid.cell_points(cell)[-1][2])/2
    return X_c,Y_c,Z_c
