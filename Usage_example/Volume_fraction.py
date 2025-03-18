from glob import glob
from pathlib import Path
from tqdm import tqdm
from joblib import Parallel, delayed
import multiprocessing
import pyvista as pv
import numpy as np
import psutil, os
import itertools
import os.path
from random import shuffle
import pickle
import math
import copy
import h5py
import pandas as pd
import overlap

def calculateVolumeFraction(rawFile, dx, dy, dz, offset=(0, 0, 0), baseDir="",test=False,params_explicit=None):
    """

    Code to calulate volume fraction

    Args:
        rawFile: path to the simulation output
        dx, dy, dz: subdomain lengths
        offset: the offset for data augmentation
        params_explicit = data of parameters and particles. Used in case of simple tests, like to particles domain

    Returns:
        grid (pyvista object): Contains the grid where VOF is calculated

    """

    # Calculating the folder, file name, output folder names
    folderName = rawFile.split("/")[-2]
    fileName = rawFile.split("/")[-1].split(".")[0]
    outFolder = baseDir + str(folderName) + "_VOF_processed/"
    print(outFolder)

    outVTKPoints = outFolder + "VTKPoints/"
    outFolder1 = outFolder + "dx_" + str(dx) + "_dy_" + str(dy) + "_dz_" + str(dz)

    # Creating parent directories if required
    Path(outFolder1).mkdir(parents=True, exist_ok=True)

    outVTKVOF = outFolder1 + "/VTKVOF" + "/"
    Path(outVTKVOF).mkdir(parents=True, exist_ok=True)
    Path(outVTKPoints).mkdir(parents=True, exist_ok=True)

    # Calulating the output file path
    if offset == (0, 0, 0):
        outPath = outVTKVOF + "VOF_" + fileName + ".vtk"
    else:
        outPath = outVTKVOF + "/offset_" + str('_'.join(map(str, offset))) + "/" + "VOF_" + fileName + ".vtk"

    # Checking if a file is already present in the output location if yes then skipping
    if os.path.isfile(outPath) is True and test is False:
        print("job already finished skipping")
        return

    # Calculating the volume fraction
    grid = volumeFractionCalculationStandard(rawFile, dx, dy, dz, outVTKVOF, offset=offset,params_explicit=params_explicit)
    return grid
    # Saving the particles as a VTK if required uncomment the following line
    # savePointDataAsVTK(rawFile, particlesDF, outVTKPoints)


def volumeFractionCalculationStandard(path, dx, dy, dz, outPath, slack=1.2, offset=(0, 0, 0),params_explicit=None):
    """

    Code to calculate volume fraction in a uniform grid.

    We use pyvista(vtk) to manage the grid although it
    should be faster if we use just numpy

    Args:
        path (str): location for the .h5 file having the particle details
        dx (float), dy (float), dz (float): subdomain size
        outPath (str): location to store the output vtk file
        slack (float):

    Returns:
        grid (pyvista object): Contains the grid where VOF is calculated
        particlesDF (pandas data frame): particle input information
        extent (numpy): has the extents of the grid

    """

    # A function to read simulation parameters
    if params_explicit == None:
        params = getSimParams(path)
        particlesDF = getSimParticles(path)

        #pd.DataFrame(data=[params]).to_csv(outPath+"parameters.csv")
        #particlesDF.to_csv(outPath+"particles.csv")
        
    else:
        params,particlesDF = params_explicit
   
    
    # print(particlesDF)

    # dx = dy = dz = 1.0

    # calculating the number of subdomains
    nx = math.ceil((params["xmax"] - params["xmin"]) / dx)
    ny = math.ceil((params["ymax"] - params["ymin"]) / dy)
    nz = math.ceil((params["zmax"] - params["zmin"]) / dz)

    # Creating a grid of zeros to store the volume fraction values
    gridValues = np.zeros((nx, ny, nz))

    # Setting the grid parameters
    grid = pv.UniformGrid()
    grid.dimensions = [nx + 1, ny + 1, nz + 1]
    grid.origin = (params["xmin"] + offset[0], params["ymin"] + offset[0], params["zmin"] + offset[0])
    grid.spacing = (dx, dy, dz)

    # Creating a scalar values to store the VOF and setting them as zeros
    grid.cell_arrays["values"] = gridValues.flatten(order="F")

    cubeSize = max(dx, dy, dz)

    # Calculating the extents of the domain
    extent = np.abs(
        [grid.bounds[1] - grid.bounds[0], grid.bounds[3] - grid.bounds[2], grid.bounds[5] - grid.bounds[4]])

    # lopping through individual particles and updating the Volume fraction of the cells that particle affects
    for idx in range(particlesDF.values[0:, 0:3].shape[0]):

        # finding the cell closest to the particle and getting its ID. This process is quite slow
        gridIdx = grid.find_closest_cell(particlesDF.values[idx, 0:3])
        cell = grid.extract_cells(gridIdx)

        if cell.n_cells == 0:
            # print("here", particlesDF.values[idx, 0:3], np.array([0, 0, 0]))
            cell, particleTemp, gridIdx = applyPeriodicBoundaryCondition(particlesDF.values[idx, 0:4], np.expand_dims(particlesDF.values[idx, 0:3], axis=0),
                                                  np.array([0, 0, 0]), grid, extent)
            # print(particleTemp, cell.cell_centers().points, gridIdx)
            particlesDF.values[idx, 0:4] = particleTemp

        # Calculating the number of adjacent cells the point might occupy
        searchRadius = int(np.ceil(((particlesDF.values[idx, 3] / cubeSize) * slack) + 1))
        searchRange = list(range(-searchRadius + 1, searchRadius))

        # Creating a neighbour list with the indices of the cells that might intersect with the particle
        neighbourhood = list(itertools.product(searchRange, searchRange, searchRange))

        # getting the cell center of the cell closest to the point
        cellCenterPos = cell.cell_centers().points


        # Iterating though the neighbour cells
        for neighbour in neighbourhood:

            # Calculating the offset of the neighbour and extracting the respective cell (also slow)
            offSet = np.array(neighbour) * [dx, dy, dz]
            gridIdx = grid.find_closest_cell(cellCenterPos + offSet)
            cell = grid.extract_cells(gridIdx)

            # If there is no cell in the respective position this means the cell we are looking for is outside
            # the bounding box and we apply the periodic boundary condition case
            if cell.n_cells == 0:
                # Create a copy of the particle
                particleTemp = copy.deepcopy(particlesDF.values[idx, 0:4])

                cell, particleTemp, gridIdx = applyPeriodicBoundaryCondition(particleTemp, cellCenterPos, offSet, grid, extent)
                # Calculate how much the particle intersects with the cell and add to the cells value
                grid.cell_arrays["values"][gridIdx] = grid.cell_arrays["values"][gridIdx] + \
                                                      calculateOverlap(cell, particleTemp)

            # If the periodic boundary check does not take place calculate the overlapping volume and add to
            # the corresponding cell
            elif checkPotentialIntersection(cell, particlesDF.values[idx, 0:4], cubeSize, slack=slack):

                grid.cell_arrays["values"][gridIdx] = grid.cell_arrays["values"][gridIdx] + \
                                                      calculateOverlap(cell, particlesDF.values[idx, 0:4])

    # divide the cell values by volume of the cell
    grid.cell_arrays["values"] = grid.cell_arrays["values"] / (dx * dy * dz)

    if offset == (0, 0, 0):
        # Save the VOF grid
        grid.save(outPath + "VOF_" + path.split("/")[-1][0:-3] + ".vtk")

    else:

        Path(outPath + "/offset_" + str('_'.join(map(str, offset))) + "/").mkdir(parents=True, exist_ok=True)

        # Save the VOF grid
        grid.save(outPath + "/offset_" + str('_'.join(map(str, offset))) + "/" + "VOF_" + path.split("/")[-1][0:-3] + ".vtk")

    return grid


def getSimParams(path):
    """
    Read the simulation parameters

    Args:
        path (str): path for the h5 file

    Returns:
        dict: returns a dictionary with the simulation parameters

    """
    f = h5py.File(path, 'r')
    keys = list(f.keys())
    keys.remove("fixed")

    params = {}
    for key in keys:
        # print(key)
        params[key] = f[key][0]

    return params


def getSimParticles(path):
    """
    Read the particle data

    Args:
        path (str): path for the h5 file

    Returns:
        pandas data frame: returns a data frame with the particle data

    """

    # Reading the particle positions and converting them to DF
    f = h5py.File(path, 'r')
    pos = f["fixed"]["X"][()]
    pos = pd.DataFrame(pos, columns=['xPos', 'yPos', 'zPos'])

    # Adding particle radius to the DF
    R = f["fixed"]["R"][()]
    R = pd.DataFrame(R, columns=['R'])

    # Adding Re to the DF
    Re = f["Re"][()]
    Re = pd.DataFrame(np.repeat(Re, R.shape[0], axis=0), columns=["Re"])

    # Adding phi to DF
    phi = f["phi"][()]
    phi = pd.DataFrame(np.repeat(phi, R.shape[0], axis=0), columns=["phi"])

    # Adding F to the df
    F = f["fixed"]["F"][()]
    F = pd.DataFrame(F, columns=['xForce', 'yForce', 'zForce'])

    # Adding T to the df
    torque = f["fixed"]["T"][()]
    torque = pd.DataFrame(torque, columns=['xTorque', 'yTorque', 'zTorque'])

    # Merging the multiple dataframes into the required output DF
    outDF = pos.merge(R, left_index=True, right_index=True, how='inner')
    outDF = outDF.merge(Re, left_index=True, right_index=True, how='inner')
    outDF = outDF.merge(phi, left_index=True, right_index=True, how='inner')
    outDF = outDF.merge(F, left_index=True, right_index=True, how='inner')
    outDF = outDF.merge(torque, left_index=True, right_index=True, how='inner')

    return outDF

def applyPeriodicBoundaryCondition(particle, cellCenterPos, offSet, grid, extent):
    """
    Function to apply periodic condition if required

    """
    particleTemp = copy.deepcopy(particle)

    # Calculate the position of the particle after applying periodic boundary condition
    tempPos = (cellCenterPos + offSet)
    condition = tempPos - [grid.bounds[1], grid.bounds[3], grid.bounds[5]] > 0

    tempPos[condition] = tempPos[condition] - extent[condition[0]]
    particleTemp[0:3][condition[0]] = particleTemp[0:3][condition[0]] - extent[condition[0]]

    condition = tempPos - [grid.bounds[0], grid.bounds[2], grid.bounds[4]] < 0
    tempPos[condition] = tempPos[condition] + extent[condition[0]]
    particleTemp[0:3][condition[0]] = particleTemp[0:3][condition[0]] + extent[condition[0]]

    gridIdx = grid.find_closest_cell(tempPos)
    cell = grid.extract_cells(gridIdx)

    return cell, particleTemp, gridIdx

def calculateOverlap(cell, particle):
    """
    Calculate overlap between a particle and a cell

    Args:
        cell (Cell): Cell from pyvista
        particle (list of float): particles position and radius

    Returns:
        float: Intersecting volume

    """

    # verticles of the example subdomain
    vertices = cell.points

    vertices = [vertices[0], vertices[1], vertices[3], vertices[2], vertices[4], vertices[5], vertices[7], vertices[6]]

    # creating a hexahedron
    hexa = overlap.Hexahedron(vertices)

    s = overlap.Sphere((particle[0], particle[1], particle[2]), particle[3])  # creating the particle/sphere

    # Calculating the overlap between the sphere and the hexagon
    volume = overlap.overlap(s, hexa)

    return volume

# Check if a particle intersects a cell (very aproximate test)
def checkPotentialIntersection(cell, particle, cubeSize, slack=1.2):
    x1, y1, z1 = cell.center
    x2, y2, z2 = particle[0:3]

    dist = (((x2 - x1) ** 2) + ((y2 - y1) ** 2) + ((z2 - z1) ** 2)) ** (1 / 2)

    if dist < ((particle[3] + (cubeSize)) * slack):
        # print(dist, ((particle[3] + cubeSize) * slack))
        return True

    return False
