#Present script analises VOF mesh that we constructed for ht eSeyed data and
#creates a correlation plot that compare the DNS results and results of the model
#Use the command "conda activate TF_2_4_py_3_8" in order to activate the correct environment


import json
import pyvista as pv
import pandas as pd
from tqdm import tqdm
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import r2_score
import os

#Supressing all warnings - careful!!!
import warnings
warnings.filterwarnings("ignore")

import sys
from Volume_fraction import getSimParams
from Volume_fraction import getSimParticles
import particle_subdomain

if __name__ == "__main__":

    #Enter input variables
    model_type = "LVFB_L=0.5_W=5"
    model_name = "solar-sweep-245"
    model_loc = "../Models"
    model_dir = model_loc + "/" + model_type + "/" + model_name + "/"
    no_of_neighbours = 5
    out_folder_name = "output"
    vtk_file = "Re_2_phi_20_VOF_processed/dx_0.5_dy_0.5_dz_0.5/VTKVOF/VOF_Particle_data_1.vtk"
    h5_file = "../Test_data/Re_2_phi_20/Particle_data_1.h5"
    csv_name = out_folder_name+"/"+model_type+"_"+model_name+'.csv'

    
    #Load existing data (True) or generate new (False)
    load_res_flag = True

    #Inintialize main input and output data
    params = getSimParams(h5_file)
    particlesDF = getSimParticles(h5_file)

    grid = pv.read(vtk_file)
    y_real = []
    y_predict = []
    
    #Load model
    model = tf.keras.models.load_model(model_dir+"model",compile=False)
    try:
        scalerY = model_dir + "scalerY.pkl"
        scalerY = pickle.load(open(scalerY, "rb"))
    except:
        print("No scalar Y have been found")
        scalerY = None
    
    try:
        scalerX = model_dir + "scalerX.pkl"
        scalerX = pickle.load(open(scalerX, "rb"))
    except:
        print("No scalar X have been found")
        scalerX = None        

    if load_res_flag != True:
        for idx,row in tqdm(particlesDF.iterrows()):
            
            #Preparethe the input vector
            idx_1D = particle_subdomain.return_neighbours_1D(grid,row["xPos"],row["yPos"],row["zPos"],no_of_neighbours)
            input_phi = []
            for i in idx_1D: 
                input_phi.append(grid.cell_data["values"][i])
            
            idx_1D_surrounding = particle_subdomain.return_neighbours_1D(grid,row["xPos"],row["yPos"],row["zPos"],6)

            X_c,Y_c,Z_c = particle_subdomain.return_cell_center(grid,row["xPos"],row["yPos"],row["zPos"])
            inp_glob = np.array([row["xPos"]-X_c,row["yPos"]-Y_c,row["zPos"]-Z_c,"0.5",params["Re"],params["phi"]])
            inp = np.concatenate([inp_glob,input_phi])

            #Transform input array into numpy array
            inp_res = np.array(inp).reshape(-1, ((2*no_of_neighbours+1)**3+6))
            if scalerX is not None: inp_res = scalerX.transform(inp_res)

            #Predict forces and torques
            output = model.predict(inp_res)
            if scalerY is not None: output_res = scalerY.inverse_transform(output)

            #Save values for output array
            y_predict.append(output_res[0][0])
            y_real.append(row["xForce"])

        #Save the comparison between prDNS and model
        data = {"Fx_real":y_real,"Fx_predict":y_predict}     
        result = pd.DataFrame(data=data)
        result.to_csv(csv_name)


    #Load data
    result = pd.read_csv(csv_name,index_col=0)

    #Print data
    plt.figure(figsize=(5,5))
    sns.scatterplot("Fx_real","Fx_predict",data=result)
    plt.plot([result["Fx_real"].min(),result["Fx_real"].max()],[result["Fx_real"].min(),result["Fx_real"].max()])
    r2 = r2_score(result["Fx_real"],result["Fx_predict"])
    plt.title(f"$R^2 = {r2:.2f}$")
    plt.xlabel(r"$F_x^{DNS}$")
    plt.ylabel(r"$F_x^{Pr}$")
    plt.show()

    