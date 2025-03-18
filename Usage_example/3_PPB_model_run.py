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
import math


import sys
sys.path.append("../support_files/")
from Volume_fraction import getSimParams
from Volume_fraction import getSimParticles
import particle_subdomain



def check_part_collision(p1,p2,L):
    dist = []
    sym = [[0,0,0],
           [L,0,0],
           [-L,0,0],
           [0,L,0],
           [0,-L,0],
           [L,L,0],
           [L,-L,0],
           [-L,L,0],
           [-L,-L,0],       
           [0,0,L],
           [L,0,L],
           [-L,0,L],
           [0,L,L],
           [0,-L,L],
           [L,L,L],
           [L,-L,L],
           [-L,L,L],
           [-L,-L,L],  
           [0,0,-L],
           [L,0,-L],
           [-L,0,-L],
           [0,L,-L],
           [0,-L,-L],
           [L,L,-L],
           [L,-L,-L],
           [-L,L,-L],
           [-L,-L,-L]]
    for add1 in sym:
        for add2 in sym:
            dist.append(np.sqrt(((p1[0]+add1[0])-(p2[0]+add2[0]))**2 +
                                ((p1[1]+add1[1])-(p2[1]+add2[1]))**2 + 
                                ((p1[2]+add1[2])-(p2[2]+add2[2]))**2))
    return min(dist)

def calc_relative_distance(x,y,L):
    dist = []
    dist.append([abs(x-y),x-y])
    dist.append([abs((x+L)-y),(x+L)-y])
    dist.append([abs((x-L)-y),(x-L)-y])
    dist.sort()
    return dist[0][1]

def return_list_of_closest(p_idx,particlesDF,L,np=30):
    #Create a list of tuples idx and absolute distance between particles
    res = []
    p1 = [particlesDF.iloc[p_idx]["xPos"], particlesDF.iloc[p_idx]["yPos"],particlesDF.iloc[p_idx]["zPos"]]
    for idx,row in particlesDF.iterrows():
        if idx == p_idx: continue
        p2 = [row["xPos"], row["yPos"],row["zPos"]]
        dist = check_part_collision(p1,p2,L)
        res.append((dist,idx))
        
    res.sort()
    final_res = []
    for i,res_i in enumerate(res):
        if i == np: break
        final_res.append(res_i[1])
    return final_res

        






if __name__ == "__main__":
    #input variables & model related inputs
    model_type = "PPB_R_eff=6.0"
    model_name = "dandy-sweep-331"
    model_loc = "../Models"
    out_folder_name = "output"
    model_dir = model_loc + "/" + model_type + "/" + model_name + "/"
    no_of_np = 40
    h5_file = "../Test_data/Re_2_phi_20/Particle_data_1.h5"
    csv_name = out_folder_name+"/"+model_type+"_"+model_name+'.csv'
    load_res_flag = True
    L_domain = 9
    
    #Inintialize main input and output data
    params = getSimParams(h5_file)
    particlesDF = getSimParticles(h5_file)

    y_real = []
    y_predict = []
    

    #Load model & input/output scalers
    #print()
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

            inp_glob = np.array(["0.5",params["Re"],params["phi"]])
            neig_idx_list = return_list_of_closest(idx,particlesDF,L_domain,np=no_of_np)
            
            inp_part = []
            for neig in neig_idx_list:
                inp_part.append(calc_relative_distance(particlesDF.iloc[idx]["xPos"],particlesDF.iloc[neig]["xPos"],L_domain))
                inp_part.append(calc_relative_distance(particlesDF.iloc[idx]["yPos"],particlesDF.iloc[neig]["yPos"],L_domain))
                inp_part.append(calc_relative_distance(particlesDF.iloc[idx]["zPos"],particlesDF.iloc[neig]["zPos"],L_domain))
            inp_part = np.asarray(inp_part)
            inp = np.concatenate([inp_glob,inp_part])
            
            #Transform input array into numpy array
            inp_res = np.array(inp).reshape(-1, (no_of_np*3+3))
            if scalerX is not None: inp_res = scalerX.transform(inp_res)

            #Predict forces and torques
            output = model.predict(inp_res)

            #if scalerY is not None: output_res = scalerY.inverse_transform(output)

            y_predict.append(output[0][0])
            y_real.append(row["xForce"])


        data = {"Fx_real":y_real,"Fx_predict":y_predict}     
        res = pd.DataFrame(data=data)
        res.to_csv(csv_name)


    res = pd.read_csv(csv_name,index_col=0)

    plt.figure(figsize=(5,5))
    sns.scatterplot("Fx_real","Fx_predict",data=res)
    plt.plot([res["Fx_real"].min(),res["Fx_real"].max()],[res["Fx_real"].min(),res["Fx_real"].max()])
    r2 = r2_score(res["Fx_real"],res["Fx_predict"])
    plt.title(f"$R^2 = $ {r2}")
    plt.xlabel(r"$F_x^{DNS}$")
    plt.ylabel(r"$F_x^{Pr}$")
    plt.show()

    