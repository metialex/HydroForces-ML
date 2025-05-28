#!/usr/bin/env python3
"""
Benchmark LVFB model prediction performance on CPU or GPU.
"""

import os
import sys

import warnings

warnings.filterwarnings("ignore")
from Volume_fraction import getSimParams
from Volume_fraction import getSimParticles
import particle_subdomain
from tqdm import tqdm


import time
import pickle

import numpy as np
import pyvista as pv
import tensorflow as tf

# Ensure local modules can be imported
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)


def main():
    # Set input variables for model and data paths
    model_type = "LVFB_L=0.5_W=5"
    model_name = "solar-sweep-245"
    model_loc = "../Models"
    model_dir = model_loc + "/" + model_type + "/" + model_name + "/"
    no_of_neighbours = 5
    out_folder_name = "output"
    vtk_file = (
        "Re_2_phi_20_VOF_processed/dx_0.5_dy_0.5_dz_0.5/VTKVOF/VOF_Particle_data_1.vtk"
    )
    h5_file = "../Test_data/Re_2_phi_20/Particle_data_1.h5"
    csv_name = out_folder_name + "/" + model_type + "_" + model_name + ".csv"

    # Load existing data (True) or generate new (False)
    load_res_flag = True

    # Load simulation parameters and particle data
    params = getSimParams(h5_file)
    particlesDF = getSimParticles(h5_file)

    # Load VTK grid data
    grid = pv.read(vtk_file)
    y_real = []
    y_predict = []
    tf.config.optimizer.set_jit(False)
    tf.config.optimizer.set_experimental_options({"disable_meta_optimizer": True})

    # Load trained model
    model = tf.keras.models.load_model(model_dir + "model", compile=False)
    # Warm-up model with random input for initialization
    random_input = np.random.rand(128, 1337).astype(np.float32)
    model.predict(random_input, batch_size=64)

    # Load output scaler if available
    try:
        scalerY = model_dir + "scalerY.pkl"
        scalerY = pickle.load(open(scalerY, "rb"))
    except:
        print("No scalar Y have been found")
        scalerY = None

    # Load input scaler if available
    try:
        scalerX = model_dir + "scalerX.pkl"
        scalerX = pickle.load(open(scalerX, "rb"))
    except:
        print("No scalar X have been found")
        scalerX = None

    input_data = []
    # Iterate over each particle to prepare input features
    for idx, row in tqdm(particlesDF.iterrows()):
        # Find indices of neighboring cells for the particle
        idx_1D = particle_subdomain.return_neighbours_1D(
            grid, row["xPos"], row["yPos"], row["zPos"], no_of_neighbours
        )
        input_phi = []
        for i in idx_1D:
            input_phi.append(grid.cell_data["values"][i])

        idx_1D_surrounding = particle_subdomain.return_neighbours_1D(
            grid, row["xPos"], row["yPos"], row["zPos"], 6
        )

        X_c, Y_c, Z_c = particle_subdomain.return_cell_center(
            grid, row["xPos"], row["yPos"], row["zPos"]
        )
        inp_glob = np.array(
            [
                row["xPos"] - X_c,
                row["yPos"] - Y_c,
                row["zPos"] - Z_c,
                "0.5",
                params["Re"],
                params["phi"],
            ]
        )
        inp = np.concatenate([inp_glob, input_phi])

        # Transform input array into numpy array
        inp_res = np.array(inp).reshape(-1, ((2 * no_of_neighbours + 1) ** 3 + 6))
        if scalerX is not None:
            inp_res = scalerX.transform(inp_res)

        input_data.append(inp_res)

    # Combine all input data into a single numpy array
    input_data = np.concatenate(input_data, axis=0)

    print(input_data.shape)

    # Select device: GPU if available, else CPU
    if tf.config.list_physical_devices("GPU"):
        device = "GPU"
    else:
        device = "CPU"

    print("using device: ", device)

    # Move model and data to the specified device
    if device == "GPU":
        with tf.device("/GPU:0"):
            model = tf.keras.models.load_model(model_dir + "model", compile=False)
            X = tf.convert_to_tensor(input_data, dtype=tf.float32)
    else:
        with tf.device("/CPU:0"):
            model = tf.keras.models.load_model(model_dir + "model", compile=False)
            X = tf.convert_to_tensor(input_data, dtype=tf.float32)

    # Disable JIT compilation for model call
    model.call = tf.function(model.call, jit_compile=False)
    model.jit_compile = False

    # Test different batch sizes for benchmarking
    batch_size_to_test = [1, 16, 32, 64, 128, 256, 512]

    for batch_size in batch_size_to_test:
        # Artificially increase the input data size to 1000 times the batch size for benchmarking
        N = 100 * batch_size
        N = N if N > X.shape[0] else X.shape[0]
        X_tmp = tf.tile(
            X, [N // X.shape[0], 1]
        )  # Repeat the input data to reach N samples

        start = time.time()
        preds = model.predict(X_tmp, batch_size=batch_size)
        elapsed = time.time() - start

        throughput = X_tmp.shape[0] / elapsed if elapsed > 0 else float("nan")
        print(
            "Batch size:",
            batch_size,
            "N:",
            N,
            "throughput:",
            throughput,
            "elapsed time:",
            elapsed,
        )


if __name__ == "__main__":
    main()
