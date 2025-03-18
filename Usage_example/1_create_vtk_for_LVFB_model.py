#Present script takes restructured files for the Seyed simulations
#and creates a volume fraction grid that is used for VF method

import pyvista as pv
from Volume_fraction import calculateVolumeFraction

if __name__ == "__main__":  
    
    #Prepare VF model and input for tests
    generate_new_VTK = True
    subdomain_length = 0.5
    h5_file =  "../Test_data/Re_2_phi_20/Particle_data_1.h5"
    
    if generate_new_VTK == True:
        grid = calculateVolumeFraction(h5_file,
                                    subdomain_length,
                                    subdomain_length,
                                    subdomain_length)
    else:
        vtk_file = f"Re_2_phi_20_VOF_processed/dx_{str(subdomain_length)}_dy_{str(subdomain_length)}_dz_{str(subdomain_length)}/VTKVOF/VOF_Particle_data_1.vtk"
        grid = pv.read(vtk_file)

    p1 = pv.Plotter()
    p1.add_mesh(grid,opacity = 0.8)
    p1.show()

            

            

            
