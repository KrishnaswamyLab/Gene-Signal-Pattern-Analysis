import numpy as np
from tqdm import tqdm
from helper import *

"""
Learn gene embeddings with diffusion wavelets.

dictionary, wavelet_sizes = calculate_wavelet_dictionary(G)
"""

def calculate_wavelet_dictionary(G, J=-1, use_reduced=True):
    P = calculate_diffusion_operator(G)
    N = G.N
    size_of_wavelets_per_scale = []
    
    power = 2
    if J == -1:
        J = int(np.log(N))
   
    I = np.eye(N)
    I = normalize(I)
    wavelets = [I]
    size_of_wavelets_per_scale.append(I.shape[1])
    P_j = np.linalg.matrix_power(P, power)

    if use_reduced:
        Psi_j_tilde = column_subset(I-P_j, epsilon=epsilon)
        
        if Psi_j_tilde.shape[1]: 
            print("Wavelets calculated; J = 1")
            return np.array(wavelets)

        Psi_j_tilde = normalize(Psi_j_tilde)
        size_of_wavelets_per_scale.append(Psi_j_tilde.shape[1])
        wavelets += [Psi_j_tilde]

        for i in tqdm(range(2,J)):
            P_j_new = np.linalg.matrix_power(P_j,power)
            Psi_j = P_j - P_j_new
            P_j = P_j_new
            Psi_j_tilde = column_subset(Psi_j,1e-3)
            if Psi_j_tilde.shape[1]: 
                print(f"Wavelets calculated; J = {i}")
                return np.array(wavelets)
            
            Psi_j_tilde = normalize(Psi_j_tilde)

            size_of_wavelets_per_scale.append(Psi_j_tilde.shape[1])
            wavelets += [Psi_j_tilde]
    else:
        print(f"Calculating Wavelets J = {J}")
        wavelets += [I-P_j]
        size_of_wavelets_per_scale.append((I-P_j).shape[1])
        for i in tqdm(range(2,J)):
            P_j_new = np.linalg.matrix_power(P_j,power)
            Psi_j = P_j - P_j_new
            P_j = P_j_new
            Psi_j = normalize(Psi_j)
            size_of_wavelets_per_scale.append(Psi_j.shape[1])
            wavelets += [Psi_j]
            
    return np.array(wavelets).flatten(), np.array(size_of_wavelets_per_scale)
