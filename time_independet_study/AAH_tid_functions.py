import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import AAH_model as aah
import AAH_tools as aah_tools
import tqdm

'''
Time-independent functions to study GAAH parameter space.

'''
def EnergyPhiSpectrum(N: int, V1: float, V2: float, phi_lin: np.ndarray, t: float = 1, theta: float = 0.0,
                      data_path: str = "data", filename: str = None, save = False)-> tuple[np.ndarray,np.ndarray]:
    """
    Calculates the energy spectrum as a function of the phason angle phi.

    Parameters:
    - N (int): Number of lattice sites in the chain.
    - V1 (float): Amplitude of the on-site potential.
    - V2 (float): Amplitude of the opping potential.
    - phi_lin (np.ndarray): Array of phason angles (phi) to iterate over.
    - t (float): Hopping between neighboring sites (default is 1).
    - theta (float): Phase offset parameter (default is 0.0).
    - data_path (str): Folder directory name for saving data (default is "data").
    - filename (str): Custom file name for the saved output (optional).
    - save (bool): Flag to determine whether to save the output data to a file (default is False).
    """
    energies = []
    angles = []

    for p in phi_lin:
        AAH = aah.AAH_chain(N=N, t=t, V1=V1, V2=V2, theta=theta, phi=p)

        eigvals, eigvecs = AAH.diagonalize()
        energies.append(eigvals)
        angles.append(AAH.phi / np.pi)

    if save:
        if filename is None:
            filename = f"EnergyPhiSpectrum_N{N}_V1{V1}_V2{V2}_theta{theta}.npz"

        dir_name = data_path
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)

        full_path = os.path.join(dir_name, filename) if dir_name else filename

        np.savez(
            full_path,
            energies=np.array(energies),
            angles=np.array(angles),
            N=N, V1=V1, V2=V2, theta=theta,
        )

        print(f"Saved data to {full_path}")

    return np.array(energies), np.array(angles)


def G_Phi(energies: np.ndarray, angles: np.ndarray, level_i: int, level_j: int,
    N: int, V1: float, V2: float, theta: float = 0.0, t: float = 1,
    data_path: str = "data", filename: str = None, save = False)->tuple[np.ndarray,np.ndarray]:
    """
    Calculates the energy gap between two energy levels across phason angles spectrum.

    Parameters:
    - energies (np.ndarray): Array of energy eigenvalues across phason angles.
    - angles (np.ndarray): Array of the phason angles.
    - level_i (int): Index of the first energy level.
    - level_j (int): Index of the second energy level.
    - N (int): Number of lattice sites in the chain.
    - V1 (float): Amplitude of the on-site potential.
    - V2 (float): Amplitude of the opping potential.
    - theta (float): Phase offset parameter (default is 0.0).
    - t (float): Hopping between neighboring sites  (default is 1).
    - data_path (str): Folder directory name for saving data (default is "data").
    - filename (str): Custom file name for the saved output (optional).
    - save (bool): Flag to determine whether to save the output data to a file (default is False).
    """
    G = []

    for i in range(len(angles)):
        E1 = energies[i][level_i]
        E2 = energies[i][level_j]
        G.append(np.abs(E2 - E1)/t)

    if save:
        if filename is None:
            filename = f"GPhiSpectrum_N{N}_V1{V1}_V2{V2}_theta{theta}.npz"

        dir_name = data_path
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)

        full_path = os.path.join(dir_name, filename) if dir_name else filename

        np.savez(
            full_path,
            G=np.array(G),
            angles=np.array(angles),
            N=N, V1=V1, V2=V2, theta=theta,
        )

        print(f"Saved data to {full_path}")

    return np.array(G), np.array(angles)


def dIPR_Phi(N: int, V1: float, V2: float, phi_lin: np.ndarray, level_i: int, t: float = 1, theta: float = 0.0,
             dparameter: float = 0.3, data_path: str = "data", filename: str = None, save=False) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculates the directional Inverse Participation Ratio (dIPR) for a specific energy level across phason angle spectrum.

    Parameters:
    - N (int): Number of lattice sites in the chain.
    - V1 (float): Amplitude of the on-site potential.
    - V2 (float): Amplitude of the opping potential.
    - phi_lin (np.ndarray): Array of phason angles (phi) to iterate over.
    - level_i (int): Index of the eigenstate/energy level.
    - t (float): Hopping between neighboring sites (default is 1).
    - theta (float): Phase offset parameter (default is 0.0).
    - dparameter (float): Parameter for the dIPR calculation (default is 0.3).
    - data_path (str): Folder directory name for saving data (default is "data").
    - filename (str): Custom file name for the saved output (optional).
    - save (bool): Flag to determine whether to save the output data to a file (default is False).
    """

    dIPR_list = []
    angles = []

    for i in range(len(phi_lin)):
        AAH = aah.AAH_chain(N=N, t=t, V1=V1, V2=V2, theta=theta, phi=phi_lin[i])
        AAH.diagonalize()
        dipr = AAH.dIPR(dparameter=dparameter)  # shape (N,), one value per eigenstate[cite: 1]
        dIPR_list.append(dipr[level_i])
        angles.append(AAH.phi / np.pi)

    if save:
        if filename is None:
            filename = f"dIPRPhiSpectrum_N{N}_V1{V1}_V2{V2}_theta{theta}.npz"

        dir_name = data_path
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)

        full_path = os.path.join(dir_name, filename) if dir_name else filename

        np.savez(
            full_path,
            dIPR=np.array(dIPR_list),
            angles=np.array(angles),
            N=N, V1=V1, V2=V2, theta=theta, level_i=level_i,
        )

        print(f"Saved data to {full_path}")

    return np.array(dIPR_list), np.array(angles)


def ProbabilityDensityPhiSpectrum(N, V1, V2, phi_lin, level_i, t=1.0, theta=0.0,
                                  data_path="data", filename=None, save=True):
    """
    Calculates the probability density distribution across sites for a specific level over phason angle spectrum.

    Parameters:
    - N (int): Number of lattice sites in the chain.
    - V1 (float): Amplitude of the on-site potential.
    - V2 (float): Amplitude of the opping potential.
    - phi_lin (np.ndarray): Array of phason angles (phi) to iterate over.
    - level_i (int): Index of the eigenstate/energy level.
    - t (float): Hopping between neighboring sites (default is 1).
    - theta (float): Phase offset parameter (default is 0.0).
    - data_path (str): Folder directory name for saving data (default is "data").
    - filename (str): Custom file name for the saved output (optional).
    - save (bool): Flag to determine whether to save the output data to a file (default is False).
    """
    prob_density_list = []
    angles = []

    for p in phi_lin:
        AAH = aah.AAH_chain(N=N, t=t, V1=V1, V2=V2, theta=theta, phi=p)
        AAH.diagonalize()
        prob = AAH.probability_density()  # shape (num_states, N)
        prob_density_list.append(prob[level_i])
        angles.append(AAH.phi / np.pi)

    prob_density_arr = np.array(prob_density_list)  # shape (len(phi_lin), N)
    angles = np.array(angles)

    if save:
        if filename is None:
            filename = f"WavefunctionPhiSpectrum_N{N}_V1{V1}_V2{V2}_theta{theta}.npz"

        dir_name = data_path
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)

        full_path = os.path.join(dir_name, filename) if dir_name else filename

        np.savez(
            full_path,
            prob_density=prob_density_arr,
            angles=angles,
            phi_lin=phi_lin,
            N=N, V1=V1, V2=V2, theta=theta, level_i=level_i,
        )

        print(f"Saved data to {full_path}")

    return prob_density_arr, angles


def GPhiHeatmapData(N, phi_lin, level_i, level_j, V1=None, V2=None, V=None,
                     t=1.0, theta=0.0, data_path='data', filename=None, save=True):
    """
    Generates heatmap data for energy gaps (G) as a function of phason angle and potential V1/V2.

    Parameters:
    - N (int): Number of lattice sites in the chain.
    - phi_lin (np.ndarray): Array of phason angles (phi) to iterate over.
    - level_i (int): Index of the first energy level.
    - level_j (int): Index of the second energy level.
    - V1 (float or np.ndarray): Amplitude(s) of the on-site potential.
    - V2 (float or np.ndarray): Amplitude(s) of the hopping potential.
    - V (float or np.ndarray): Unified potential array when V1=V2.
    - t (float): Hopping term (default is 1.0).
    - theta (float): Phase offset parameter (default is 0.0).
    - data_path (str): Folder directory name for saving data (default is "data").
    - filename (str): Custom file name for the saved output (optional).
    - save (bool): Flag to determine whether to save the output data to a file (default is False).
    """
    if V is not None:
        V_arr = np.atleast_1d(np.asarray(V, dtype=float))
        if V_arr.size < 2:
            raise ValueError("V must be array-like with more than one value.")
        V1_list = V_arr
        V2_list = V_arr
        case = "III"
    else:
        V1_arr = np.atleast_1d(np.asarray(V1, dtype=float))
        V2_arr = np.atleast_1d(np.asarray(V2, dtype=float))

        if V1_arr.size > 1 and V2_arr.size > 1:
            raise ValueError("To sweep V1 = V2 together, pass a single array via V instead of V1/V2.")
        elif V1_arr.size > 1:
            V1_list = V1_arr
            V2_list = np.full_like(V1_arr, V2_arr[0])
            case = "I"
        elif V2_arr.size > 1:
            V2_list = V2_arr
            V1_list = np.full_like(V2_arr, V1_arr[0])
            case = "II"
        else:
            raise ValueError("At least one of V1, V2 must be array-like to sweep over.")

    n_rows = len(V1_list)
    G_matrix = np.zeros((n_rows, len(phi_lin)))

    for row, (v1, v2) in enumerate(
        tqdm.tqdm(list(zip(V1_list, V2_list)), desc="G(phi) heatmap", unit="row")
    ):
        energies, angles = EnergyPhiSpectrum(
            N=N, V1=v1, V2=v2, phi_lin=phi_lin, t=t, theta=theta, save=False,
        )
        G, _ = G_Phi(
            energies=energies, angles=angles, level_i=level_i, level_j=level_j,
            N=N, V1=v1, V2=v2, theta=theta, t=t, save=False,
        )
        G_matrix[row] = G

    if save:
        if filename is None:
            if case == "I":
                tag = f"V2{V2_list[0]:.1f}"
            elif case == "II":
                tag = f"V1{V1_list[0]:.1f}"
            else:
                tag = "V1V2"
            filename = f"GPhiHeatmap_N{N}_{tag}_theta{theta}.npz"

        dir_name = data_path
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)

        full_path = os.path.join(dir_name, filename) if dir_name else filename

        np.savez(
            full_path,
            G_matrix=G_matrix,
            V1_list=V1_list,
            V2_list=V2_list,
            phi_lin=phi_lin,
            N=N, theta=theta, level_i=level_i, level_j=level_j, case=case,
        )

        print(f"Saved data to {full_path}")

    return G_matrix

def IPRPhiHeatmapData(N, phi_lin, level_i, V1=None, V2=None, V=None,
                       t=1.0, theta=0.0, dparameter=0.3,
                       data_path='data', filename=None, save=True):
    """
    Generates heatmap data for IPR/dIPR values as a function of phason angle and potential strength(s).

    Parameters:
    - N (int): Number of lattice sites in the chain.
    - phi_lin (np.ndarray): Array of phason angles (phi) to iterate over.
    - level_i (int): Index of the first energy level.
    - V1 (float or np.ndarray): Amplitude(s) of the on-site potential.
    - V2 (float or np.ndarray): Amplitude(s) of the hopping potential.
    - V (float or np.ndarray): Unified potential array when V1=V2.
    - t (float): Hopping term (default is 1.0).
    - theta (float): Phase offset parameter (default is 0.0).
    - dparameter (float): Parameter for the dIPR calculation (default is 0.3).
    - data_path (str): Folder directory name for saving data (default is "data").
    - filename (str): Custom file name for the saved output (optional).
    - save (bool): Flag to determine whether to save the output data to a file (default is False).
    """
    if V is not None:
        V_arr = np.atleast_1d(np.asarray(V, dtype=float))
        if V_arr.size < 2:
            raise ValueError("V must be array-like with more than one value.")
        V1_list = V_arr
        V2_list = V_arr
        case = "III"
    else:
        V1_arr = np.atleast_1d(np.asarray(V1, dtype=float))
        V2_arr = np.atleast_1d(np.asarray(V2, dtype=float))

        if V1_arr.size > 1 and V2_arr.size > 1:
            raise ValueError("To sweep V1 = V2 together, pass a single array via V instead of V1/V2.")
        elif V1_arr.size > 1:
            V1_list = V1_arr
            V2_list = np.full_like(V1_arr, V2_arr[0])
            case = "I"
        elif V2_arr.size > 1:
            V2_list = V2_arr
            V1_list = np.full_like(V2_arr, V1_arr[0])
            case = "II"
        else:
            raise ValueError("At least one of V1, V2 must be array-like to sweep over.")

    n_rows = len(V1_list)
    IPR_matrix = np.zeros((n_rows, len(phi_lin)))

    for row, (v1, v2) in enumerate(
        tqdm.tqdm(list(zip(V1_list, V2_list)), desc="IPR(phi) heatmap", unit="row")
    ):
        dipr, _ = dIPR_Phi(
            N=N, V1=v1, V2=v2, phi_lin=phi_lin, level_i=level_i, t=t, theta=theta,
            dparameter=dparameter, save=False,
        )
        IPR_matrix[row] = dipr

    if save:
        if filename is None:
            if case == "I":
                tag = f"V2{V2_list[0]:.1f}"
            elif case == "II":
                tag = f"V1{V1_list[0]:.1f}"
            else:
                tag = "V1V2"
            filename = f"IPRPhiHeatmap_N{N}_{tag}_theta{theta}.npz"

        dir_name = data_path
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)

        full_path = os.path.join(dir_name, filename) if dir_name else filename

        np.savez(
            full_path,
            IPR_matrix=IPR_matrix,
            V1_list=V1_list,
            V2_list=V2_list,
            phi_lin=phi_lin,
            N=N, theta=theta, level_i=level_i, dparameter=dparameter, case=case,
        )

        print(f"Saved data to {full_path}")

    return IPR_matrix


def compute_theta_phi_heatmap(N, V1, V2, lvl_i, lvl_j, t=1, N_THETA=200, N_PHI=200, DPARAMETER=0.3):

    phi_lin = np.linspace(0, 2 * np.pi, N_PHI)
    theta_lin = np.linspace(0, 2 * np.pi, N_THETA)

    G_matrix = np.zeros((N_THETA, N_PHI))
    dIPR_matrix = np.zeros((N_THETA, N_PHI))

    for row, theta in enumerate(
            tqdm.tqdm(theta_lin, desc=f"theta sweep (V1={V1}, V2={V2})", unit="row")
    ):
        energies, angles = EnergyPhiSpectrum(
            N=N, V1=V1, V2=V2, phi_lin=phi_lin, t=t, theta=theta, save=False,
        )

        G, _ = G_Phi(
            energies=energies, angles=angles,
            level_i=lvl_i, level_j=lvl_j,
            N=N, V1=V1, V2=V2, theta=theta, t=t, save=False,
        )
        G_matrix[row] = G

        dipr, _ = dIPR_Phi(
            N=N, V1=V1, V2=V2, phi_lin=phi_lin, level_i=lvl_j,
            t=t, theta=theta, dparameter=DPARAMETER, save=False,
        )
        dIPR_matrix[row] = dipr

    return G_matrix, dIPR_matrix