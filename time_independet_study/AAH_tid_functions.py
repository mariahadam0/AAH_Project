import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import AAH_model as aah
import AAH_tools as aah_tools
import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.optimize import minimize_scalar
from scipy.signal import find_peaks

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


# ======================================================================
#  G_min(V1, V2) and phi_0(V1, V2) maps
# ======================================================================
def _gap_at_phi(phi: float, N: int, V1: float, V2: float, theta: float,
                t: float, level_i: int) -> float:
    """|E[level_i+1] - E[level_i]| / t at a single phason angle phi (eigenvalues only)."""
    H = aah.AAH_chain(N=N, t=t, V1=V1, V2=V2, theta=theta, phi=phi).Hamiltonian()
    E = np.linalg.eigvalsh(H)
    return abs(E[level_i + 1] - E[level_i]) / t


def GminPhi0(N: int, V1: float, V2: float, level_i: int, theta: float = 0.0,
             t: float = 1.0, n_phi: int = 360, flat_tol: float = 1e-9) -> tuple[float, float]:
    """
    Minimal gap G_min between levels level_i and level_i+1 over the phason angle phi,
    and the phason angle phi_0 at which it occurs.

    A coarse scan over n_phi values of phi in [0, 2pi] locates the minimum, which is
    then refined with a bounded 1D minimization inside the two neighbouring grid cells
    (the Hamiltonian is 2pi-periodic in phi, so the search wraps around the edges).

    Parameters:
    - N (int): Number of lattice sites in the chain.
    - V1 (float): Amplitude of the on-site potential.
    - V2 (float): Amplitude of the hopping potential.
    - level_i (int): Index of the lower level; the gap is taken to level_i + 1.
    - theta (float): Phason offset (default is 0.0).
    - t (float): Hopping between neighboring sites (default is 1).
    - n_phi (int): Number of points of the coarse phi scan.
    - flat_tol (float): If G varies by less than this over phi (e.g. V1 = V2 = 0),
                        phi_0 is undefined and returned as NaN.

    Returns:
    - G_min (float), phi_0 (float, in units of pi, in [0, 2))
    """
    phi_lin = np.linspace(0, 2 * np.pi, n_phi)
    G = np.array([_gap_at_phi(p, N, V1, V2, theta, t, level_i) for p in phi_lin])

    k = int(np.argmin(G))
    dphi = phi_lin[1] - phi_lin[0]
    res = minimize_scalar(_gap_at_phi, args=(N, V1, V2, theta, t, level_i),
                          bounds=(phi_lin[k] - dphi, phi_lin[k] + dphi),
                          method="bounded", options={"xatol": 1e-8})

    if res.fun < G[k]:
        g_min, phi0 = float(res.fun), float(res.x)
    else:
        g_min, phi0 = float(G[k]), float(phi_lin[k])

    if np.ptp(G) < flat_tol:
        return g_min, np.nan
    return g_min, (phi0 % (2 * np.pi)) / np.pi


def _gmin_row(args):
    """Worker: one row (fixed V2, all V1) of the G_min / phi_0 maps."""
    row, V2, V1_lin, N, level_i, theta, t, n_phi = args
    g = np.zeros(len(V1_lin))
    p = np.zeros(len(V1_lin))
    for col, V1 in enumerate(V1_lin):
        g[col], p[col] = GminPhi0(N=N, V1=V1, V2=V2, level_i=level_i,
                                  theta=theta, t=t, n_phi=n_phi)
    return row, g, p


def GminMapFilename(N: int, theta: float, level_i: int) -> str:
    return f"GminPhi0Map_N{N}_theta{theta}_lvl{level_i}.npz"


def GminPhi0Map(N: int, level_i: int, V1_lin: np.ndarray, V2_lin: np.ndarray,
                theta: float = 0.0, t: float = 1.0, n_phi: int = 360,
                n_workers: int = None, data_path: str = "data",
                filename: str = None, save: bool = True
                ) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes G_min and phi_0 on a (V1, V2) grid (see GminPhi0).

    Matrices have shape (len(V2_lin), len(V1_lin)): rows = V2, columns = V1.
    phi_0 is stored in units of pi.

    Parameters:
    - N (int): Number of lattice sites in the chain.
    - level_i (int): Index of the lower level; the gap is taken to level_i + 1.
    - V1_lin, V2_lin (np.ndarray): Grids of V1 and V2 values.
    - theta (float): Phason offset (default is 0.0).
    - t (float): Hopping between neighboring sites (default is 1).
    - n_phi (int): Number of points of the coarse phi scan per (V1, V2) point.
    - n_workers (int): Number of processes (default: all CPUs; 1 = serial).
    - data_path (str): Folder directory name for saving data (default is "data").
    - filename (str): Custom file name for the saved output (optional).
    - save (bool): Whether to save the output data to a .npz file.
    """
    V1_lin = np.asarray(V1_lin, dtype=float)
    V2_lin = np.asarray(V2_lin, dtype=float)
    Gmin_matrix = np.zeros((len(V2_lin), len(V1_lin)))
    phi0_matrix = np.zeros_like(Gmin_matrix)

    jobs = [(r, V2, V1_lin, N, level_i, theta, t, n_phi) for r, V2 in enumerate(V2_lin)]

    if n_workers is None:
        n_workers = os.cpu_count() or 1

    if n_workers == 1:
        for job in tqdm.tqdm(jobs, desc="G_min / phi_0 map", unit="row"):
            r, g, p = _gmin_row(job)
            Gmin_matrix[r], phi0_matrix[r] = g, p
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futures = [ex.submit(_gmin_row, job) for job in jobs]
            for f in tqdm.tqdm(as_completed(futures), total=len(futures),
                               desc="G_min / phi_0 map", unit="row"):
                r, g, p = f.result()
                Gmin_matrix[r], phi0_matrix[r] = g, p

    if save:
        if filename is None:
            filename = GminMapFilename(N, theta, level_i)
        if data_path:
            os.makedirs(data_path, exist_ok=True)
        full_path = os.path.join(data_path, filename) if data_path else filename

        np.savez(
            full_path,
            Gmin_matrix=Gmin_matrix, phi0_matrix=phi0_matrix,
            V1_lin=V1_lin, V2_lin=V2_lin,
            N=N, theta=theta, t=t, level_i=level_i, n_phi=n_phi,
        )
        print(f"Saved data to {full_path}")

    return Gmin_matrix, phi0_matrix


# ======================================================================
#  Number of local minima of G(phi) / maxima of E_lvl_i(phi) maps
# ======================================================================
def LevelsVsPhi(N: int, V1: float, V2: float, level_i: int, phi_lin: np.ndarray,
                theta: float = 0.0, t: float = 1.0, chunk: int = 64
                ) -> tuple[np.ndarray, np.ndarray]:
    """
    Fast (vectorized) E[level_i](phi) and E[level_i+1](phi) for all phi in phi_lin.
    Builds exactly the same Hamiltonian as AAH_chain.Hamiltonian(), but diagonalizes
    many phi at once and computes eigenvalues only.

    Returns:
    - E_i, E_next: arrays of length len(phi_lin)
    """
    Q = aah.tau
    phi_lin = np.asarray(phi_lin, dtype=float)
    j = np.arange(N)
    k = np.arange(N - 1)
    E_i = np.empty(len(phi_lin))
    E_next = np.empty(len(phi_lin))

    for s in range(0, len(phi_lin), chunk):
        phi = phi_lin[s:s + chunk, None]
        H = np.zeros((phi.shape[0], N, N))
        H[:, j, j] = V1 * np.cos((j + 1) * Q + phi + theta)
        hop = t + V2 * np.cos((k + 1 + 0.5) * Q + phi)
        H[:, k, k + 1] = hop
        H[:, k + 1, k] = hop
        E = np.linalg.eigvalsh(H)
        E_i[s:s + chunk] = E[:, level_i]
        E_next[s:s + chunk] = E[:, level_i + 1]

    return E_i, E_next


def _count_prominent(y: np.ndarray, prom_frac: float, flat_tol: float = 1e-9) -> int:
    """Number of maxima of the periodic signal y with prominence > prom_frac * (max - min)."""
    n = len(y)
    rng = np.ptp(y)
    if rng < flat_tol:
        return 0
    peaks, _ = find_peaks(np.tile(y, 3), prominence=prom_frac * rng)   # 3 periods: handles wrap-around
    return int(np.sum((peaks >= n) & (peaks < 2 * n)))


def CountPoint(N: int, V1: float, V2: float, level_i: int, theta: float = 0.0,
               t: float = 1.0, n_phi: int = 360, prom_frac: float = 0.05
               ) -> tuple[int, int]:
    """
    For one (V1, V2): number of significant local minima of the gap
    G(phi) = E[level_i+1] - E[level_i], and of significant maxima of E[level_i](phi),
    over the periodic phason angle phi in [0, 2pi). "Significant" = prominence larger
    than prom_frac times the range of the function (ignores tiny wiggles).

    Returns:
    - (n_minima_of_G, n_maxima_of_E)
    """
    phi_lin = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    E_i, E_next = LevelsVsPhi(N, V1, V2, level_i, phi_lin, theta=theta, t=t)
    return (_count_prominent(-(E_next - E_i), prom_frac),
            _count_prominent(E_i, prom_frac))


def _count_row(args):
    """Worker: one row (fixed V2, all V1) of the count maps."""
    row, V2, V1_lin, N, level_i, theta, t, n_phi, prom_frac = args
    nG = np.zeros(len(V1_lin), dtype=int)
    nE = np.zeros(len(V1_lin), dtype=int)
    for col, V1 in enumerate(V1_lin):
        nG[col], nE[col] = CountPoint(N, V1, V2, level_i, theta, t, n_phi, prom_frac)
    return row, nG, nE


def CountMapFilename(N: int, theta: float, level_i: int) -> str:
    return f"ExtremaCountMap_N{N}_theta{theta}_lvl{level_i}.npz"


# ======================================================================
#  Number of gap closings between level_i and level_i+1
# ======================================================================
def CountGapClosings(N: int, V1: float, V2: float, level_i: int, theta: float = 0.0,
                     t: float = 1.0, n_phi: int = 360, gap_tol: float = 1e-3) -> int:
    """
    Number of phason angles phi in [0, 2pi) at which the gap
    G(phi) = (E[level_i+1] - E[level_i]) / t closes, i.e. G has a local minimum
    with G < gap_tol.

    1. G is sampled on a periodic grid of n_phi points; every sampled local minimum
       is a candidate.
    2. Candidates whose sampled value is too large to hide a closing are discarded
       (a closing is a cusp, so the sampled value can exceed the true minimum by at most
       about (V1 + 2*V2) * dphi, since |dE/dphi| <= ||dH/dphi|| <= |V1| + 2|V2|).
    3. Each remaining candidate is refined with a bounded 1D minimization of G in the
       two neighbouring grid cells, and counted if the refined minimum is < gap_tol.
    4. Candidates that converge to the same phi are counted once.
    """
    phi_lin = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    dphi = phi_lin[1] - phi_lin[0]
    E_i, E_next = LevelsVsPhi(N, V1, V2, level_i, phi_lin, theta=theta, t=t)
    G = (E_next - E_i) / t

    cand = np.flatnonzero((G < np.roll(G, 1)) & (G <= np.roll(G, -1)))
    cand = cand[G[cand] < gap_tol + (abs(V1) + 2 * abs(V2)) * dphi / t]

    closings = []
    for k in cand:
        res = minimize_scalar(_gap_at_phi, args=(N, V1, V2, theta, t, level_i),
                              bounds=(phi_lin[k] - dphi, phi_lin[k] + dphi),
                              method="bounded", options={"xatol": 1e-9})
        g_ref, phi_ref = min((res.fun, res.x), (G[k], phi_lin[k]))
        if g_ref < gap_tol:
            closings.append(phi_ref % (2 * np.pi))

    # merge duplicates (periodic distance)
    closings.sort()
    merged = []
    for c in closings:
        if not merged or abs(c - merged[-1]) > 1e-4:
            merged.append(c)
    if len(merged) > 1 and abs(merged[0] + 2 * np.pi - merged[-1]) <= 1e-4:
        merged.pop()
    return len(merged)


def _closing_row(args):
    """Worker: one row (fixed V2, all V1) of the gap-closing count map."""
    row, V2, V1_lin, N, level_i, theta, t, n_phi, gap_tol = args
    n = np.zeros(len(V1_lin), dtype=int)
    for col, V1 in enumerate(V1_lin):
        n[col] = CountGapClosings(N, V1, V2, level_i, theta, t, n_phi, gap_tol)
    return row, n


def ClosingMapFilename(N: int, theta: float, level_i: int) -> str:
    return f"GapClosingCountMap_N{N}_theta{theta}_lvl{level_i}.npz"


# ======================================================================
#  (theta, phi) maps of the gap G and of the dIPR
# ======================================================================
class Cancelled(Exception):
    """Raised by the map calculations when the cancel event is set (e.g. by a GUI)."""


def _theta_row(args):
    """
    Worker: one row (fixed theta, all phi) of the G and dIPR maps.
    One diagonalization per (theta, phi) point gives both quantities.
    """
    row, theta, phi_lin, N, V1, V2, t, level_i, level_j, dparameter = args
    G = np.zeros(len(phi_lin))
    D = np.zeros(len(phi_lin))
    for col, phi in enumerate(phi_lin):
        chain = aah.AAH_chain(N=N, t=t, V1=V1, V2=V2, theta=theta, phi=phi)
        E, _ = chain.diagonalize()
        G[col] = abs(E[level_j] - E[level_i]) / t
        D[col] = chain.dIPR(dparameter=dparameter)[level_j]
    return row, G, D


def ThetaPhiMapFilename(N: int, V1: float, V2: float, level_i: int, level_j: int,
                        t: float = 1.0, dparameter: float = 0.3,
                        n_theta: int = 200, n_phi: int = 200) -> str:
    """File name used by ThetaPhiMap; it encodes every parameter of the calculation."""
    return (f"G_dIPR_ThetaPhi_N{N}_V1{V1:g}_V2{V2:g}_t{t:g}_lvl{level_i}-{level_j}"
            f"_d{dparameter:g}_{n_theta}x{n_phi}.npz")


def LoadThetaPhiMap(path: str, N: int, V1: float, V2: float, level_i: int, level_j: int,
                    t: float = 1.0, dparameter: float = 0.3,
                    n_theta: int = 200, n_phi: int = 200):
    """
    Loads a saved (theta, phi) map and checks that it was calculated for exactly the
    given parameters.

    Returns:
    - dict(G_matrix, dIPR_matrix, phi_lin, theta_lin) if the file exists and matches,
      otherwise None (missing, unreadable or for different parameters).
    """
    if not os.path.exists(path):
        return None
    try:
        d = np.load(path)
        if (d["G_matrix"].shape == (n_theta, n_phi)
                and d["dIPR_matrix"].shape == (n_theta, n_phi)
                and int(d["N"]) == N and float(d["V1"]) == V1 and float(d["V2"]) == V2
                and float(d["t"]) == t and int(d["level_i"]) == level_i
                and int(d["level_j"]) == level_j and float(d["dparameter"]) == dparameter):
            return dict(G_matrix=d["G_matrix"], dIPR_matrix=d["dIPR_matrix"],
                        phi_lin=d["phi_lin"], theta_lin=d["theta_lin"])
    except Exception:
        pass
    return None


def ThetaPhiMap(N: int, V1: float, V2: float, level_i: int, level_j: int,
                t: float = 1.0, dparameter: float = 0.3,
                n_phi: int = 200, n_theta: int = 200,
                n_workers: int = None, data_path: str = "data", filename: str = None,
                save: bool = True, force: bool = False,
                progress_cb=None, cancel_evt=None, mp_context=None) -> dict:
    """
    G(theta, phi) = |E[level_j] - E[level_i]| / t and dIPR(theta, phi) of the state
    level_j, on a grid of the phason offset theta and the phason angle phi, both in
    [0, 2pi] (end points included).

    Matrices have shape (n_theta, n_phi): rows = theta, columns = phi. The calculation is
    parallelised over rows of theta. With save=True the result is written to
    data_path/filename (same keys as the files of the data notebook), and if a file made
    for exactly the same parameters already exists it is loaded instead of recomputed
    (unless force=True).

    Parameters:
    - N (int): Number of lattice sites in the chain.
    - V1, V2 (float): Amplitudes of the on-site / hopping modulation.
    - level_i, level_j (int): Levels whose gap is G; the dIPR is taken for level_j.
    - t (float): Hopping between neighboring sites (default is 1).
    - dparameter (float): Parameter for the dIPR calculation (default is 0.3).
    - n_phi, n_theta (int): Number of grid points along phi and theta.
    - n_workers (int): Number of processes (default: all CPUs; 1 = serial).
    - data_path (str): Folder directory name for saving data (default is "data").
    - filename (str): Custom file name (default: ThetaPhiMapFilename(...)).
    - save (bool): Whether to save the result (and to look for a saved one).
    - force (bool): Recalculate even if a matching saved file exists.
    - progress_cb (callable): Called as progress_cb(done_rows, total_rows) after each row.
    - cancel_evt (threading.Event): If set during the calculation, Cancelled is raised.
    - mp_context: multiprocessing context for the worker processes (e.g.
                  multiprocessing.get_context("spawn"), safer in a process with Tk loaded).

    Returns:
    - dict(G_matrix, dIPR_matrix, phi_lin, theta_lin, cached) where cached is True if the
      result was loaded from a file.
    """
    if filename is None:
        filename = ThetaPhiMapFilename(N, V1, V2, level_i, level_j, t, dparameter,
                                       n_theta, n_phi)
    full_path = os.path.join(data_path, filename) if data_path else filename

    if save and not force:
        loaded = LoadThetaPhiMap(full_path, N, V1, V2, level_i, level_j, t, dparameter,
                                 n_theta, n_phi)
        if loaded is not None:
            loaded["cached"] = True
            return loaded

    phi_lin = np.linspace(0, 2 * np.pi, n_phi)
    theta_lin = np.linspace(0, 2 * np.pi, n_theta)
    G_matrix = np.zeros((n_theta, n_phi))
    dIPR_matrix = np.zeros_like(G_matrix)
    jobs = [(r, th, phi_lin, N, V1, V2, t, level_i, level_j, dparameter)
            for r, th in enumerate(theta_lin)]

    if n_workers is None:
        n_workers = os.cpu_count() or 1

    def _check_cancel():
        if cancel_evt is not None and cancel_evt.is_set():
            raise Cancelled()

    done = 0
    if n_workers == 1:
        for job in tqdm.tqdm(jobs, desc="theta-phi map", unit="row", disable=progress_cb is not None):
            _check_cancel()
            r, g, d = _theta_row(job)
            G_matrix[r], dIPR_matrix[r] = g, d
            done += 1
            if progress_cb is not None:
                progress_cb(done, len(jobs))
    else:
        with ProcessPoolExecutor(max_workers=n_workers, mp_context=mp_context) as ex:
            futures = [ex.submit(_theta_row, job) for job in jobs]
            for f in tqdm.tqdm(as_completed(futures), total=len(futures),
                               desc="theta-phi map", unit="row", disable=progress_cb is not None):
                if cancel_evt is not None and cancel_evt.is_set():
                    for g in futures:
                        g.cancel()
                    raise Cancelled()
                r, g, d = f.result()
                G_matrix[r], dIPR_matrix[r] = g, d
                done += 1
                if progress_cb is not None:
                    progress_cb(done, len(jobs))

    if save:
        if data_path:
            os.makedirs(data_path, exist_ok=True)
        np.savez(
            full_path,
            G_matrix=G_matrix, dIPR_matrix=dIPR_matrix,
            phi_lin=phi_lin, theta_lin=theta_lin,
            N=N, V1=V1, V2=V2, t=t,
            level_i=level_i, level_j=level_j, dparameter=dparameter,
        )
        print(f"Saved data to {full_path}")

    return dict(G_matrix=G_matrix, dIPR_matrix=dIPR_matrix,
                phi_lin=phi_lin, theta_lin=theta_lin, cached=False)
