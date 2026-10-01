import numpy as np
from AAH import AAH_model as aah


def density_of_states_over_phason(N: int, V1: float, V2: float, phi_lin: list, t: float = 1, k: float = 0,
                                   idx: int = 0) -> tuple[np.ndarray, list]:
    '''
    Compute the probability density for a given eigenstate across a range of phason angles.

    :param:
        N: chain length
        V1: on-site potential amplitude
        V2: hopping potential amplitude
        phi_lin: list of phason angles to compute the probability density for
        t: hopping term
        k: parameter
        idx: index of the eigenstate to compute the probability density for

    :return:
        sites: array of site indices
        densities: list of probability densities for each phason angle
    '''

    densities = []  # list of probability densities

    for phi in phi_lin:
        AAH = aah.AAH_chain(N=N, t=t, V1=V1, V2=V2, k=k, phi=phi)  # create an AAH chain
        _ = AAH.probability_density()                              # diagonalize and get probability densities

        density = AAH._amplitudes[idx]  # probability density of the desired eigenstate
        densities.append(density)

    densities = np.array(densities)
    sites = np.arange(1, N + 1)  # site indices

    return sites, densities


def find_winding_states(N: int, energies: list, parameters: np.ndarray, n_gaps: int = 1, only_positive: bool = False) -> list:
    '''
    Finds the indices of the largest gaps in the energy spectrum.

    :param:
        N: number of levels in the spectrum
        energies: energy values of the spectrum (list, over the swept parameter, of eigenvalue arrays)
        parameters: list of swept parameters
        n_gaps: number of levels to find
        only_positive: if True, only consider states whose energy is positive
    :return:
        list of indices of the levels to highlight
    '''

    param = np.asarray(parameters)
    E = np.empty((param.size, N))

    for j in range(len(param)):
        E[j, :] = energies[j]  # shape (angles, N)

    deltaE = np.diff(E, axis=1)

    max_over_phi = deltaE.max(axis=0)  # shape (N-1,)
    gap_below = np.concatenate(([-np.inf], max_over_phi))  # shape (N,)
    gap_above = np.concatenate((max_over_phi, [-np.inf]))  # shape (N,)

    winding_score = np.minimum(gap_below, gap_above)

    if only_positive:
        mean_E = E.mean(axis=0)          # shape (N,) - reference energy per state
        winding_score = np.where(mean_E > 0, winding_score, -np.inf)

    order = np.argsort(winding_score)[::-1]
    top_n = order[:n_gaps]

    results = []

    for rank, k in enumerate(top_n, start=1):
        k = int(k)
        results.append(k)

    return results


def find_gaps(N: int, energies: list, parameters: np.ndarray, n_gaps: int = 1, only_positive: bool = False) -> list:
    '''
    Finds the indices and the phi indecies of the largest gaps in the energy spectrum.

    :param:
        N: number of levels in the spectrum
        energies: list of energy arrays
        parameters: list of swept parameters
        n_gaps: number of gaps to find
        only_positive:  if True, only consider states, whose energy is positive
    :return:
        list of dictionaries containing the indices of the levels and the phi index of the gap
    '''

    param = np.asarray(parameters)
    E = np.empty((param.size, N))

    for j in range(len(param)):
        E[j, :] = energies[j]  # shape (N,) per phi

    deltaE = np.diff(E, axis=1)  # shape (n_phi, N-1)

    max_over_phi = deltaE.max(axis=0)
    argmax_over_phi = deltaE.argmax(axis=0)

    if only_positive:
        valid = np.where(max_over_phi > 0)[0]
    else:
        valid = np.arange(max_over_phi.size)

    order = valid[np.argsort(max_over_phi[valid])[::-1]]
    top_n = order[:n_gaps]

    results = []
    for rank, i_star in enumerate(top_n, start=1):
        j_star = argmax_over_phi[i_star]
        results.append({
            "levels": (int(i_star), int(i_star) + 1),
            "phi_index": int(j_star),
            "phi_value": float(param[j_star]),
        })

    return results

def find_min_gap(N: int, energies: list, parameters: np.ndarray, level_i: int, level_j: int) -> dict:
    '''
    Finds the parameter (phi) value for which the energy gap between two specified levels is smallest.

    :param:
        N: number of levels in the spectrum
        energies: energy values of the spectrum (list over the swept parameter)
        parameters: list of swept parameters (e.g. phason angle phi)
        level_i: index of the first level
        level_j: index of the second level

    :return:
         the phi index and value at which the gap is minimal
    '''

    param = np.asarray(parameters)
    E = np.empty((param.size, N))

    for j in range(len(param)):
        E[j, :] = energies[j]

    gap = np.abs(E[:, level_j] - E[:, level_i])  # shape (n_phi,)

    j_star = int(np.argmin(gap))
    p = param[j_star]*np.pi

    return j_star, p


def find_max_ipr_delta(N: int, t: float, V1: float, V2: float, theta: float, phi: float,
                        level: int, delta_list: np.ndarray = None, vary: str = 'phi', min_abs_delta: float = 1e-6) -> tuple:
    '''
    :parameters:
        N, t, V1, V2: AAH_chain parameters (same meaning as in AAH_model).
        theta: fixed theta value if vary == 'phi'
        phi: fixed phi value if vary == 'theta'
        level: index of the eigenstate whose IPR is checked.
        delta_list: iterable of candidate delta values (radians) to scan.
        vary: which angle delta perturbs -- 'phi' (default) or 'theta'.
        min_abs_delta: deltas with absolute value below this are excluded from the search (default 1e-6; effectively excludes delta = 0).

    :return:
        (max_ipr, best_delta)
    '''
    if vary not in ('phi', 'theta'):
        raise ValueError("vary must be 'phi' or 'theta'")

    if delta_list is None:
        delta_list = np.linspace(min_abs_delta, np.pi, 4000)

    iprs = np.full(len(delta_list), np.nan)

    for j, d in enumerate(delta_list):
        if vary == 'phi':
            chain = aah.AAH_chain(N=N, t=t, V1=V1, V2=V2, theta=theta, phi=phi+d)
        else:  # vary == 'theta'
            chain = aah.AAH_chain(N=N, t=t, V1=V1, V2=V2, theta=theta+d, phi=phi)

        chain.diagonalize()
        iprs[j] = chain.IPR()[level]

    best_idx = np.argmax(iprs)
    best_delta = delta_list[best_idx]
    max_ipr = iprs[best_idx]

    return max_ipr, best_delta


def weighted_fidelity(psi: np.ndarray, psi_T: np.ndarray, amp_T: np.ndarray) -> float:
    '''
    Finds the fidelity between two states.
    :param
        psi: recieved state
        psi_T: target state
        amp_T: amplitudes of the target state

    :return:
        fidelity: fidelity between the two states
    '''

    # Helper function to calculate alpha
    def alpha(amplitudes):
        alpha = (amplitudes[-1] - amplitudes[0]) / (max(amplitudes))
        return alpha

    a = alpha(amp_T) # find alpha
    fidelity = a * (np.abs(psi.conj() @ psi_T) ** 2)

    return fidelity




