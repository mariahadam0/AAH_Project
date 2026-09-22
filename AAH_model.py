import os
from datetime import datetime

import numpy as np
import scipy.linalg
import tqdm

'''
AAH model based on Hamiltonian in F.Liu PhysRevB.91.014108 (2015)

Side note:
Symobols for values of phason angle and phason offset were switched.
In the code below, phi is phason angle and k is phason offset.
'''

tau = (1+np.sqrt(5))*np.pi # irrational frequency of the modulation

class AAH_chain:
    """
    A class representing the Aubry-André-Harper (AAH) chain, providing methods for constructing
    Hamiltonians, calculating derivatives, and analyzing localization properties.

    The AAH chain is a one-dimensional model widely studied in the context
    of quasiperiodic systems, topological physics, and localization phenomena.

    :ivar N: Length of the chain.
    :type N: int
    :ivar t: Hopping term in the chain.
    :type t: float
    :ivar V1: Amplitude of the on-site potential.
    :type V1: float
    :ivar V2: Amplitude of the hopping modulation term.
    :type V2: float
    :ivar Q: Frequency of the modulation, constant.
    :type Q: float
    :ivar phi: Phason angle of the chain.
    :type phi: float
    :ivar theta: Phason offset of the model.
    :type theta: float
    """
    def __init__(self, N: int, t: float = 1, V1: float = 0, V2: float = 0, phi: float = 0, theta: float = 0):
        self.N = N              # chain length
        self.t = t              # hopping term
        self.V1 = V1            # amplitude in on-site potential
        self.V2 = V2            # amplitude in hopping potential
        self.Q = tau            # frequency of the modulation (set constant)
        self.phi = phi          # phason angle
        self.theta = theta      # phason offset

        self._H = None          # Hamiltonian
        self._eigvals = None    # eigenvalues
        self._eigvecs = None    # eigenvectors
        self._amplitudes = None # probability amplitudes

    def Hamiltonian(self):
        if self._H is not None:
            return self._H

        H = np.zeros([self.N, self.N])

        # Hopping terms
        for i in range(self.N-1):
            jump_i = self.t+self.V2*np.cos((i+1+0.5)*self.Q+self.phi)
            H[i, i+1] = jump_i
            H[i+1, i] = jump_i

        # On-site terms
        for i in range(self.N):
            onsite_i = self.V1*np.cos((i+1) * self.Q + self.phi + self.theta)
            H[i, i] = onsite_i

        ### Particle-hole space
        # (that is what causes the doubling of the winding in phason spectrum) - not used here
        #
        # self._H = np.block([
        #
        #     [ H,               np.zeros_like(H)],
        #     [np.zeros_like(H), -np.conjugate(H)]
        # ])

        self._H = H

        return self._H

    def dHamiltonian_dphi(self, phi: float = None):
        '''
        Analytic derivative dH/dphi, holding N, t, V1, V2, theta fixed.

        Since phi enters H only through cos(...+phi) terms, this is used
        (together with dphi/dt) to build dH/dt for the adiabaticity check.

        :param phi: phason angle at which to evaluate the derivative.
                    Defaults to self.phi.
        :return: dH/dphi, an (N, N) array with the same structure as H.
        '''
        if phi is None:
            phi = self.phi

        dH = np.zeros([self.N, self.N])

        for i in range(self.N - 1):
            djump_i = -self.V2 * np.sin((i+1+0.5)*self.Q + phi)
            dH[i, i+1] = djump_i
            dH[i+1, i] = djump_i

        for i in range(self.N):
            donsite_i = -self.V1 * np.sin((i+1) * self.Q + phi + self.theta)
            dH[i, i] = donsite_i

        return dH

    @staticmethod
    def adiabaticity_measure(ref_index: int, eigvals: np.ndarray, eigvecs: np.ndarray,
                              dHdt: np.ndarray, gap_eps: float = 1e-10) -> float:
        '''
        Evaluate the adiabaticity criterion of Eq. (3) in F.Liu et al.,
        Phys. Rev. A 105, L061502 (2022), at a single instant.

        Adiabatic evolution of the state `ref_index` requires eta << 1.

        Parameters:
        - ref_index: column index (in eigvecs) of the state being adiabatically followed.
        - eigvals: instantaneous eigenvalues E_l, shape (N,)
        - eigvecs: instantaneous eigenvectors, columns psi_l, shape (N, N)
        - dHdt: instantaneous dH/dt, shape (N, N)
        - gap_eps: gaps smaller than this are treated as degenerate with the reference state and excluded (would blow up)

        :return: eta, the (real, non-negative) adiabaticity measure
        '''
        psi_ref = eigvecs[:, ref_index]
        E_ref = eigvals[ref_index]

        # <psi_l| dH/dt |psi_ref> for every l at once
        matrix_elements = eigvecs.conj().T @ dHdt @ psi_ref

        gaps = eigvals - E_ref
        mask = np.arange(len(eigvals)) != ref_index
        mask &= np.abs(gaps) > gap_eps

        eta = np.sum(np.abs(matrix_elements[mask] / gaps[mask]))
        return float(eta)

    # Diagonalization

    @property
    def eigvals(self):
        self._ensure_diagonalized()
        return self._eigvals

    @property
    def eigvecs(self):
        self._ensure_diagonalized()
        return self._eigvecs

    def _ensure_diagonalized(self):
        if self._H is None:
            self.Hamiltonian()

        if self._eigvals is None:
            self._eigvals, self._eigvecs = np.linalg.eigh(self.Hamiltonian())
        return None

    def diagonalize(self):
        return self.eigvals, self.eigvecs

    # Localization analysis

    def probability_density(self):
        self._ensure_diagonalized()

        self._amplitudes = (
            np.abs(self._eigvecs[:self.N, :])**2
            # + np.abs(self._eigvecs[self.N:, :])**2  ### particle-hole space
        ).T

        return self._amplitudes

    def IPR(self):
    # Inverse participation ratio
        if self._amplitudes is None:
            self.probability_density()
        return np.sum(self._amplitudes**2, axis=1)

    def MIPR(self):
    # Mean inverse participation ratio
        if self._amplitudes is None:
            self.probability_density()
        ipr = self.IPR()
        return np.sum(ipr) / (2*self.N)

    def PR(self):
    # Participation ratio
        if self._amplitudes is None:
            self.probability_density()
        return 1/np.sum(self._amplitudes**2, axis=1)

    def dIPR(self, dparameter=0.3):
        '''
        Directional inverse participation ratio as in Eq. (20) 4Q.-B. Zeng and R. Lü, Phys. Rev. B 105, 245407 (2022)
        '''

        self._ensure_diagonalized()
        if self._amplitudes is None:
            self.probability_density()

        ipr = self.IPR()

        N = self.N
        j = np.arange(1, N + 1)        # physical site labels - starting from 1
        weight = j - N / 2 - dparameter           # shape (N,)

        abs_vecs = np.abs(self._eigvecs)     # shape (N, num_states)
        norm = np.linalg.norm(self._eigvecs, axis=0)  # should be 1, kept for safety
        P = np.sign(weight @ abs_vecs / norm)          # shape (num_states,)

        return P * ipr

    def dMIPR(self, dparameter=0.3):
        return np.mean(self.dIPR(dparameter=dparameter))


    # Time evolution
    def time_evolution_phi(self, idx: int, target: float,
                            omega: float = 1e-4, NStep: int = 1000,
                            save: bool = False, filename: str = None, folder_name: str = None,
                            track_adiabaticity: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

        '''
        Perform time evolution of the AAH chain at a given state, by adiabatically driving the phason angle (phi) from its current value to a target value.
        theta stays fixed throughout.

        The driven parameter follows:
            phi(t) = phi1 + (phi2 - phi1) * F(t)
        where F(t) is a cosine ramp satisfying F(t=0) = 0 and F(t=t_final) = 1:
            F(t) = (1 - cos(pi * t / t_final)) / 2

        :param
            idx: index of the initial state (eigenstate) to be evolved
            target: final value of phi (phi2). The initial value (phi1) is taken
                    from the chain's current self.phi.
            omega: transfer rate, sets total evolution time T = 2*pi/omega and
                    final time t_final = T/4
            NStep: number of time steps
            save: if True, save (energies, states, times) to a .npz file
            filename: optional path/name for the output file (used only if save=True).
                  If not given, defaults to 'time_evolution_idx{idx}_omega{omega}.npz'
            folder_name: optional folder name for the output file
            track_adiabaticity: if True, also compute, at every time step, the
                    adiabaticity measure of Eq. (3) in F.Liu et al., Phys. Rev. A
                    105, L061502 (2022):

                        eta(t) = sum_{l != idx} |<psi_idx(t)| dH(t)/dt |psi_l(t)>|
                                                / (E_l(t) - E_idx(t))

                    where |psi_idx(t)> is the instantaneous eigenstate continuously
                    connected to the initial state `idx` (tracked by maximal overlap
                    from one step to the next, so it is robust to eigenvalue
                    reordering). eta(t) << 1 indicates the evolution is adiabatic;
                    values approaching or exceeding 1 signal breakdown of adiabaticity
                    (e.g. near the minimum-gap point of the phason sweep).
        :return:
            energies: array of energies at each time step
            states: array of states at each time step
            times: array of times at each time step
            adiabaticity: array of eta(t) at each time step (only returned if
                    track_adiabaticity=True)
        '''

        # Save original parameter values and reset state
        self.Hamiltonian()
        self._ensure_diagonalized()

        phi1 = self.phi
        phi2 = target

        # Define time-stepping parameters
        ti = 0  # initial time
        T = 2 * np.pi / omega
        tf = T / 4  # final time
        deltat = (tf - ti) / NStep  # time step

        # Initial state: eigenstate at idx
        psi0 = self.eigvecs[:, idx].copy().astype(complex)

        # F(t): cosine ramp with F(0) = 0 and F(tf) = 1
        def F(t):
            return np.cos(omega*t)

        # Helper: build H at time t by driving phi
        def H_t(t):
            self.phi = phi2 - (phi2 - phi1) * F(t)
            self._H = None

            return self.Hamiltonian()

        # Time evolution: accumulate full propagator U_F
        NN = self.N
        U_F = np.eye(NN, dtype=complex)

        energies = []
        states = []
        adiabaticity = [] if track_adiabaticity else None

        if track_adiabaticity:
            # instantaneous eigenstate continuously connected to `idx`,
            # tracked step-to-step by maximal overlap (robust to reordering)
            ref_vec = psi0.copy()

        for n in tqdm.tqdm(range(NStep + 1), desc="Time evolution", unit="step"):
            t = n * deltat
            H = H_t(t)  # sets self.phi = phi(t) and returns H(phi(t))
            V = scipy.linalg.expm(-1j * deltat * H)
            U_F = V @ U_F

            psi_t = U_F @ psi0

            energies.append(np.real(psi_t.conj() @ H @ psi_t))
            states.append(psi_t)

            if track_adiabaticity:
                # instantaneous spectrum at the current phi(t)
                inst_eigvals, inst_eigvecs = np.linalg.eigh(H)

                # follow the branch adiabatically connected to psi0 via
                # maximal overlap with the previous instant's eigenvector
                overlaps = np.abs(ref_vec.conj() @ inst_eigvecs)
                ref_index = int(np.argmax(overlaps))
                ref_vec = inst_eigvecs[:, ref_index]

                # dH/dt = (dH/dphi) * (dphi/dt), with phi(t) = phi2 - (phi2-phi1)*F(t)
                # and F(t) = cos(omega*t)  =>  dphi/dt = (phi2-phi1)*omega*sin(omega*t)
                dphidt = (phi2 - phi1) * omega * np.sin(omega * t)
                dHdphi = self.dHamiltonian_dphi(self.phi)
                dHdt = dHdphi * dphidt

                eta = self.adiabaticity_measure(ref_index, inst_eigvals, inst_eigvecs, dHdt)
                adiabaticity.append(eta)

        # Restore original parameter and state
        self.phi = phi1
        self._H = None
        self._eigvals = None
        self._eigvecs = None
        self._amplitudes = None
        self.Hamiltonian()
        self.diagonalize()

        times = np.array([n * deltat for n in range(NStep + 1)])
        energies = np.array(energies)
        states = np.array(states)
        if track_adiabaticity:
            adiabaticity = np.array(adiabaticity)

        if save:
            date = datetime.now().strftime('%Y-%m-%d')
            fol = 'data'

            if filename is None:
                filename = f"time_evolution_idx{idx}_omega{omega}_NStep{NStep}_{date}.npz"

            if folder_name is None:
                folder_name = fr'N{self.N}_V1{self.V1:.2f}_V2{self.V2:.2f}_phi{phi1:.2f}to{phi2:.2f}'

            full_dir = os.path.join(fol, folder_name)
            os.makedirs(full_dir, exist_ok=True)  # ensure data/folder_name exists

            save_kwargs = dict(energies=energies, states=states, times=times,
                                idx=idx, target=target, omega=omega, NStep=NStep)
            if track_adiabaticity:
                save_kwargs['adiabaticity'] = adiabaticity

            np.savez(os.path.join(full_dir, filename), **save_kwargs)
            print(f"Saved time evolution results to '{os.path.join(full_dir, filename)}'")

        if track_adiabaticity:
            return energies, states, times, adiabaticity

        return energies, states, times