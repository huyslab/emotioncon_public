import numpy as np
from pykalman.pykalman import *
import matplotlib.pyplot as plt
from scipy import stats, spatial
import control
from numpy import ma
from typing import Optional

class Modelling:
    """
    Class for modeling a Linear Dynamical System and performing related operations.
    """

    def __init__(self):
        pass

    def sample_dynamicsmatrix(self, n_dim_state: int) -> np.array:
        """
        Sample a dynamics/transition matrix 'A' with eigenvalues between 0 and 1.
        Args:
            n_dim_state (int): Number of dimensions for the state
        Returns:
            ndarray: Sampled dynamics matrix 'A'
        """
        A_diag = stats.uniform.rvs(size=n_dim_state)
        A_offdiag = stats.multivariate_normal().rvs([n_dim_state, n_dim_state]) / n_dim_state
        A = np.diag(A_diag) + (A_offdiag - np.diag(A_offdiag))

        while any(np.abs(np.linalg.eigvals(A)) >= 1):
            A_diag = stats.uniform.rvs(size=n_dim_state)
            A_offdiag = stats.multivariate_normal().rvs([n_dim_state, n_dim_state]) / n_dim_state
            A = np.diag(A_diag) + (A_offdiag - np.diag(A_offdiag))

        return A

    def initialize_lds(self, options: dict) -> dict:
        """
        Initialize parameters to simulate a Linear Dynamical System.
        Args:
            options (dict): Options for LDS initialization
        Returns:
            dict: Initialized parameters for LDS simulation
        """
        n_dim_state = options['n_dim_state']
        n_dim_obs = options['n_dim_obs']
        n_dim_control = options['n_dim_control']

        if options['cov'] == 'diagonal':
            Q = 0.1 * np.eye(n_dim_obs)
            R = 0.1 * np.eye(n_dim_obs)
        elif options['cov'] == 'full':
            Q = 0.1 * np.random.rand(n_dim_state, n_dim_state)
            Q = np.dot(Q, Q.T)
            R = 0.1 * np.random.rand(n_dim_obs, n_dim_obs)
            R = np.dot(R, R.T)

        params = {
            'A': self.sample_dynamicsmatrix(n_dim_state),  # state transition matrix
            'Q': Q,  # state noise covariance
            'B': np.eye(n_dim_state),  # observation matrix
            'C': stats.multivariate_normal.rvs(size=[n_dim_state, n_dim_control]),  # observation matrix
            'R': R,  # observation noise covariance
            'mu_0': np.zeros(n_dim_state),  # initial state mean
            'sigma_0': 0.1 * np.eye(n_dim_state),  # initial state noise covariance
        }
        return params

    def create_inputs(self, T: int, n_dim_control: int) -> np.array:
        """
        Create inputs for the LDS simulation.
        Args:
            T (int): Number of time steps
            n_dim_control (int): Number of dimensions for the control inputs
        Returns:
            ndarray: Inputs for the LDS simulation
        """
        inp = np.zeros([T, n_dim_control])
        i = np.tile(np.arange(0, n_dim_control), int(T / n_dim_control))
        for k in range(len(i)):
            inp[k, i[k]] = 1
        np.random.shuffle(inp)
        return inp

    def sample_lds(self, n_timesteps: int, params: dict, options: dict) -> tuple:
        """
        Generate samples from a Linear Dynamical System.
        Args:
            n_timesteps (int): Number of time steps to simulate
            params (dict): Dictionary of model parameters (A, Q, B, C, R, mu_0, sigma_0)
            options (dict): Options for LDS simulation
        Returns:
            tuple: Simulated state and observation arrays
        """
        n_dim_state = params['A'].shape[0]
        n_dim_obs = params['B'].shape[0]

        if options['inputs']:
            inp = options['inp']
        else:
            inp = np.zeros([n_timesteps, np.shape(params['C'])[1]])

        if 'h' not in params:
            params['h'] = np.zeros((1, n_dim_state))

        zi = stats.multivariate_normal(cov=params['Q'], allow_singular=True).rvs(n_timesteps)
        eta = stats.multivariate_normal(cov=params['R'], allow_singular=True).rvs(n_timesteps)

        state = np.zeros((n_timesteps, n_dim_state))
        obs = np.zeros((n_timesteps, n_dim_obs))

        for t in range(n_timesteps):
            if t == 0:
                state[t] = stats.multivariate_normal(mean=params['mu_0'] + params['C'] @ inp[0],
                                                     cov=params['sigma_0']).rvs(1)
            else:
                state[t] = params['A'] @ state[t - 1] + params['h'] + params['C'] @ inp[t] + zi[t]

            obs[t] = params['B'] @ state[t] + eta[t]

        return state, obs

    def run_KF(self, obs: np.array, options: dict) -> object:
        """
        Run the Kalman Filter (KF) algorithm for state estimation.
        Args:
            obs (ndarray): Observations from the LDS
            options (dict): Options for the KF algorithm
        Returns:
            KalmanFilter: Kalman Filter object
        """
        kf = KalmanFilter(
            n_dim_state=options['n_dim_state'],
            n_dim_obs=options['n_dim_obs'],
            em_vars=options['em_vars'],
        )

        if 'A' in options:
            kf.transition_matrices = options['A']
        if 'h' in options:
            kf.transition_offsets = options['h']
        if 'C' in options:
            kf.control_matrix = options['C']
        if 'G' in options:
            kf.observation_covariance = options['G']
        if 'S' in options:
            kf.transition_covariance = options['S']

        if options['inputs']:
            kf.n_dim_control = options['n_dim_control']

        nanidx = np.isnan(obs[:, 0])
        obs[nanidx, :] = np.ma.masked

        kf.initial_state_mean = obs[0, :]
        kf.initial_state_covariance = 0.1 * np.eye(options['n_dim_state'])

        if options['cov'] == 'diagonal':
            for i in range(options['maxiter']):
                if options['inputs']: 
                    kf = kf.em(obs, n_iter=1, control_inputs = options['inp'])
                    if options['constraint control']:
                        kf.control_matrix = np.diagflat(np.diag(kf.control_matrix))
                else: 
                    kf = kf.em(obs, n_iter=1)
                kf.observation_covariance = np.diagflat(np.diag(kf.observation_covariance))
                kf.transition_covariance = np.diagflat(np.diag(kf.transition_covariance))
        elif options['cov'] == 'full':
            if options['inputs']: 
                kf = kf.em(obs, n_iter=options['maxiter'], control_inputs = options['inp'])
                if options['constraint control']:
                    kf.control_matrix = np.diagflat(np.diag(kf.control_matrix))
            else: kf = kf.em(obs, n_iter=options['maxiter'])
        return kf
    
    def save_kf_options(self, kf: object, options: dict) -> dict:
        """
        Save the Kalman Filter (KF) options to a dictionary.
        Args:
            kf (KalmanFilter): Kalman Filter object
            options (dict): Options dictionary to save the KF options

        Returns:
            dict: Updated options dictionary with KF options
        """
        options['A'] = kf.transition_matrices
        options['h'] = kf.transition_offsets
        options['C'] = kf.control_matrix
        options['G'] = kf.observation_covariance
        options['S'] = kf.transition_covariance
        return options

    def parameter_short(self, l: list) -> list:
        """
        Shortens parameter names for easier referencing.
        Args:
            l (list): List of parameter names
        Returns:
            list: Updated list with shortened parameter names
        """
        parameter = []
        if 'S' in l:
            l = list(map(lambda x: x.replace('S', 'transition_covariance'), l))
        if 'G' in l:
            l = list(map(lambda x: x.replace('G', 'observation_covariance'), l))
        if 'A' in l:
            l = list(map(lambda x: x.replace('A', 'transition_matrices'), l))
        if 'h' in l:
            l = list(map(lambda x: x.replace('h', 'transition_offsets'), l))
        if 'C' in l:
            l = list(map(lambda x: x.replace('C', 'control_matrix'), l))
        if 'B' in l:
            l = list(map(lambda x: x.replace('C', 'observation_matrices'), l))
        return l

    def calculate_distance(self, true_dynamics: np.array, est_dynamics: np.array) -> float:
        """
        Calculates the absolute dot product between the first eigenvectors of true and estimated dynamics.
        Args:
            true_dynamics (ndarray): True dynamics matrix
            est_dynamics (ndarray): Estimated dynamics matrix
        Returns:
            float: Absolute dot product between the first eigenvectors
        """
        true_eigenspectrum = np.linalg.eig(true_dynamics)
        true_eigvals = np.abs(true_eigenspectrum[0])
        true_eigvecs = np.abs(true_eigenspectrum[1])
        idx = np.where(true_eigvals == max(true_eigvals))
        true_first_eigvec = true_eigvecs[:, idx[0][0]]

        est_eigenspectrum = np.linalg.eig(est_dynamics)
        est_eigvals = np.abs(est_eigenspectrum[0])
        est_eigvecs = np.abs(est_eigenspectrum[1])
        idx = np.where(est_eigvals == max(est_eigvals))
        est_first_eigvec = est_eigvecs[:, idx[0][0]]

        return np.abs(np.dot(true_first_eigvec, est_first_eigvec))

    def AICBIC_calc(self, logL: float, numObs: int, numParam: int) -> tuple:
        """
        Calculates the Akaike Information Criterion (AIC) and Bayesian Information Criterion (BIC) given log-likelihood,
        number of observations, and number of parameters.
        Args:
            logL (float): Log-likelihood
            numObs (int): Number of observations
            numParam (int): Number of parameters
        Returns:
            tuple: AIC and BIC values
        """
        aic = -2 * logL + 2 * numParam
        bic = -2 * logL + np.log(numObs) * numParam
        return aic, bic

    def eigenspectrum(self, matrix: np.array) -> tuple:
        """
        Computes the eigenvalues and eigenvectors of a matrix, sorted in descending order by eigenvalues.
        Args:
            matrix (ndarray): Input matrix
        Returns:
            tuple: Sorted eigenvalues and corresponding eigenvectors
        """
        eigvals, eigvecs = np.linalg.eig(matrix)
        idx_sorted = np.flip(np.argsort(eigvals))
        return eigvals[idx_sorted], eigvecs[:, idx_sorted]

    def align_vectors(self, v1: np.array, v2: np.array) -> tuple:
        """
        Aligns two vectors in the same direction by checking their dot product and flipping the second vector if needed.
        Args:
            v1 (ndarray): First vector
            v2 (ndarray): Second vector
        Returns:
            tuple: Aligned vectors
        """
        angle = np.dot(v1, v2)
        if angle < 0:
            v2 = -v2
        return v1, v2

    def controllability(self, A: np.array, C: np.array) -> tuple:
        """
        Computes the controllability matrix and performs Singular Value Decomposition to obtain singular values and
        right singular vectors.
        Args:
            A (ndarray): State transition matrix
            C (ndarray): Control matrix
        Returns:
            tuple: Singular values and right singular vectors
        """
        controllability = control.ctrb(A, C)
        u, s, vh = np.linalg.svd(controllability)

        return s, u

