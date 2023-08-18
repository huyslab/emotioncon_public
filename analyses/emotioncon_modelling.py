import numpy as np
from pykalman.pykalman import *
import matplotlib.pyplot as plt
from scipy import stats, spatial
import control
from numpy import ma
from typing import Optional
import pandas as pd

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
    
    def input_from_videos(self, filepath) -> np.array:
        """
        Generates an array that specifies which video category was shown at a specific time point.
        Args:
        Returns:
            inp (ndarray): Input array 0 = not input present, 1 = input present.
        """
        # Read the input data from the CSV file
        input_df = pd.read_csv(filepath)

        # Convert the 'category' column to dummy variables using one-hot encoding
        inp_dummies = pd.get_dummies(input_df['category'])

        # Reindex the columns to ensure consistent order and fill missing categories with zeros
        inp_dummies = inp_dummies.reindex(columns=['Disgust', 'Amusement', 'Calmness', 'Anxiety', 'Sadness'])

        # Convert the one-hot encoded categorical variables to a numpy array
        inp = inp_dummies.values

        # Add zero rows at the beginning and end of the 'inp' array to prepare inputs
        inp = np.vstack([np.zeros(np.shape(inp)[1]), inp, np.zeros(np.shape(inp)[1]), inp])
        return inp
    
    def define_kf_options(self, data: dict, inp: np.array) -> dict:
        """
        Define options for the Kalman Filter.
        Args:
            data (dict): Data dictionary.
            inp (ndarray): Input array 0 = not input present, 1 = input present.
        Returns:
            option (dict) s: Including all set options. 
        """
        # Create a dictionary to store the KF options
        options = {'cov': 'diagonal', 'inputs': True, 'inp': inp}

        # Calculate the number of observed dimensions (features) in the data
        options['n_dim_obs'] = np.shape(data)[1]

        # Set the number of state dimensions equal to the number of observed dimensions
        options['n_dim_state'] = options['n_dim_obs']

        # Calculate the number of control dimensions based on the input data
        options['n_dim_control'] = np.shape(inp)[1]

        # Set the maximum number of iterations for optimization in KF
        options['maxiter'] = 100

        # Set the constraint control to False (not using constraint control)
        options['constraint control'] = False

        # Define the list of variables for Expectation-Maximization (EM) updates
        options['em_vars'] = ['transition_matrices', 'transition_offsets', 'control_matrix', \
                              'observation_covariance', 'transition_covariance']

        # Initialize matrices A, h, and C to zeros based on the state and control dimensions
        options['A'] = np.zeros((options['n_dim_state'], options['n_dim_state']))
        options['h'] = np.zeros(options['n_dim_state'])
        options['C'] = np.zeros((options['n_dim_state'], options['n_dim_control']))

        return options

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
    
    def plot_simulated_data(self, data: dict, inp: np.array, mood_categories: list):
        """
        Plot emotion ratings and on top simulated data using parameter estimates.
        Args:
            data (dict): Data dictionary.
            df (pd.DataFrame): 
        Returns:
        """
        mood_ratings = data['ratings']
        n_mooditems, T = data['ratings'].shape
      
        fig, axs = plt.subplots(1, n_mooditems, figsize=[20,3])
        x = np.empty((T,5))
        
        for t in range(2):
            inp_split = inp[t*int(T/2):t*int(T/2) + int(T/2),:]
            z, x[t*int(T/2):t*int(T/2) + int(T/2),:] = \
            data['results_split'][13+t].sample(int(T/2),  initial_state=data['ratings'][:,t*int(T/2)], \
                                                    control_inputs=inp_split)

        for k in range(n_mooditems):
            axs[k].plot(mood_ratings[k,:]);
            axs[k].plot(x[:,k], linestyle='--');
            axs[k].set_title(mood_categories[k]);
            axs[k].set_ylabel('ratings');
            axs[k].set_xlabel('trials');
            axs[k].vlines(np.where(inp[:,k]), ymin=0, ymax=100, color='gray', linestyles = 'dashed')
            ylim = axs[k].get_ylim()
            axs[k].add_patch(plt.Rectangle((0, ylim[0]), 54, np.abs(ylim[0]) + ylim[1], edgecolor=(0, 0, 1, 0.1), facecolor=(0, 0, 1, 0.1)))
            axs[k].add_patch(plt.Rectangle((55, ylim[0]), 54, np.abs(ylim[0]) + ylim[1], edgecolor=(0, 1, 0, 0.1), facecolor=(0, 1, 0, 0.1)))
        plt.legend(['empirical data','model data', 'video clips','before intervention', 'after intervention'], loc=(1.04,0))
        plt.tight_layout()
        plt.show()

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
    
    def KF_comparison(self, data: dict, likelihood: str, number_parameters: np.array, number_models: int) -> tuple:
        """
        Calculate bic for each subject and hence the best fitting model.
        Args:
            data (dict): Data dict
            number_parameters (ndarray): Array containing the number of parameters for each model
            number_models (int): Total number of models
        Returns:
            tuple: bic scores and histogram of bestmodel per subject
        """
        Nsj = len(data)
        ll_all_subjects = np.array([data[i][likelihood] for i in range(Nsj)]).T

        Ndatapoints = np.prod(np.shape(data[0]['ratings']))
        bic,bestmodel=[],[]
        for i in range(Nsj):
            a, b = self.AICBIC_calc(ll_all_subjects[:,i], Ndatapoints*np.ones(number_models), number_parameters)
            bestmodel.append(np.where(b==min(b))[0])
            bic.append(b)
        
        aic, bic = self.AICBIC_calc(np.nanmean(ll_all_subjects,axis=1), Ndatapoints*np.ones(number_models), number_parameters)
        
        return aic, bic, bestmodel
    
    def plot_BIC(self, fig: object, ax: object, bic: np.array, labels: list, nparameter: object, bestmodel: Optional[list]=None):
        """
        Plot the BIC scores for different models.
        Args:
            fig (figure)
            ax (axis)
            bic (ndarray): Array containing the BIC scores for different models.
            labels (list): List of labels corresponding to each model.
            nparameter (object): Number of parameters for each model.
            bestmodel (Optional[list]): List containing the best model index for each subject. Default is None.
        Returns:
            None.
        """
        ax.barh(np.arange(len(bic)), np.squeeze(np.array(bic - min(bic))), align='center', alpha=0.5)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(nparameter)
        ax.set_ylabel('# of parameters')
        ax.set_xlabel('$\Delta$ BIC score');
        for bar, label in zip(ax.patches, labels[::-1]):
            ax.text(2, bar.get_y()+bar.get_height()/2, label, ha = 'left', va = 'center', size=18)
        
        x_location = max(bic - min(bic)) * 1.01
        ax.scatter(x_location, np.where(np.squeeze(np.array(bic - min(bic)))==0), marker = '*', s=500, color='red')
        if bestmodel != None:
            for i in range(len(labels)):
                ax.text(x_location * 1.1, i, sum(np.array(bestmodel)==i)[0], ha = 'left', va = 'center', size=16) 
    

