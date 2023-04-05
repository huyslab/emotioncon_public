import numpy as np
from pykalman.pykalman import *
import matplotlib.pyplot as plt
from scipy import stats, spatial
import control
from numpy import ma

class Modelling:

    def __init__(self):
        pass

    # ensure that dynamics/transition matrix A has eigenvalues between 0 and 1
    def sample_dynamicsmatrix(self, n_dim_state):
        
        A_diag = stats.uniform.rvs(size=n_dim_state)
        A_offdiag = stats.multivariate_normal().rvs([n_dim_state,n_dim_state]) / n_dim_state
        A = np.diag(A_diag) + (A_offdiag - np.diag(A_offdiag))
        
        while any(np.abs(np.linalg.eigvals(A)) >= 1):
            A_diag = stats.uniform.rvs(size=n_dim_state)
            A_offdiag = stats.multivariate_normal().rvs([n_dim_state,n_dim_state]) / n_dim_state
            A = np.diag(A_diag) + (A_offdiag - np.diag(A_offdiag))
            
        return A

    # initialize parameters to simulate lds
    def initialize_lds(self, options):
        n_dim_state = options['n_dim_state']
        n_dim_obs = options['n_dim_obs']
        n_dim_control = options['n_dim_control']

        if options['cov'] == 'diagonal':
            Q = 0.1 * np.eye(n_dim_obs)
            R = 0.1 * np.eye(n_dim_obs)
        elif options['cov'] == 'full':
            Q = 0.1 * np.random.rand(n_dim_state, n_dim_state)
            Q = np.dot(Q,Q.T)
            R = 0.1 * np.random.rand(n_dim_obs, n_dim_obs)
            R = np.dot(R,R.T)
            
        params = {
            'A': self.sample_dynamicsmatrix(n_dim_state), # state transition matrix
            'Q': Q, # state noise covariance
            'B': np.eye(n_dim_state), # observation matrix
            'C': stats.multivariate_normal.rvs(size=[n_dim_state, n_dim_control]), # observation matrix
            'R': R, # observation noise covariance
            'mu_0': np.zeros(n_dim_state), # initial state mean
            'sigma_0': 0.1 * np.eye(n_dim_state), # initial state noise covariance
        }
        return params

    def create_inputs(self, T, n_dim_control):
        inp = np.zeros([T,n_dim_control])
        i = np.tile(np.arange(0,n_dim_control),int(T/n_dim_control))
        for k in range(len(i)):
            inp[k,i[k]] = 1
        np.random.shuffle(inp)
        return inp

    # generate samples from a Linear Dynamical System
    def sample_lds(self, n_timesteps, params, options):
        # n_timesteps (int): the number of time steps to simulate
        # params (dict): a dictionary of model paramters: (F, Q, H, R, mu_0, sigma_0)
        n_dim_state = params['A'].shape[0]
        n_dim_obs = params['B'].shape[0]
        
        if options['inputs']: inp = options['inp']
        else: inp = np.zeros([n_timesteps,np.shape(params['C'])[1]])
            
        if 'h' not in params: params['h'] = np.zeros((1,n_dim_state))

        # precompute random samples from the provided covariance matrices 
        # mean defaults to 0
        zi = stats.multivariate_normal(cov=params['Q'],allow_singular=True).rvs(n_timesteps)
        eta = stats.multivariate_normal(cov=params['R'],allow_singular=True).rvs(n_timesteps)

        # initialize state and observation arrays
        state = np.zeros((n_timesteps, n_dim_state))
        obs = np.zeros((n_timesteps, n_dim_obs))

        # simulate the system
        for t in range(n_timesteps):
            # write the expressions for computing state values given the time step
            if t == 0:
                state[t] = stats.multivariate_normal(mean=params['mu_0'] + params['C'] @ inp[0], cov=params['sigma_0']).rvs(1)
            else:
                state[t] = params['A'] @ state[t-1] + params['h'] + params['C'] @ inp[t] + zi[t]

            # write the expression for computing the observation
            obs[t] = params['B'] @ state[t] + eta[t]

        return state, obs

    def run_KF(self, obs, options):
        kf = KalmanFilter(
            n_dim_state=options['n_dim_state'],
            n_dim_obs=options['n_dim_obs'],
            em_vars=options['em_vars'],
        )

        if 'A' in options: kf.transition_matrices = options['A']
        if 'h' in options: kf.transition_offsets = options['h']
        if 'C' in options: kf.control_matrix = options['C']
        if 'G' in options: kf.observation_covariance = options['G']
        if 'S' in options: kf.transition_covariance = options['S']

        if options['inputs']: 
            kf.n_dim_control = options['n_dim_control']

        nanidx = np.isnan(obs[:,0])
        obs[nanidx,:] = np.ma.masked
  
        kf.initial_state_mean = obs[0,:]
        kf.initial_state_covariance = 0.1*np.eye(options['n_dim_state'])
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
    
    def save_kf_options(self, kf, options):
        options['A'] = kf.transition_matrices
        options['h'] = kf.transition_offsets
        options['C'] = kf.control_matrix
        options['G'] = kf.observation_covariance
        options['S'] = kf.transition_covariance
        return options
    
    def parameter_short(self, l):
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
    
    # calculate abs dot product between first eigenvectors
    def calculate_distance(self, true_dynamics, est_dynamics):
        true_eigenspectrum = np.linalg.eig(true_dynamics)
        true_eigvals = np.abs(true_eigenspectrum[0])
        true_eigvecs = np.abs(true_eigenspectrum[1])
        idx = np.where(true_eigvals == max(true_eigvals))
        true_first_eigvec = true_eigvecs[:,idx[0][0]]

        est_eigenspectrum = np.linalg.eig(est_dynamics)
        est_eigvals = np.abs(est_eigenspectrum[0])
        est_eigvecs = np.abs(est_eigenspectrum[1])
        idx = np.where(est_eigvals == max(est_eigvals))
        est_first_eigvec = est_eigvecs[:,idx[0][0]]

        return np.abs(np.dot(true_first_eigvec,est_first_eigvec))

    def AICBIC_calc(self, logL, numObs, numParam):
        aic  = -2*logL + 2*numParam;
        bic  = -2*logL + np.log(numObs)*numParam; 
        return aic, bic
    
    def eigenspectrum(self, matrix):
        eigvals, eigvecs = np.linalg.eig(matrix)
        idx_sorted = np.flip(np.argsort(eigvals))
        return eigvals[idx_sorted], eigvecs[:,idx_sorted]

    def align_vectors(self, v1, v2):
        angle = np.dot(v1, v2)
        if angle < 0:
            v2 = -v2
        return v1, v2
    
    def controllability(self, A, C):
        controllability = control.ctrb(A, C)
        u,s,vh = np.linalg.svd(controllability)
        
        return s, u

