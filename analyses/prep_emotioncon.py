import numpy as np
from numpy import ma
import pandas as pd
from emotioncon_modelling import Modelling
emo = Modelling()
import control
from typing import Optional

class PrepEmotioncon:
    """
    Class for preparing data in the Emotioncon project.
    """

    def __init__(self):
        pass

    def stimulus_numbering(self, data: str, mood_categories: list) -> list:
        """
        Assign numbers to the stimuli based on their order in mood_categories.
        Args:
            data (str): Data containing the stimuli names.
            mood_categories (list): List of mood categories.
        Returns:
            list: Numbered array representing the stimuli order.
        """
        number_list = []
        data = data.split('"')
        x = [x for x in data if x in mood_categories]
        for i in mood_categories:
            number_list.append(x.index(i))
        return number_list

    def extract_reponse(self, data: str) -> list:
        """
        Extract response data from a string and convert it to integers.
        Args:
            data (str): Response data string.
        Returns:
            list: List of response values as integers.
        """
        if data != 'null':
            data = data.split(',')
            data[0] = data[0][1:]
            data[-1] = data[-1][:-1]
            response = [int(x) for x in data]
        else:
            response = np.full(5, np.nan)
        return response

    def extract_attention_checks(self, data: str) -> np.array:
        """
        Extract attention check data from a string and convert it to integers.
        Args:
            data (str): Attention check data string.
        Returns:
            ndarray: Array of attention check values as integers.
        """
        x = np.array(data.split(','))
        x[x == 'null'] = 2
        return x[-10:].astype(int)

    def normalize_2d(self, matrix: np.array) -> np.array:
        """
        Normalize a 2-dimensional matrix by dividing each element by the 2-norm of the matrix.
        Args:
            matrix (ndarray): Input matrix.
        Returns:
            ndarray: Normalized matrix.
        """
        nanidx = np.isnan(matrix)
        matrix[nanidx] = np.ma.masked
        norm = np.linalg.norm(matrix, 2)
        matrix = matrix / norm  
        return matrix

    def normalize(self, matrix: np.array) -> np.array:
        """
        Normalize a matrix by subtracting the mean and dividing by the standard deviation.
        Args:
            matrix (ndarray): Input matrix.
        Returns:
            ndarray: Normalized matrix.
        """
        new_matrix = np.full_like(matrix, np.nan)
        nonnanidx = ~np.isnan(matrix[:, 0])
        new_matrix[nonnanidx, :] = (matrix[nonnanidx, :] - np.mean(matrix[nonnanidx, :], axis=0)) \
                                / np.nanstd(matrix[nonnanidx, :], axis=0)
        return new_matrix

    def extract_questionnaire_reponse(self, data: str) -> list:
        """
        Extract questionnaire response data from a string.
        Args:
            data (str): Questionnaire response data string.
        Returns:
            list: List of questionnaire response values as integers.
        """
        data = data.split(',')
        return [int(x.split(':')[1][0]) for x in data]

    def extract_intervention_success(self, data: str) -> int:
        """
        Extract intervention success data and convert it to numerical values.
        Args:
            data (str): Intervention success data string.
        Returns:
            int: Numerical value representing intervention success.
        """
        if 'all' in data.lower():
            return 0
        if 'moderately' in data.lower():
            return 1
        if 'know' in data.lower(): 
            return 2
        if 'pretty' in data.lower(): 
            return 3
        if 'extremely' in data.lower(): 
            return 4
        
    def zscore(self, data: np.array) -> np.array:
        """
        Compute the z-score of data.
        Args:
            data (ndarray): Input data.
        Returns:
            ndarray: Z-scored data.
        """
        data_zscored = (data - data.mean()) / data.std(ddof=0)
        return data_zscored
    
    def convert_stats_table(self, table: object, cells: object) -> pd.DataFrame:
        """
        Convert a statsmodels results table to a pandas DataFrame.
        Args:
            table: Statsmodels results table object.
            cells: Indices of cells to extract from the table.
        Returns:
            DataFrame: Extracted cells as a DataFrame.
        """
        results_as_html = table.summary().tables[1].as_html()
        x = pd.read_html(results_as_html, header=0, index_col=0)[0]
        return x.iloc[cells]
    
    def create_df_from_dict(self, data: dict) -> tuple:
        """
        Create a DataFrame from a dictionary of data.
        Args:
            data (dict): Dictionary of data.   
        Returns:
            tuple: DataFrame, mood ratings array, symptoms array.
        """
        mood_rating_list, symptoms_list = [], []
        df = pd.DataFrame()
        
        for j, i in enumerate(data):
            mood_rating_list.append(i['ratings'])
            df.loc[j, 'randomized_condition'] = int(i['iddoc']['intervention_condition'] == 0)
            df.loc[j, 'randomized_videoset'] = i['iddoc']['videoset_condition']
            df.loc[j, 'intervention_success_raw'] = i['datadoc']['intervention_sucess_response']
            df.loc[j, 'intervention_success'] = self.extract_intervention_success(i['datadoc']['intervention_sucess_response'])
            symptoms_list.append(i['symptoms'])
        mood_ratings = np.array(mood_rating_list).T
        symptoms = np.array(symptoms_list)
        return df, mood_ratings, symptoms
    
    def rearrange_data_for_stats(self, data: list) -> tuple:
        """
        Rearrange data for statistical analysis.
        Args:
            data (list): List of data.
        Returns:
            tuple: Dynamics and controls data.
        """
        Nsj = len(data)
        controls, dynamics, noise = [dict() for i in range(3)]
        controls['matrix_alldata'], dynamics['matrix_alldata'] = [np.empty((5, 5, Nsj), dtype=np.complex_) for i in range(2)]
        controls['matrix'], controls['vec'], dynamics['matrix'], dynamics['vec'] = [np.empty((5, 5, Nsj, 2), dtype=np.complex_) for i in range(4)]
        controls['val'], controls['HL'], dynamics['val'], dynamics['HL'], dynamics['det'], noise['G'], noise['S'] = [np.empty((5, Nsj, 2), dtype=np.complex_) for i in range(7)]
        
        for sj, i in enumerate(data):
            controls['matrix_alldata'][:, :, sj] = i['results_split'][0].control_matrix
            dynamics['matrix_alldata'][:, :, sj] = i['results_split'][0].transition_matrices
            
            for k, j in enumerate([-2, -1]):
                noise['G'][:, sj, k] = np.diag(i['results_split'][j].observation_covariance)
                noise['S'][:, sj, k] = np.diag(i['results_split'][j].transition_covariance)
                controls['matrix'][:, :, sj, k] = i['results_split'][j].control_matrix
                dynamics['matrix'][:, :, sj, k] = i['results_split'][j].transition_matrices
                
                controllability = control.ctrb(i['results_split'][j].transition_matrices, i['results_split'][j].control_matrix)
                u, s, vh = np.linalg.svd(controllability)
                controls['vec'][:, :, sj, k] = np.real(u)
                controls['val'][:, sj, k] = np.real(s)
                controls['HL'][:, sj, k] = np.log(0.5) / np.log(s)
                
                e, v = emo.eigenspectrum(i['results_split'][j].transition_matrices)
                dynamics['vec'][:, :, sj, k] = np.real(v)
                dynamics['val'][:, sj, k] = np.real(e)
                dynamics['HL'][:, sj, k] = np.log(0.5) / np.log(e)
                dynamics['det'][:, sj, k] = np.linalg.det(i['results_split'][j].transition_matrices)

        for i in ['val', 'vec']:
            for p in [controls, dynamics]:
                p['abs' + i] = np.abs(p[i])

        controls['valvec'] = controls['val'] * controls['vec']
        dynamics['valvec'] = dynamics['val'] * dynamics['vec']
        
        return dynamics, controls

