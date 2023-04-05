import numpy as np
from numpy import ma
import pandas as pd
from emotioncon_modelling import Modelling
emo = Modelling()
import control

class PrepEmotioncon:

    def __init__(self):
        pass

    def stimulus_numbering(self, data, mood_categories):
        number_array = []
        data = data.split('"')
        x = [x for x in data if x in mood_categories]
        for i in mood_categories:
            number_array.append(x.index(i))
        return number_array

    def extract_reponse(self, data):
        if data != 'null':
            data = data.split(',')
            data[0] = data[0][1:]
            data[-1] = data[-1][:-1]
            response = [int(x) for x in data]
        else:
            response = np.full(5, np.nan)
        return response

    def extract_attention_checks(self, data):
        x = np.array(data.split(','))
        x[x == 'null'] = 2
        return x[-10:].astype(int)

    def normalize_2d(self, matrix):
        nanidx = np.isnan(matrix)
        matrix[nanidx] = np.ma.masked
        # Only this is changed to use 2-norm put 2 instead of 1
        norm = np.linalg.norm(matrix, 1)
        # normalized matrix
        matrix = matrix/norm  
        return matrix
    
    def normalize(self, matrix):
        new_matrix = np.full_like(matrix, np.nan)
        nonnanidx = ~np.isnan(matrix[:,0])
        new_matrix[nonnanidx,:] = (matrix[nonnanidx,:] - np.mean(matrix[nonnanidx,:],axis=0)) \
                                / np.nanstd(matrix[nonnanidx,:],axis=0)
        return new_matrix

    def extract_questionnaire_reponse(self, data):
        data = data.split(',')
        return [int(x.split(':')[1][0]) for x in data]

    def extract_intervention_success(self, data):
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
        
    def zscore(self, data):
        data_zscored = (data - data.mean())/data.std(ddof=0)
        return data_zscored
    
    def convert_stats_table(self, table, cells):
        results_as_html = table.summary().tables[1].as_html()
        x = pd.read_html(results_as_html, header=0, index_col=0)[0]
        return x.iloc[cells]
    
    def create_df_from_dict(self, data):
        mood_rating_list, symptoms_list = [], []
        df = pd.DataFrame()
        for j, i in enumerate(data):
            mood_rating_list.append(i['ratings'])
            df.loc[j, 'randomized_condition'] = int(i['iddoc']['intervention_condition']==0)
            df.loc[j, 'randomized_videoset'] = i['iddoc']['videoset_condition']
            df.loc[j, 'intervention_success'] = \
            self.extract_intervention_success(i['datadoc']['intervention_sucess_response'])
            symptoms_list.append(i['symptoms'])
        mood_ratings = np.array(mood_rating_list).T
        symptoms = np.array(symptoms_list)
        return df, mood_ratings, symptoms
    
    def rearrange_data_for_stats(self, data):
        Nsj = len(data)
        controls, dynamics, noise = [dict() for i in range(3)]
        controls['matrix_alldata'], dynamics['matrix_alldata'] = \
            [np.empty((5,5,Nsj), dtype = np.complex_) for i in range(2)]
        controls['matrix'], controls['vec'], dynamics['matrix'], dynamics['vec'] = \
            [np.empty((5,5,Nsj,2), dtype = np.complex_) for i in range(4)]
        controls['val'], controls['HL'], dynamics['val'], dynamics['HL'], dynamics['det'], noise['G'], noise['S'] = \
            [np.empty((5,Nsj,2), dtype = np.complex_) for i in range(7)]
        for sj, i in enumerate(data):
            controls['matrix_alldata'][:,:,sj] = i['results_split'][0].control_matrix
            dynamics['matrix_alldata'][:,:,sj] = i['results_split'][0].transition_matrices
            for k,j in enumerate([-2,-1]):
                noise['G'][:,sj,k] = np.diag(i['results_split'][j].observation_covariance)
                noise['S'][:,sj,k] = np.diag(i['results_split'][j].transition_covariance)
                controls['matrix'][:,:,sj,k] = i['results_split'][j].control_matrix
                dynamics['matrix'][:,:,sj,k] = i['results_split'][j].transition_matrices
                controllability = control.ctrb(i['results_split'][j].transition_matrices, \
                                               i['results_split'][j].control_matrix)
                u,s,vh = np.linalg.svd(controllability)
#                 e,v = emo.eigenspectrum(controllability@controllability.T)
                controls['vec'][:,:,sj,k] = np.real(u)
                controls['val'][:,sj,k] = np.real(s)
                controls['HL'][:,sj,k] = np.log(0.5) / np.log(s)
#                 controls['det'][:,sj,k] = np.linalg.det(controllability)
                e,v = emo.eigenspectrum(i['results_split'][j].transition_matrices)
                dynamics['vec'][:,:,sj,k] = np.real(v)
                dynamics['val'][:,sj,k] = np.real(e)
                dynamics['HL'][:,sj,k] = np.log(0.5) / np.log(e)
                dynamics['det'][:,sj,k] = np.linalg.det(i['results_split'][j].transition_matrices)

        for i in ['val', 'vec']:
            for p in [controls, dynamics]:
                p['abs' + i] = np.abs(p[i])

        controls['valvec'] = controls['val'] * controls['vec']
        dynamics['valvec'] = dynamics['val'] * dynamics['vec']
        return dynamics, controls

    