import numpy as np
from pykalman.pykalman import *
import matplotlib.pyplot as plt
import seaborn as sns
from statannotations.Annotator import Annotator
import statsmodels.formula.api as smf
from scipy import stats, spatial
from scipy.stats import f
import control
from numpy import ma
import pandas as pd
import scipy
from typing import Optional

class Statistics:

    def __init__(self):
        pass

    def TwoSampleT2Test(self, X: np.array, Y: np.array) -> tuple:
        """
        Perform a two-sample Hotelling's T^2 test.
        Args:
            X (ndarray): First sample data matrix.
            Y (ndarray): Second sample data matrix.
        Returns:
            tuple: T^2 statistic, test statistic, p-value.
        """
        nx, p = X.shape
        ny, _ = Y.shape
        delta = np.mean(X, axis=0) - np.mean(Y, axis=0)
        Sx = np.cov(X, rowvar=False)
        Sy = np.cov(Y, rowvar=False)
        S_pooled = ((nx-1)*Sx + (ny-1)*Sy)/(nx+ny-2)
        t_squared = (nx*ny)/(nx+ny) * np.matmul(np.matmul(delta.transpose(), np.linalg.inv(S_pooled)), delta)
        statistic = t_squared * (nx+ny-p-1)/(p*(nx+ny-2))
        F = f(p, nx+ny-p-1)
        p_value = 1 - F.cdf(statistic)
        return t_squared, statistic, p_value
    
    
    def PairedSampleT2Test(self, X: np.array, Y: np.array) -> tuple:
        """
        Perform a paired Hotelling's T^2 test.
        Args:
            X (ndarray): First sample data matrix.
            Y (ndarray): Second sample data matrix.
        Returns:
            tuple: T^2 statistic, test statistic, p-value.
        """
        n, p = X.shape
        D = X - Y
        delta = np.mean(D, axis=0)
        S = np.cov(D, rowvar=False)
        t_squared = n * np.matmul(np.matmul(delta.transpose(), np.linalg.inv(S)), delta)
        statistic = t_squared * (n-p)/(p*(n-1))
        F = f(p, n-p)
        p_value = 1 - F.cdf(statistic)
        return t_squared, statistic, p_value
    
    
    def stats_group_difference(self, data: np.array, df: pd.DataFrame) -> tuple:
        """
        Calculate group differences and perform statistical tests.
        Args:
            data (ndarray): Data matrix.
            df (DataFrame): DataFrame containing condition information.
        Returns:
            DataFrame: Results of statistical tests.
            list: List of p-values for single item tests.
            list: List of t-values for single item tests.
        """
        single_pval, single_tval = [], []  # for stats between single items
        df_stats = pd.DataFrame(columns=['T2', 'Fstats', 'pvalue'])

        # group difference before and after
        for i in range(2):
            dd = data[:, :, i]
            T2, tstats, pval = self.TwoSampleT2Test(dd[:, df['randomized_condition'] == 0].T,
                                                    dd[:, df['randomized_condition'] == 1].T)
            df_stats.loc[i] = [T2, tstats, pval]

            for i in range(np.shape(data)[0]):
                tstats = stats.ttest_ind(dd[i, df['randomized_condition'] == 0].T,
                                         dd[i, df['randomized_condition'] == 1].T)
                single_pval.append(tstats[1])
                single_tval.append(tstats[0])

        # change
        dd = data[:, :, 1] - data[:, :, 0]
        T2, tstats, pval = self.TwoSampleT2Test(dd[:, df['randomized_condition'] == 0].T,
                                                dd[:, df['randomized_condition'] == 1].T)
        df_stats.loc[2] = [T2, tstats, pval]

        df_stats.index = ['before intervention', 'after intervention', 'change (after - before)']
        sig = df_stats['pvalue'] < 0.001
        df_stats['pvalue'] = df_stats['pvalue'][~sig].round(3)
        df_stats['pvalue'][sig] = '< 0.001'
        df_stats[['T2', 'Fstats']] = df_stats[['T2', 'Fstats']].round(2)

        return df_stats, single_pval, single_tval
    
    
    def make_df(self, data: np.array, df: pd.DataFrame, data_labels: list) -> pd.DataFrame:
        """
        Create a DataFrame for plotting group differences.
        Args:
            data (ndarray): Data matrix.
            df (DataFrame): DataFrame containing condition information.
            data_labels (list): List of data labels.
        Returns:
            DataFrame: Melted DataFrame for plotting.
        """
        Nsj = len(df)
        if len(data.shape) > 2:
            d1 = data[:, :, 0]
            d2 = data[:, :, 1]
        else:
            d1 = data[:, 0]
            d2 = data[:, 1]
        tmp = pd.DataFrame(np.hstack((d1, d2)).T, columns=data_labels)
        tmp['time'] = np.hstack((np.zeros(Nsj), np.ones(Nsj)))
        tmp['condition'] = 2 * df['randomized_condition'].to_list()
        tmp['condition_time'] = tmp['time'] + tmp['condition'] * 2
        df_melted = tmp.melt(id_vars=['condition_time'], value_vars=data_labels, value_name='rating', var_name='category')

        return df_melted


    def plot_group_difference(self, df: pd.DataFrame, data_labels: list, data_name: str, \
                              pvalues: Optional[object]=None, pairs: Optional[list]=None, \
                              fig: Optional[object]=None, ax: Optional[object]=None, \
                              ms: Optional[str]=None, test: Optional[str]=None, \
                              showfliers: Optional[bool]=True, alpha_adjusted: Optional[float]=0.05, \
                             show_ns: Optional[bool]=True, show_legend: Optional[bool]=True, \
                             legend_loc: Optional[object]=(1.04, 0)) -> object:
        """
        Plot group differences and perform statistical tests.
        Args:
            df (DataFrame): DataFrame for plotting.
            data_labels (list): List of data labels.
            data_name (str): Name of the data.
            pvalues (ndarray, list): P-values for statistical tests. Defaults to None.
            pairs (list, optional): List of pairs for annotation. Defaults to None.
            fig (Figure, optional): Figure object for the plot. Defaults to None.
            ax (Axes, optional): Axes object for the plot. Defaults to None.
            ms (str, optional): Method for multiple comparison correction. Defaults to None.
            test (str, optional): Statistical test method. Defaults to 'Mann-Whitney'.
            showfliers (bool, optional): Whether to show outliers in the boxplot. Defaults to True.
        Returns:
            Figure: Figure object.
            Axes: Axes object.
        """
        if pairs is None:
            pairs = []
            for t in range(2):
                for i in data_labels:
                    pairs.append([(i, 0+t), (i, 2+t)])

        hue_plot_params = {
            'data': df,
            'x': 'category',
            'y': 'rating',
            "hue": "condition_time",
            "hue_order": [2, 0, 3, 1],
            "palette": {0: "lightgrey", 1: "grey", 2: "plum", 3: "purple"},
            "showmeans": True,
        }

        # Create new plot
        if fig is None:
            fig, ax = plt.subplots(figsize=[20, 8])

        # Plot with seaborn
        ax = sns.boxplot(ax=ax, **hue_plot_params, showfliers=showfliers)
        ax.set_ylim(ax.get_ylim())  # solves weird bug in annotator

        # Add annotations
        if ms is None:
            ms = "Holm-Bonferroni"
        if ms is False:
            ms = None
        if test is None:
            test = 'Mann-Whitney'

        if pvalues is not None and show_ns==False:
            sig_pairs = np.where(np.array(pvalues) <= alpha_adjusted)[0]
            pairs = [pairs[i] for i in sig_pairs]
            pvalues = [pvalues[i] for i in sig_pairs]

        annotator = Annotator(ax, pairs, **hue_plot_params)
        if pvalues is None:
            annotator.configure(verbose=0, test=test, comparisons_correction=ms)
            annotator.apply_and_annotate()
        else:
            annotator.configure(verbose=0)
            annotator.set_pvalues_and_annotate(np.array(pvalues))

        # Label and show
        if show_legend == True:
            ax.legend(loc=legend_loc, fontsize=16)
            ax.set(xlabel='', title=data_name + ' before and after intervention')
            ax.legend_.texts[0].set_text('before emotion regulation intervention')
            ax.legend_.texts[1].set_text('before control intervention')
            ax.legend_.texts[2].set_text('after emotion regulation intervention')
            ax.legend_.texts[3].set_text('after control intervention')
        else:
            ax.legend([])
        return fig, ax
    
    
    def test_normality(self, data: np.array) -> list:
        """
        Perform normality test on the data.
        Args:
            data (ndarray): Data matrix.
        Returns:
            list: List of boolean values indicating normality assumption for each item.
        """
        dim = np.shape(data)
        normality_assumption = []
        if len(dim) > 2:
            for i in range(dim[0]):
                for j in range(dim[1]):
                    normality_test = scipy.stats.shapiro(data[i, j, :])
                    normality_assumption.append(normality_test[1] > 0.05)
        elif len(dim) == 2:
            for i in range(dim[0]):
                normality_test = scipy.stats.shapiro(data[i, :])
                normality_assumption.append(normality_test[1] > 0.05)
        elif len(dim) == 1:
            normality_test = scipy.stats.shapiro(data)
            normality_assumption.append(normality_test[1] > 0.05)

        return normality_assumption

    def t_statistic(self, x: np.array, y: np.array) -> float:
        """
        Calculate the t-statistic for two independent samples.
        Args:
            x (ndarray): First sample data.
            y (ndarray): Second sample data.
        Returns:
            float: T-statistic.
        """
        return scipy.stats.ttest_ind(x, y).statistic
    
    def detect_outlier(self, data: np.array) -> tuple:
        """
        Detect outliers in data using the Interquartile Range (IQR) method.
        Args:
            data (ndarray): Data array.
        Returns:
            ex (tuple): Indices of the outliers in the data.
        """
        Q1 = np.percentile(data, 25)
        Q3 = np.percentile(data, 75)
        IQR = 1.5 * (Q3 - Q1)
        ex = np.where((data < Q1 - IQR) | (data > Q3 + IQR))
        return ex

    def exclude_outliers(self, data: np.array, kf_before: int, kf_after: int, results: str) -> tuple:
        """
        Excludes outliers from data based on eigenvalues and singular values.
        Args:
            data (ndarray): Data array.
            kf_before (int): Determine which Kalman filter to use at t=0.
            kf_after (int): Determine which Kalman filter to use at t=1.
            results (str): Determine from where to take the Kalman filter estimates.
        Returns:
            tuple: Indices of excluded data and a boolean array indicating if data is an outlier.
        """
        Nsj = len(data)
        ee, cc = np.empty((2,Nsj)), np.empty((5,2,Nsj))
        for i in range(Nsj):
            for j, p in enumerate([kf_before, kf_after]):
                eigval,foo = np.linalg.eig(data[i][results][p].transition_matrices)
                ee[j,i] = np.sort(np.real(eigval))[-1]
                foo,cc[:,j,i],foo = np.linalg.svd(control.ctrb(data[i][results][p].transition_matrices, \
                                                          data[i][results][p].control_matrix))

        # exclude eigenvalues > 1
        ex = np.where(ee[:,:] > 1)[1]
        # exclude outliers in dominant singular value of controllability matrix
        ex = np.hstack((ex, self.detect_outlier(cc[0,:,:])[1]))

        ex_arr = np.zeros(len(data))  # Assuming an array size of 100, you can adjust this according to your requirements
        ex_arr[np.unique(ex)] = 1

        return ex, ex_arr == 1
    
    
    def four_way_statistics(self, data: dict, df: pd.DataFrame) -> tuple:
        """
        Perform four-way statistical tests between different conditions.
        Args:
            data (dict): A dictionary containing mood ratings data for different mood categories.
            df (DataFrame): DataFrame containing condition information.
        Returns:
            list: A list containing p-values for the statistical tests.
        """

        # Create an empty list to store p-values
        tstats, pvalues = [], []
        
        n_mooditems = data.shape[0]
        # Loop through mooditems
        for m in range(n_mooditems):
            # Get mood ratings data for the specific mood category
            dd = data[m, :, :]

            # Perform statistical tests for each condition (before and after)
            for t in range(2):
                # Two-sample t-test between condition 0 and condition 1 at time t
                ttest = scipy.stats.ttest_ind(dd[df['randomized_condition'] == 0, t].T,
                                               dd[df['randomized_condition'] == 1, t].T)
                pvalues.append(ttest.pvalue)
                tstats.append(ttest.statistic)

                # Paired t-test between the same condition at time 0 and time 1
                ttest = scipy.stats.ttest_rel(dd[df['randomized_condition'] == t, 0].T,
                                               dd[df['randomized_condition'] == t, 1].T)
                pvalues.append(ttest.pvalue)
                tstats.append(ttest.statistic)

        return tstats, pvalues

