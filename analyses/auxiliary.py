import numpy as np
from numpy import ma
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels as stats
from typing import Optional

class Auxiliary:

    def __init__(self):
        pass

    def convert_pvalue_to_asterisks(self, pvalue: float) -> str:
        """
        Convert p-value to asterisks notation.
        Args:
            pvalue (float): P-value.
        Returns:
            str: Asterisks notation for the p-value.
        """
        if pvalue <= 0.0001:
            return "****"
        elif pvalue <= 0.001:
            return "***"
        elif pvalue <= 0.01:
            return "**"
        elif pvalue <= 0.05:
            return "*"
        return "ns"
    

    def plot_significance(self, ax: object, x: np.array, text: list):
        """
        Plot significance markers on the given axis.
        Args:
            ax (Axes): Axes object for the plot.
            x (ndarray): x-coordinates for the significance markers.
            text (list): List of text labels for the significance markers.
        Returns:
            None.
        """
        ylim = ax.get_ylim()
        for i in range(len(text)):
            if '*' in text[i]:
                text_len = len(text[i])
            else:
                text_len = 2
            ax.text(x=x[i] - text_len * 0.05, y=max(ylim), s=text[i])
            ax.hlines(max(ylim), x[i] - 0.5, x[i] + 0.5, 'black')
            
            
    def pval_to_asterix(self, p: float, multiple_comparison: Optional[int]=None):
        """
        Convert p-value to asterisks notation with multiple comparison correction.
        Args:
            p (float): P-value.
            multiple_comparison (int, optional): Number of multiple comparisons. Defaults to None.
        Returns:
            str: Asterisks notation for the p-value.
        """
        if multiple_comparison is None: 
            multiple_comparison = 1
        if p <= 0.001/multiple_comparison: 
            asterix = '***'
        elif p <= 0.01/multiple_comparison: 
            asterix = '**'
        elif p <= 0.05/multiple_comparison: 
            asterix = '*'
        else: 
            asterix = '$^{ns}$'
        return asterix
    
    def create_annotation(self, data: np.array, pval: Optional[object]=None, parametric: Optional[bool]=None) -> np.array:
        """
        Create annotation for the given data.
        Args:
            data (ndarray): Data matrix.
            pval (ndarray, optional): P-values for the data. Defaults to None.
            parametric (bool, optional): Whether the data is parametric. Defaults to None.
        Returns:
            ndarray: Annotation for the data.
        """
        dim = np.shape(data)
        data = np.real(data)
        annotation = np.full(dim[0], np.nan, dtype=object)
        for i in range(dim[0]):
            annotation[i] = str(np.round(np.nanmean(data[i, :], 3))) + '\n± ' \
                            + str(np.round(np.nanstd(data[i, :], 2)))

            if pval is not None:
                if pval[i] < 0.05:
                    annotation[i] = pval_to_asterix(pval[i]) + '\n' + annotation[i]

        return annotation  
    
    
    def rotate_xticks_polar(self, ax: object):
        """
        Rotate x-tick labels in a polar plot.
        Args:
            ax (Axes): Axes object for the plot.
        Returns:
            None.
        """
        angles = np.linspace(0,2*np.pi,len(ax.get_xticklabels())+1)
        angles[np.cos(angles) < 0] = angles[np.cos(angles) < 0] + np.pi
        angles = np.rad2deg(angles) + np.array([-90, -90, 90, -90, 90, -90])
        labels = []
        for label, angle in zip(ax.get_xticklabels(), angles):
            x, y = label.get_position()
            lab = ax.text(x, y, label.get_text(), transform=label.get_transform(),
                          ha=label.get_ha(), va=label.get_va())
            lab.set_rotation(angle)
            labels.append(lab)
        ax.set_xticklabels([])
        
        
    def polar_plot(self, y: np.array, l: list, c: object, ax: object, plt: object, pval: Optional[object]=None):
        """
        Create a polar plot.
        Args:
            y (ndarray): Data for the plot.
            l (list): Labels for the plot.
            c (str or list): Colors for the plot.
            ax (Axes): Axes object for the plot.
            plt: Plot object.
            pval (ndarray, optional): P-values for the data. Defaults to None.
        Returns:
            None.
        """
        x = np.linspace(0, 2 * np.pi, len(y))

        lines, labels = plt.thetagrids(range(0, 360, int(360/len(l))), (l))

        if len(y.shape) > 1:
            for i in range(y.shape[1]):
                plt.plot(x, y[:, i], color=c[i], linewidth=3)
        else:
            plt.plot(x, y, color=c, linewidth=3)

        self.rotate_xticks_polar(ax)
        if pval is not None:
            pval_corr = stats.stats.multitest.multipletests(pval, method='bonferroni')[1]
            ax.set_rlim((0, np.max(y)+0.25))
            for i in range(len(y)-1):
                if self.pval_to_asterix(pval[i]) == self.pval_to_asterix(pval_corr[i]):
                    note = self.pval_to_asterix(pval[i])
                else:
                    note = self.pval_to_asterix(pval[i]) + ' $^($' + self.pval_to_asterix(pval_corr[i]) + '$^)$'
                ax.annotate(note, \
                            xy=(ax.get_xticks()[i], np.max(y, axis=(1))[i] + 0.1), \
                            ha='center', va='center')
                
    def plot_matrices(self, controls: dict, dynamics: dict, group: int, mood_categories: list):
        """
        Plot matrices for control and dynamics.
        Args:
            controls (dict): Control data.
            dynamics (dict): Dynamics data.
            group (int): Group index.
            mood_categories (list): Mood categories.
        Returns:
            None.
        """
        fig, axs = plt.subplots(3,2, figsize=[15,15])
        timepoint = ['before', 'after']
        parameter = ['control', 'dynamics']
        par = [controls['matrix'], dynamics['matrix']]
        annotation = np.full_like(controls['matrix'][:,:,0,0], np.nan)
        for j, p in enumerate(np.real(par)):
            for i in range(2):
                vmin = np.nanmean(p[:,:,group,:],axis=2).min()
                vmax = np.nanmean(p[:,:,group,:],axis=2).max()
                a = np.round(np.nanmean(np.real(p[:,:,group,i]),axis=2),2).astype(str)
                b = np.round(np.nanstd(np.real(p[:,:,group,i]),axis=2),2).astype(str)
                annotation = np.array(a,dtype=np.object) + ' ±\n ' + np.array(b,dtype=np.object)
                sns.heatmap(np.nanmean(p[:,:,group,i],axis=2),ax=axs[i,j], \
                            annot=annotation, fmt='', annot_kws={"fontsize":12}, \
                            vmin=vmin, vmax=vmax, xticklabels=mood_categories, yticklabels=mood_categories)
                axs[i,j].set_title(parameter[j] + ' ' + timepoint[i])

            sns.heatmap(np.nanmean(p[:,:,group,1] - \
                                    p[:,:,group,0], axis=2), annot=annotation, fmt='', annot_kws={"fontsize":12}, \
            xticklabels=mood_categories, yticklabels=mood_categories, ax=axs[2,j])
            axs[2,j].set_title(parameter[j] + ' after - before')
        plt.tight_layout()
