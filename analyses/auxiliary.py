import numpy as np
from numpy import ma
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class Auxiliary:

    def __init__(self):
        pass

    def convert_pvalue_to_asterisks(self, pvalue):
        if pvalue <= 0.0001:
            return "****"
        elif pvalue <= 0.001:
            return "***"
        elif pvalue <= 0.01:
            return "**"
        elif pvalue <= 0.05:
            return "*"
        return "ns"
    

    def plot_significance(self, ax, x, text):
        ylim = ax.get_ylim()
        # ax.set_ylim(ylim[0], ylim[1] + max(ylim)/10)
        for i in range(len(text)):
            if '*' in text[i]:
                text_len = len(text[i])
            else:
                text_len = 2
            ax.text(x=x[i] - text_len * 0.05, y=max(ylim), s=text[i])
            ax.hlines(max(ylim), x[i] - 0.5, x[i] + 0.5, 'black')
            
            
    def pval_to_asterix(self, p, multiple_comparison=None):
        if multiple_comparison == None: 
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
    
    def create_annotation(self, data, pval=None, parametric=None):
        dim = np.shape(data)
        data = np.real(data)
        annotation = np.full(dim[0], np.nan, dtype=object)
        for i in range(dim[0]):
            annotation[i] = str(np.round(np.nanmean(data[i,:]),3)) + '\n± ' \
                            + str(np.round(np.nanstd(data[i,:]),2))


            if pval is not None:
                if pval[i] < 0.05:
                    annotation[i] = pval_to_asterix(pval[i]) + '\n' + annotation[i]

        return annotation  
    
    
    def rotate_xticks_polar(self, ax):
        angles = np.linspace(0,2*np.pi,len(ax.get_xticklabels())+1)
        angles[np.cos(angles) < 0] = angles[np.cos(angles) < 0] + np.pi
        angles = np.rad2deg(angles) + np.array([-90, -90, 90, -90, 90, -90])
        labels = []
        for label, angle in zip(ax.get_xticklabels(), angles):
            x,y = label.get_position()
            lab = ax.text(x,y, label.get_text(), transform=label.get_transform(),
                          ha=label.get_ha(), va=label.get_va())
            lab.set_rotation(angle)
            labels.append(lab)
        ax.set_xticklabels([])
        
        
    def polar_plot(self, y, l, c, ax, plt, pval=None):
        x = np.linspace(0, 2 * np.pi, len(y))

        lines, labels = plt.thetagrids(range(0, 360, int(360/len(l))), (l))

        if len(y.shape) > 1:
            for i in range(y.shape[1]):
                plt.plot(x, y[:,i], color=c[i], linewidth=3)
        else:
            plt.plot(x, y, color=c, linewidth=3)

        self.rotate_xticks_polar(ax)
        if (pval != None):
            ax.set_rlim((0, np.max(y)+0.1))
            for i in range(len(l)):
                ax.annotate(self.pval_to_asterix(pval[i]), \
                            xy=(ax.get_xticks()[i], np.max(y, axis=(1))[i] + 0.05), \
                            ha='center', va='center')
                
    def plot_matrices(self, controls, dynamics, group, mood_categories):
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