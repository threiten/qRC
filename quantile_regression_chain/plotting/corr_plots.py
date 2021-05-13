import numpy as np
import pandas as pd
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt

def wcorr(arr1,arr2,weights):
    m1 = np.average(arr1,weights=weights)*np.ones_like(arr1)
    m2 = np.average(arr2,weights=weights)*np.ones_like(arr2)
    cov_11 = float((weights*(arr1-m1)**2).sum()/weights.sum())
    cov_22 = float((weights*(arr2-m2)**2).sum()/weights.sum())
    cov_12 = float((weights*(arr1-m1)*(arr2-m2)).sum()/weights.sum())
    return cov_12/np.sqrt(cov_11*cov_22)

class corrMat:

    def __init__(self,df_mc,df_data,varrs,varrs_corr,weightst,label=''):

        self.label=label
        self.varrs = varrs
        self.varrs_corr = varrs_corr


        self.mc_crl = np.array([[100*wcorr(df_mc[var1].values,df_mc[var2].values,df_mc[weightst]) for var2 in varrs] for var1 in varrs])
        self.mc_c_crl = np.array([[100*wcorr(df_mc[var1].values,df_mc[var2].values,df_mc[weightst]) for var2 in varrs_corr] for var1 in varrs_corr])

        self.data_crl = np.array([[100*wcorr(df_data[var1].values,df_data[var2].values,df_data['weight'].values) for var2 in varrs] for var1 in varrs])

        self.fig_name=[]

        self.mc_crl_meanabs = np.mean(np.abs(self.mc_crl))
        self.mc_c_crl_meanabs = np.mean(np.abs(self.mc_c_crl))
        self.data_crl_meanabs = np.mean(np.abs(self.data_crl))

        self.diff_crl_meanabs = np.mean(np.abs(np.array(self.mc_crl)-np.array(self.data_crl)))
        self.diff_c_crl_meanabs = np.mean(np.abs(np.array(self.mc_c_crl)-np.array(self.data_crl)))


    def plot_corr_mat(self,key):

        self.key = key
        fig1 = plt.figure()
        ax1 = fig1.add_subplot(111)

        plt.set_cmap('bwr')

        if key == 'data':
            cax1 = ax1.matshow(self.data_crl,vmin=-100, vmax=100)
            self.plot_numbers(ax1,self.data_crl)
            plt.title(r'Correlation data ' + self.label.replace('_',' ')  + ' Mean abs: {:.3f}'.format(self.data_crl_meanabs),y=1.4 )
            name = 'data_' + self.label
        elif key == 'mc':
            cax1 = ax1.matshow(self.mc_crl,vmin=-100, vmax=100)
            self.plot_numbers(ax1,self.mc_crl)
            plt.title(r'Correlation mc ' + self.label.replace('_',' ') +  ' Mean abs: {:.3f}'.format(self.mc_crl_meanabs), y=1.4)
            name = 'mc_' + self.label
        elif key == 'mcc':
            cax1 = ax1.matshow(self.mc_c_crl,vmin=-100, vmax=100)
            self.plot_numbers(ax1,self.mc_c_crl)
            plt.title(r'Correlation mc corrected ' + self.label.replace('_',' ') +  ' Mean abs: {:.3f}'.format(self.mc_c_crl_meanabs), y=1.4)
            name = 'mc_corr_' + self.label
        elif key == 'diff':
            cax1 = ax1.matshow(np.array(self.mc_crl)-np.array(self.data_crl),vmin=-15,vmax=15)
            self.plot_numbers(ax1,np.array(self.mc_crl)-np.array(self.data_crl))
            plt.title(r'Correlation difference ' + self.label.replace('_',' ') +  ' Mean abs: {:.3f}'.format(self.diff_crl_meanabs),y=1.4)
            name = 'diff_' + self.label
        elif key == 'diffc':
            cax1 = ax1.matshow(np.array(self.mc_c_crl)-np.array(self.data_crl),vmin=-15,vmax=15)
            self.plot_numbers(ax1,np.array(self.mc_c_crl)-np.array(self.data_crl))
            plt.title(r'Correlation difference corrected ' + self.label.replace('_',' ') +  ' Mean abs: {:.3f}'.format(self.diff_c_crl_meanabs), y=1.4)
            name = 'diff_corr_' + self.label

        cbar = fig1.colorbar(cax1)
        cbar.set_label(r'Correlation (\%)')

        for i in range(len(self.varrs)):
            self.varrs[i]=self.varrs[i].replace('probe','')
        ax1.set_yticks(np.arange(len(self.varrs)))
        ax1.set_xticks(np.arange(len(self.varrs)))
        ax1.set_xticklabels(self.varrs,rotation='vertical')
        ax1.set_yticklabels(self.varrs)

        self.fig_name.append((fig1,name))

    def plot_numbers(self,ax,mat):
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                c = mat[j,i]
                if np.abs(c)>=1:
                    ax.text(i,j,'{:.0f}'.format(c),fontdict={'size': 8},va='center',ha='center')

    def save(self,outDir):
        for fig,name in self.fig_name:
            fig.savefig(outDir + '/crl_' + name.replace(' ','_') + '.png',bbox_inches='tight')
            fig.savefig(outDir + '/crl_' + name.replace(' ','_') + '.pdf',bbox_inches='tight')
