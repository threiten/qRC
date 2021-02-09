import numpy as np
import pandas as pd
import quantileRegression_chain as QReg_C
import xgboost as xgb
import os
import yaml, argparse
from joblib import Parallel, delayed, parallel_backend

def main(options):

    stream = file(options.config,'r')
    inp=yaml.load(stream)

    dfs = inp['dataframes']
    variables = inp['variables']
    year = str(inp['year'])
    workDir = inp['workDir']
    weightsDir = inp['weightsDir']

    if year == '2017':
        cols=["mass","probeScEnergy","probeScEta","probePhi","run","weight","weight_clf","rho","probeR9","probeSigmaIeIe","probePhiWidth","probeEtaWidth","probeCovarianceIeIp","probeCovarianceIpIp","probeS4","probePhoIso","probeChIso03","probeChIso03worst","probeSigmaRR","probeScPreshowerEnergy","probePt"]
    elif year == '2016':
        cols=["mass","probeScEnergy","probeScEta","probePhi","run","weight","weight_clf","rho","probeR9","probeSigmaIeIe","probePhiWidth","probeEtaWidth","probeCovarianceIetaIphi","probeCovarianceIphiIphi","probeS4","probePhoIso","probeChIso03","probeChIso03worst","probeSigmaRR","probeScPreshowerEnergy","probePt","probePhoIso_corr"]

    qRC = QReg_C.quantileRegression_chain(year,options.EBEE,workDir,variables)
    qRC.setupJoblib('long_6gb')
    qRC.loadMCDF(dfs['mc_{}'.format(options.EBEE)]['input'],0,options.n_evts)
    qRC.loadDataDF(dfs['data_{}'.format(options.EBEE)]['input'],0,options.n_evts)
    
    #quantiles = [0.01,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,0.99]


    for var in variables:
        correctY_old_parallel(qRC.MC,qRC.data,var,weights='weight_clf',n_jobs=10,backend=qRC.backend)
        # correctY_old_parallel(qr_mc_EE,qr_data_EE,var,weights='weight_clf',n_jobs=10)
        qRC.loadClfs(var,weightsDir)
        qRC.correctY(var,n_jobs=10)

    if year == '2017':
        weights = ("/mnt/t3nfs01/data01/shome/threiten/QReg/ReReco17_data/camp_3_1_0/PhoIdMVAweights/HggPhoId_94X_barrel_BDT_v2.weights.xml","/mnt/t3nfs01/data01/shome/threiten/QReg/ReReco17_data/camp_3_1_0/PhoIdMVAweights/HggPhoId_94X_endcap_BDT_v2.weights.xml")
        leg2016=False
    elif year == '2016':
        weights = ("/mnt/t3nfs01/data01/shome/threiten/QReg/ReReco16/PhoIdMVAweights/HggPhoId_barrel_Moriond2017_wRhoRew.weights.xml","/mnt/t3nfs01/data01/shome/threiten/QReg/ReReco16/PhoIdMVAweights/HggPhoId_endcap_Moriond2017_wRhoRew.weights.xml")
        leg2016=True

    mvas = [ ("newPhoID","data",[]), ("newPhoIDcorrSS","qr",variables), ("newPhoIDcorrSS_old","old",variables)]

    qRC.computeIdMvas( mvas[:1],  weights,'data', n_jobs=10, leg2016=leg2016)
    qRC.computeIdMvas( mvas, weights,'mc', n_jobs=10 , leg2016=leg2016)
        
    df_backup_mc = qRC.MC.loc[:,['probeChIso03worst','probeChIso03','probePhoIso']]
    df_backup_data = qRC.data.loc[:,['probeChIso03worst','probeChIso03','probePhoIso']]

    IsoZ=False
    if options.IsoZ:
        IsoZ = options.IsoZ

    if IsoZ:
        qRC.MC['probeChIso03worst'] = get_const_val(qRC.data['probeChIso03worst'])*np.ones_like(qRC.MC['probeChIso03worst'])
        qRC.MC['probeChIso03'] = np.zeros_like(qRC.MC['probeChIso03'])
        qRC.MC['probePhoIso'] = np.zeros_like(qRC.MC['probePhoIso'])
        if year == '2016':
            qRC.MC['probePhoIso_corr'] = np.zeros_like(qRC.MC['probePhoIso_corr'])
        qRC.data['probeChIso03worst'] = get_const_val(qRC.data['probeChIso03worst'])*np.ones_like(qRC.data['probeChIso03worst'])
        qRC.data['probeChIso03'] = np.zeros_like(qRC.data['probeChIso03'])
        qRC.data['probePhoIso'] = np.zeros_like(qRC.data['probePhoIso'])

        mvasIsoZ = [ ("newPhoIDIsoZ","data",[]), ("newPhoIDIsoZcorrSS","qr",variables),("newPhoIDIsoZcorrSS_old","old",variables)]
        mvas.extend(mvasIsoZ)
        qRC.computeIdMvas( mvasIsoZ[:1],  weights,'data', n_jobs=10, leg2016=leg2016)
        qRC.computeIdMvas( mvasIsoZ, weights, 'mc',  n_jobs=10 , leg2016=leg2016)
    
    for var in ['probeChIso03worst','probeChIso03','probePhoIso']:
        qRC.MC[var] = df_backup_mc[var]
        qRC.data[var] = df_backup_data[var]
        
    for (mva,dmy1,dmy2) in mvas:
        if 'IsoZ' in mva:
            qRC.MC[maketr(mva)] = get_quantile(qRC.MC,qRC.data,mva,'newPhoIDIsoZ')
        else:
            qRC.MC[maketr(mva)] = get_quantile(qRC.MC,qRC.data,mva,'newPhoID')

    qRC.data['newPhoIDtr'] = get_quantile(qRC.data,qRC.data,'newPhoID','newPhoID')
    qRC.data['newPhoIDtrIsoZ'] = get_quantile(qRC.data,qRC.data,'newPhoIDIsoZ','newPhoIDIsoZ')

    qRC.MC.to_hdf('{}/{}'.format(workDir,dfs['mc_{}'.format(options.EBEE)]['output']),'df', mode='w', format='t')
    qRC.data.to_hdf('{}/{}'.format(workDir,dfs['data_{}'.format(options.EBEE)]['output']),'df', mode='w', format='t')

def correctY_old_parallel(df_mc,df_data,var,weights=None,n_jobs=1,backend='loky'):
    
    #centers=0.5*(binss_cum(var)[1:]+binss_cum(var)[:-1])
    sort_df = df_mc.sort_values(var)
    if weights!=None:
        w_cum = np.cumsum(sort_df[weights].values)
    else:
        w_cum = np.array(range(len(df_mc.index)))/float(len(df_mc.index))
    cdf_mc = np.vstack((w_cum/w_cum[-1],np.sort(df_mc[var].values)))
    cdf_data = np.vstack((np.array(range(len(df_data.index)))/float(len(df_data.index)),np.sort(df_data[var].values)))
    
    print "Correcting " + var + " using old method"
    with parallel_backend(backend):
        Y_corr=np.concatenate(Parallel(n_jobs=n_jobs,verbose=20)(delayed(correctY_old_arr)
                              (arr,cdf_mc,cdf_data) for arr in np.array_split(df_mc[var].values,n_jobs)))
                       
    df_mc[var+'_old_corr']=Y_corr
    
def correctY_old_evt_impr(val,cdf_mc,cdf_data):
    if np.searchsorted(cdf_mc[1],val) >= len(cdf_mc[0]):
        cum_mc=1
    else:
        cum_mc = cdf_mc[0][np.searchsorted(cdf_mc[1],val)]
    if cum_mc==1 or np.searchsorted(cdf_data[0],cum_mc)>=len(cdf_data[0]):
        return cdf_data[1][-1]
    else:
        return cdf_data[1][np.searchsorted(cdf_data[0],cum_mc)]

def correctY_old_arr(arr,cdf_mc,cdf_data):
    return [correctY_old_evt_impr(val,cdf_mc,cdf_data) for val in arr]

def get_const_val(vals,bins=1000):
    hist, bins = np.histogram(vals,bins=bins)
    centers = 0.5*(bins[1:] + bins[:-1])
    return centers[np.where(hist==hist.max())][0]

def get_quantile(df,df_ref,var,var_ref,weights=None):
    if weights == None:
        cdf = np.vstack((np.hstack((np.array(range(len(df_ref.index)))/float(len(df_ref.index)),[1])),np.hstack((np.sort(df_ref[var_ref].values),1.2*np.sort(df_ref[var_ref].values)[-1]))))
    else:
        sort_df = df_ref.sort_values(var_ref)
        w_cum = np.cumsum(sort_df[weights].values)
        cdf = np.vstack((np.hstack((w_cum/w_cum[-1],[1])),np.sort(df_ref[var_ref].values)))
    return np.apply_along_axis(transform,0,df[var].values,cdf)
    
def transform(val,cdf):
    ind = np.searchsorted(cdf[1],val)
    if val.any()>(cdf[1][-1]):
        print val, ind, cdf[0][ind]# cdf[1][ind]
    return cdf[0][ind]

def maketr(mvast):
    corrstr = mvast[(mvast.find('newPhoID')+8):]
    print corrstr
    if corrstr == '':
        return mvast
    else:
        return mvast.replace(corrstr,'tr'+corrstr)

if __name__=="__main__":
     parser=argparse.ArgumentParser()
     requiredArgs = parser.add_argument_group('Required Arguments')
     requiredArgs.add_argument('-c','--config', action='store', default='quantile_config.yaml', type=str,required=True)
     requiredArgs.add_argument('-E','--EBEE', action='store', type=str, required=True)
     requiredArgs.add_argument('-N','--n_evts', action='store', type=int, required=True)
     parser.add_argument('-I','--IsoZ', dest='IsoZ', action='store_true')
     
     options=parser.parse_args()
     main(options)


