import argparse
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from bayes_opt import BayesianOptimization
from sklearn.model_selection import cross_validate
#from sklearn.model_selection import cross_val_score

class var_test_function(object):

    def __init__(self,df,var,features,diz=False,n_jobs=1):
    
        self.df = df
        
        self.var = var
        self.features = features
        self.target = ['{}_corr_diff_scale'.format(var)]
        self.n_jobs = n_jobs
        self.robSca = RobustScaler()
        self.diz = diz
        if self.diz:
            self.df.loc[np.logical_and(self.df[self.var] != 0, self.df['{}_corr'.format(self.var)] != 0), '{}_corr_diff_scale'.format(var)] = self.robSca.fit_transform(np.array(self.df.loc[np.logical_and(self.df[self.var] != 0, self.df['{}_corr'.format(self.var)] != 0), '{}_corr'.format(var)] - self.df.loc[np.logical_and(self.df[self.var] != 0, self.df['{}_corr'.format(self.var)] != 0), var]).reshape(-1,1))
            self.df.loc[np.logical_or(self.df[self.var] == 0, self.df['{}_corr'.format(var)] == 0), '{}_corr_diff_scale'.format(var)] = 0
        else:
            self.df['{}_corr_diff_scale'.format(var)] = self.robSca.fit_transform(np.array(self.df['{}_corr'.format(var)] - self.df[var]).reshape(-1,1))


    def test_function_int(self,max_depth,gamma,reg_lambda,reg_alpha,min_child_weight,subsample,n_estimators):
    
        clf = xgb.XGBRegressor(n_estimators=n_estimators,max_depth=max_depth,gamma=gamma,reg_lambda=reg_lambda,reg_alpha=reg_alpha,min_child_weight=min_child_weight,subsample=subsample,tree_method='gpu_hist')
        if self.diz:
            X = self.df.loc[np.logical_and(self.df[self.var] != 0, self.df['{}_corr'.format(self.var)] != 0), self.features].values
            Y = self.df.loc[np.logical_and(self.df[self.var] != 0, self.df['{}_corr'.format(self.var)] != 0), self.target].values
        else:
            X = self.df.loc[:,self.features].values
            Y = self.df.loc[:,self.target].values

        cross_val = cross_validate(clf,X,Y,cv=5,n_jobs=self.n_jobs,return_train_score=True)
        print('Training sample score: ',cross_val['train_score'])
        return cross_val['test_score'].mean()

    def test_function(self,max_depth,gamma,reg_lambda,reg_alpha,min_child_weight,subsample,n_estimators):

        max_depth = int(max_depth)
        gamma = max(gamma,0)
        reg_lambda = max(reg_lambda,0)
        reg_alpha = max(reg_alpha,0)
        subsample = max(min(subsample,1),0)
        n_estimators = int(n_estimators)

        return self.test_function_int(max_depth,gamma,reg_lambda,reg_alpha,min_child_weight,subsample,n_estimators)

def main(options):
    df = pd.read_hdf('/work/threiten/QReg/ReReco18ABCPromptD/df_mc_EB_train_corr_5M.h5',columns=['probePt','probeScEta','probePhi','rho','probeCovarianceIeIp','probeS4','probeR9','probePhiWidth','probeSigmaIeIe','probeEtaWidth','probeCovarianceIeIp_corr','probeS4_corr','probeR9_corr','probePhiWidth_corr','probeSigmaIeIe_corr','probeEtaWidth_corr','probePhoIso','probeChIso03','probeChIso03worst','probePhoIso_corr','probeChIso03_corr','probeChIso03worst_corr'])

    pbounds = {'max_depth': (3,15), 'gamma': (0,10), 'reg_lambda': (0,10), 'reg_alpha': (0,10), 'min_child_weight': (1,1000), 'subsample': (0.5,1), 'n_estimators': (1000,5000)}
    if options.variable == "probePhoIso":
        features = ['probePt','probeScEta','probePhi','rho','probePhoIso']
        diz = True
    elif options.variable == "probeChIso03" or options.variable == "probeChIso03worst":
        features = ['probePt','probeScEta','probePhi','rho','probeChIso03','probeChIso03worst']
        diz = True
    else:
        features = ['probePt','probeScEta','probePhi','rho','probeCovarianceIeIp','probeS4','probeR9','probePhiWidth','probeSigmaIeIe','probeEtaWidth']
        diz = False
    optimizer = BayesianOptimization(f = var_test_function(df,options.variable,features,diz,options.n_jobs).test_function, pbounds = pbounds, verbose=2)

    print("Starting Gaussian process!")
    optimizer.maximize(n_iter=30,init_points=2)
    print("Best parameters: ", optimizer.max)
    for i, res in enumerate(optimizer.res):
        print("Iteration {}: \n\t{}".format(i, res))

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    requiredArgs = parser.add_argument_group('Required Arguements')
    requiredArgs.add_argument('-v','--variable', action='store', type=str, required=True)
    # requiredArgs.add_argument('-E','--EBEE', action='store', type=str, required=True)
    requiredArgs.add_argument('-n','--n_jobs', action='store', type=int, required=True)
    # requiredArgs.add_argument('-N','--n_evts', action='store', type=int, required=True)
    # requiredArgs.add_argument('-c','--config', action='store', type=str, required=True)
    options=parser.parse_args()
    main(options)
