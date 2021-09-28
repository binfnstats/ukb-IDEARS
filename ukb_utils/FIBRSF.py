import pandas as pd
import numpy as np
import re
import statsmodels.api as sm
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

from matplotlib.pyplot import figure

from time import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import sklearn
from sklearn.model_selection import KFold
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from xgboost import plot_importance
from sklearn.feature_selection import SelectKBest
from sklearn.inspection import partial_dependence
from sklearn.inspection import plot_partial_dependence
import shap
from xgboost import XGBClassifier


config = dict(scale_pos_weight = 6,subsample = 1, min_child_weight = 5, max_depth = 5, gamma= 2, 
              colsample_bytree= 0.6,smote=1,reps=2)
from datetime import datetime # Current date time in local system 
rundate=datetime.date(datetime.now())

mod_xgb=XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
           colsample_bynode=1, learning_rate=0.1,
           max_delta_step=0,  missing=None,
           n_estimators=60, n_jobs=4, nthread=4, objective='binary:logistic',
           random_state=0, reg_alpha=0, reg_lambda=1, scale_pos_weight=config['scale_pos_weight'],
           min_child_weight=config['min_child_weight'],
           gamma=config['gamma'], colsample_bytree=config['colsample_bytree'],max_depth=config['max_depth'],
           seed=42, silent=None, subsample=1, verbosity=1,eval_metric='auc')


def findcols(df,string):
    return [col for col in df.columns if string in col]

path="../../Data/"

def maskapoedf(df,apoe=1):
    apoemask=(df['Genotype_e3/e4']==1)|(df['Genotype_e4/e4']==1)|\
    (df['Genotype_e2/e4']==1)|(df['Genotype_e1/e4']==1)
    non_apoemask=(df['Genotype_e2/e3']==1)|(df['Genotype_e3/e3']==1)|\
    (df['Genotype_e1/e2']==1)|(df['Genotype_e2/e2']==1)
    
    if apoe==3:
        return df

    if apoe==2:
        return df[apoemask|non_apoemask]
    elif apoe==1:  
        return df[apoemask]
    elif apoe==0:  
        return df[non_apoemask]

    
def col_spec_chars(df):
    df.columns=df.columns.str.replace(',','_')
    df.columns=df.columns.str.replace('<','_')
    df.columns=df.columns.str.replace('>','_')
    df.columns=df.columns.str.replace('[','_')
    df.columns=df.columns.str.replace(']','_')
    return df

def shapplot(list_shap_values,list_test_sets):
    test_set = list_test_sets[0]
    shap_values = np.array(list_shap_values[0])
    for i in range(1,len(list_test_sets)):
        test_set = np.concatenate((test_set,list_test_sets[i]),axis=0)
        shap_values = np.concatenate((shap_values,np.array(list_shap_values[i])),axis=1)
    #bringing back variable names    
    X_test = pd.DataFrame(X[test_set],columns=columns)
    shap.summary_plot(shap_values[1], X_test)
    

def borutafeats(df):
    rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5)
    feat_selector = BorutaPy(rf, n_estimators='auto', verbose=0, random_state=1)

    # find all relevant features - 5 features should be selected
    feat_selector.fit(dem_moddata5years_2[topcols].drop(columns='Dementia').fillna(0).values, 
                      dem_moddata5years_2['Dementia'].values)

    # check selected features - first 5 features are selected
    
    
    borutacols=['eid','Dementia']
    for i,col in enumerate([col for col in df.columns if col!='Dementia']):
        if feat_selector.support_[i]==True:
            borutacols.append(col)
    return feat_selector.support_,borutacols


def rebalance(df,depvar,resizeratio=1):
    mask_disease=(df[depvar]==1)  
    df_out=pd.concat([df[mask_disease],df[~mask_disease].sample(len(df[mask_disease])*resizeratio)],axis=0)
    return df_out


def borutarun(df,depvar,resizeratio=1):
    
    df=rebalance(df,depvar,resizeratio)
    
    print(df.shape)
    rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5)
    feat_selector = BorutaPy(rf, n_estimators='auto', verbose=0, random_state=1)
    feat_selector.fit(df.drop(columns=depvar).values,df[depvar].values)
    
    df_boruta=pd.DataFrame({'column':df.drop(columns=depvar).columns.tolist(),
            'ranking':feat_selector.ranking_,'valid':feat_selector.support_ }).sort_values(by='ranking',ascending=True)
    
    return df_boruta,feat_selector.support_,feat_selector.ranking_


def runmodel(df,dropcols,reps,splits,model,depvar='Dementia',tree=1,plot_type='dot',featsfit=30,LRcheck=0,
            verbose=0,resize=1,resizeratio=20):
    
    if len(dropcols)>0:
        df_out=df.drop(dropcols,axis=1)
    else:
        df_out=df
    
        
    df_test_out=pd.DataFrame([])
    X_test_full=pd.DataFrame([])
    shap_values_full=np.asmatrix([])
    
    list_shap_values = list([])
    list_test_sets = list([]) 
    importance_df_full=pd.DataFrame([])
    importances_full=pd.DataFrame([])
    
    k=0   
    
    for reps in range(reps):
        if reps-round(reps/5)*5==0:
            print(reps)
        kf = KFold(n_splits=splits,shuffle=True)

        for train_index, test_index in kf.split(df_out):
            
            k=k+1
            df_train, df_test = df_out.iloc[train_index,: ], df_out.iloc[test_index, :]
            
            print(df_train[depvar].sum()/df_train.shape[0])
            
            df_score=df_test[['eid',depvar]]
            
            X=df_out.drop(columns=['eid',depvar])
            
            if resize==1:
                mask_disease=(df_train[depvar]==1)  
                df_train=pd.concat([df_train[mask_disease],df_train[~mask_disease].
                                    sample(len(df_train[mask_disease])*resizeratio)],axis=0)


            X_train, X_test = df_train.drop(columns=['eid',depvar]), df_test.drop(columns=['eid',depvar])
            y_train, y_test = df_train[depvar], df_test[depvar]

            mod=model.fit(X_train,y_train)   
        
            
            df_score['risk']=mod.predict_proba(X_test)[:, 1]
            df_score['y_pred']=mod.predict(X_test)
            df_score['y_test']=y_test.tolist()
            
            
            if tree==1:
                explainer = shap.TreeExplainer(model)
                expected_value = explainer.expected_value
                #shap_values_train = explainer.shap_values(X_train)
                shap_values = explainer.shap_values(X_test)
                #print("train SHAP")
                #shap.summary_plot(shap_values_train, X_train,max_display=30,plot_type=plot_type)
                
                print("Val SHAP")
                
                list_shap_values.append(shap_values)
                print(len(shap_values))
                list_test_sets.append(test_index)
                print(len(list_test_sets))
                
                if verbose==1:
                    print("SHAP for all variables")
                    shap.summary_plot(shap_values, X_test,max_display=20,plot_type=plot_type)
                
                
                
                shap_sign_sum=shap_values.mean(axis=0)
    
                shap_sum = np.abs(shap_values).mean(axis=0)
                
                importances = pd.DataFrame(data={'Attribute': X_train.columns,
                'Importance': mod.feature_importances_})
                importances = importances.sort_values(by='Importance', ascending=False)
                importances['iteration']=k
                
                #xgboost built in top features
                topcols=[col for col in X_train.columns if col in importances['Attribute'].head(featsfit).values]
                mod2=model.fit(X_train[topcols],y_train)
                
                if verbose==1:
                    print("SHAP for XGBoost Selection")
                    explainer = shap.TreeExplainer(mod2)
                    expected_value = explainer.expected_value
                    shap_values = explainer.shap_values(X_test[topcols])
                    shap.summary_plot(shap_values, X_test[topcols],max_display=30,plot_type=plot_type)
                
                if LRcheck==1:
                    model_lr = sklearn.linear_model.LogisticRegression(penalty="l2", C=0.1,max_iter=10000)
                    mod3=model_lr.fit(X_train[topcols],y_train)
                    df_score['risk_lr']=mod3.predict_proba(X_test[topcols])[:, 1]

                    
                    
                df_score['risk_xgb']=mod2.predict_proba(X_test[topcols])[:, 1]
                df_score['y_pred_xgb']=mod2.predict(X_test[topcols])
                
                
                if verbose==1:
                    figure(figsize=(15, 10), dpi=300)
                    sns.barplot(y='Attribute',x='Importance',data=importances.head(30),color="b")
                    plt.show()
        
        
                importance_df = pd.DataFrame([X_test.columns.tolist(), shap_sum.tolist(),shap_sign_sum.tolist()]).T
                importance_df.columns = ['column_name', 'shap_importance','shap_sign_importance']
                importance_df['shap_importance']=pd.to_numeric(importance_df['shap_importance'])
                importance_df['shap_sign_importance']=pd.to_numeric(importance_df['shap_sign_importance'])
                importance_df = importance_df.sort_values('shap_importance', ascending=False)
                importance_df['iteration']=k
                importance_df['rank']=np.arange(len(importance_df))
                #list_shap_values.append(shap_values)
                #list_test_sets.append(test_index)
                importance_df_full=pd.concat([importance_df_full,importance_df],axis=0)
                
                importances_full=pd.concat([importances_full,importances],axis=0)
                
                #shap top features
                topcols2=[col for col in X_train.columns if col in
                          importance_df['column_name'].head(featsfit).values]
                mod3=model.fit(X_train[topcols2],y_train)
                df_score['risk_shap']=mod3.predict_proba(X_test[topcols2])[:, 1]
                df_score['y_pred_shap']=mod3.predict(X_test[topcols2])
                
            else:
                print(np.exp(mod.coef_))
                
                importances = pd.DataFrame(data={'Attribute': X_train.columns,
                                                 'Odds Ratio': np.exp(mod.coef_[0]),'Importance': abs(mod.coef_[0])})
                importances = importances.sort_values(by='Importance', ascending=False)
                
                
                figure(figsize=(15, 10), dpi=300)
                sns.barplot(y='Attribute',x='Odds Ratio',data=importances,color="b")
                plt.show()
                #print(importances)

                #log_reg = sm.Logit(y_train, X_train).fit()
                #print(log_reg.summary())
                
                print("later")
            df_test_out=pd.concat([df_test_out,df_score])
    #shapplot(list_shap_values,list_test_sets)
    
    
    if tree==1:
        xgb_FI=pd.DataFrame(importances_full.groupby('Attribute')['Importance'].mean()).reset_index().\
sort_values(by='Importance',ascending=False)

        shap_FI=importance_df_full.groupby('column_name').\
agg({'shap_importance':'mean','shap_sign_importance':'mean'}).reset_index()\
.sort_values(by='shap_importance',ascending=False)


        #take top 200 features from each feature selection method
        cols=list(set(list((shap_FI['column_name'].head(200)))+list(xgb_FI['Attribute'].head(200))))
        k=len(list_shap_values) 
        shapvals=np.concatenate([list_shap_values[i][:, [X.columns.get_loc(col) for col in cols]] for i in range(k)])

        X2=X.iloc[:, [X.columns.get_loc(col) for col in cols]] 
        df_list2=[X2.iloc[list_test_sets[i],: ] for i in range(k)]
        colvals = pd.concat(df_list2, axis=0) 
        
        return df_test_out,shap_FI,xgb_FI,shapvals,colvals,X
        
       

    else:
        return df_test_out
    

def shapplot(list_test_sets,list_shap_values,X):
    test_set = list_test_sets[0]
    shap_values = np.array(list_shap_values[0])
    for i in range(0,len(list_test_sets)):#maybe put -1 here to remove last one
        test_set = np.concatenate((test_set,list_test_sets[i]),axis=0)
        shap_values = np.concatenate((shap_values,np.array(list_shap_values[i])),axis=0)
    #bringing back variable names
    X_test = pd.DataFrame(np.asmatrix(X)[test_set],columns=X.columns)
    print("SHAP summary dot plot for selected feature number")
    shap.summary_plot(shap_values, X_test,max_display=30,plot_type='dot')   
    
def meanimp(df):
    for col in df.columns:
        if df[col].dtype=="uint8" or df[col].dtype=="float64":
            df[col][pd.isnull(df[col])]=df[col][pd.notnull(df[col])].mean()
    return df

def plot_ROCAUC(y_test, y_score,figname='Fig2aAUCROC'):
    
    
    fpr, tpr, _ = roc_curve(y_test,y_score)
    
    colors = ['darksalmon', 'gold'] 
    
    mean_auc = auc(fpr, tpr)
  
    print ("AUC Score (Train): %f" % metrics.roc_auc_score(y_test, y_score))


    plt.plot(fpr, tpr, 'black', alpha = 0.8,
             label=r'%s (AUC = %0.2f)' % (1,mean_auc))
 
    #plt.plot([0, 1], [0, 1], linestyle = '--', lw = 2, color = 'r', label = 'Luck', alpha= 0.8)
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.legend(loc="lower right", fontsize=14)
    plt.xticks(fontsize='18')
    plt.yticks(fontsize='18')
    
    plt.savefig(figname+'.svg', dpi=300)
    plt.show()
    
def plot_ROCAUC_mult(y_test, y_score,y_test1,y_score1,y_test2,y_score2,l1,l2,l3,figname='Fig2aAUCROC'):
    
    
    fpr, tpr, _ = roc_curve(y_test,y_score)
    fpr1, tpr1, _ = roc_curve(y_test1,y_score1)
    fpr2, tpr2, _ = roc_curve(y_test2,y_score2)
    
    colors = ['darksalmon', 'gold', 'royalblue', 'mediumseagreen', 'violet'] 
    
    mean_auc = auc(fpr, tpr)
    mean_auc1 = auc(fpr1, tpr1)
    mean_auc2 = auc(fpr2, tpr2)
   
    
    print ("AUC Score (Train): %f" % metrics.roc_auc_score(y_test, y_score))


    plt.plot(fpr, tpr, 'black', alpha = 0.8,
             label=r'%s (AUC = %0.2f)' % (l1,mean_auc))
    plt.plot(fpr1, tpr1, 'red', alpha = 0.8,
             label=r'%s APOE4+ve (AUC = %0.2f)' % (l2,mean_auc1))
    plt.plot(fpr2, tpr2, 'yellow', alpha = 0.8,
             label=r'%s APOE4-ve (AUC = %0.2f)' % (l3,mean_auc2))

    #plt.plot([0, 1], [0, 1], linestyle = '--', lw = 2, color = 'r', label = 'Luck', alpha= 0.8)
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.legend(loc="lower right", fontsize=14)
    plt.xticks(fontsize='18')
    plt.yticks(fontsize='18')

    plt.savefig(figname+'.svg', dpi=300)
    plt.show()
    
def ABS_SHAP(df_shap,df,max_disp=20):
    #import matplotlib as plt
    # Make a copy of the input data
    shap_v = pd.DataFrame(df_shap)
    feature_list = df.columns
    shap_v.columns = feature_list
    df_v = df.copy().reset_index().drop('index',axis=1)
    
    # Determine the correlation in order to plot with different colors
    corr_list = list()
    for i in feature_list:
        b = np.corrcoef(shap_v[i],df_v[i])[1][0]
        corr_list.append(b)
    corr_df = pd.concat([pd.Series(feature_list),pd.Series(corr_list)],axis=1).fillna(0)
    # Make a data frame. Column 1 is the feature, and Column 2 is the correlation coefficient
    corr_df.columns  = ['Variable','Corr']
    corr_df['Sign'] = np.where(corr_df['Corr']>0,'red','blue')
    
    # Plot it
    shap_abs = np.abs(shap_v)
    k=pd.DataFrame(shap_abs.mean()).reset_index()
    k.columns = ['Variable','SHAP_abs']
    k2 = k.merge(corr_df,left_on = 'Variable',right_on='Variable',how='inner')
    k2 = k2.sort_values(by='SHAP_abs',ascending = True)
    k2=k2.tail(max_disp)
    colorlist = k2['Sign']
    ax = k2.plot.barh(x='Variable',y='SHAP_abs',color = colorlist, figsize=(5,6),legend=False)
    ax.set_xlabel("SHAP Value (Red = Positive Impact)")
    return k2