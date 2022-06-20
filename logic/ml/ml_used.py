import os

import sys


sys.path.append("../../../ukb-dementia-shap/")


sys.path.append("../Pain/code/")
from logic.data_processing.data_import import dataload
from logic.data_processing.data_processing import data_proc_main
from logic.analysis.analysis import AnalysisCharts
from logic.ml.classification_shap import IDEARs_funcs
from ukb_utils.utils import basic_funcs

import matplotlib.pyplot as plt
import seaborn as sns
from pandas.tseries.offsets import DateOffset
import datetime as dt

import pandas as pd
import numpy as np
from scipy import stats
import re

class ml_funcs():

    """
    This class is to extract key information from XMLDoc text and return as a set of dataframes
    """

    def __init__(self) -> None:

    	


    def prep_data(self,df,eids_in,eids_inc_depvar,eids_exc_depvar,depvar):
	    mask=(df['eid'].isin(eids_in))&~(df['eid'].isin(eids_exc_depvar))
	    df_out=df.loc[mask,]

	    df_out[depvar]=0
	    df_out.loc[df_out['eid'].isin(eids_inc_depvar),depvar]=1
	    
    	return df_out

    def shap_multy(self,df,figname='test',depvar='PD',resize=1,resizeratio=10,meanshapmin=0.03,minrecs=3,rank=25,runs=10,barplots=1,
              holdout_ratio=0.5,df_val=pd.DataFrame([])):
    
	    feats_full=[]
	    aucs_full=[]
	    for i in range(runs):
	        rets=ds.shapruns_new(df=df,run='test',\
	    remwords=ml.wordsremovePD+'|dementia',depvar=depvar,resize=resize,resizeratio=resizeratio,perc=True,barplots=barplots,
	                            holdout_ratio=holdout_ratio,df_val=df_val)

	        feats=ml.shapgraphs_tuple(rets['shaps'],max_disp=30,figname='SHAP test',plot=False)
	        feats['Variable2']=feats['Variable']
	        feats['Variable']=feats['Variable'].map(ds.varmap)
	        feats.loc[pd.isnull(feats['Variable']),'Variable']=feats['Variable2']
	        feats_full.append(feats)
	        aucs_full.append(rets['aucs'])

	     return feats_full,rets,aucs_full

	def shapruns_new(self,run,df,remwords='diabetes|H360|total_dis',depvar='polyneuropathy',resizeratio=5,resize=1,perc=False,
        compvars=None,stream=False,runs=2,barplots=1,holdout_ratio=0.5,df_val=pd.DataFrame([])):
        
        shap_obj=self.process_run(df=df,depvar=depvar,resize=resize,resizeratio=resizeratio,remwords=remwords,runs=runs,
            holdout_ratio=holdout_ratio,df_val=df_val)

        print(len(shap_obj))
        feats_all=ml.shapgraphs_tuple(shap_obj,max_disp=30,figname='SHAP IDEARS '+run+self.date_suff,stream=stream)
        aucs=ml.ROCAUC_tuples(df_out_list=[shap_obj[2]],labels=['IDEARS - all'],cols=['blue'],figname='ROCAUC '+run+self.date_suff,stream=stream)

        rets=dict({'feats_all':feats_all,'shaps':shap_obj,'aucs':aucs})
        if barplots==1:
            data_sum=self.runplots_static(df=df,depvar=depvar,fig_name='Inflamm boxplots '+run+self.date_suff,perc=perc,compvars=compvars)
            rets['data_sum']=data_sum
        return rets

     def data_clean(self,df,depvar='neuropathy',remwords='xxxxxxx'):

        df=ml.col_spec_chars(df=df)
        df=df.loc[pd.notnull(df[depvar]),]

        dropvars=list(set([c for c in df.columns if  re.search(ml.wordsremovePD,c)]+[c for c in df.columns if  re.search(remwords,c)]))

        return df,dropvars



    def process_run(self,df,depvar='neuropathy',resize=1,remwords='xxxxxxx',resizeratio=20,runs=2,holdout_ratio=0.5,df_val=pd.DataFrame([])):

        df,dropvars=self.data_clean(df,depvar=depvar,remwords=remwords)
        
        shap_tuple=ml.run_entire_data_pd(df=df,drops=dropvars,wordsremove='consultant',outfile='test_pain',savefile=False,
        save_featslist=False,runs=runs,depvar=depvar,agemin=10,agemax=90,resize=resize,holdout_ratio=holdout_ratio,
        resizeratio=resizeratio,verbose=False,df_val=df_val)
        
        return shap_tuple

    def run_entire_data(self,df,drops,wordsremove,outfile,savefile=False,
		save_featslist=True,runs=5,holdout_ratio=0.2,depvar='PD',agemin=50,agemax=70,resize=1,resizeratio=20,verbose=False):

		#shap_values_list=[]
		X_list=[]
		df_out_list=[]

		for i in range(runs):
			if verbose:
				print(i)
			df_train,df_val=self.holdout_data(df=df,agemin=agemin,agemax=agemax,depvar=depvar,holdout_ratio=holdout_ratio)
			mod1=self.simpletrain(df=df_train,model=self.model(),dropcols=drops,
				wordsremove=wordsremove,depvar=depvar,resizeratio=resizeratio,shapshow=0,resize=resize)
			if verbose:
				print("trained")
			shap_values, X, df_out=self.simple_eval(df=df_val,model=mod1,dropcols=drops,
				wordsremove=wordsremove,depvar=depvar,resize=resize,resizeratio=resizeratio,
			shapshow=1)

			if verbose:
				print("shap done")
			shap_values_list.append(shap_values)
			X_list.append(X)
			df_out_list.append(df_out)

		shap_tuple=[shap_values_list,X_list,df_out_list]

		if savefile:
			shap_tuple_file=open(self.path+outfile,'wb')
			pickle.dump(shap_tuple,shap_tuple_file)
			shap_tuple_file.close()

		if save_featslist:
			feats_all=self.shapgraphs_tuple(shap_tuple,max_disp=20,figname=outfile)
			feats_all.to_parquet(self.path+outfile+'.parquet')
		if verbose:
			print('completed run entire data')
		return shap_tuple


	def holdout_data(self,df,agemin=50,agemax=70,depvar='dementia',apoe_filt=True,apoe=3,holdout_ratio=0.2):
		mask_age=(df['age_when_attended_assessment_centre_f21003_0_0']>=agemin)&(df['age_when_attended_assessment_centre_f21003_0_0']<=agemax)

		if apoe_filt:
			df=self.maskapoedf(df[mask_age],apoe=apoe)
		df=self.meanimp(df)

		mask=(df[depvar]==1)
		print('Total '+depvar+' in data: '+str(sum(mask)))
		df_val=pd.concat([df[mask].sample(round(holdout_ratio*df[mask].shape[0])),df[~mask].sample(round(holdout_ratio*df[~mask].shape[0]))],axis=0)
		mask_val=(df['eid'].isin(df_val['eid']))
		df_train=df[~mask_val]

		return df_train,df_val


	def simpletrain(self,df,model,dropcols,depvar,wordsremove,resize,resizeratio=20,shapshow=1):
		
		df_out=self.preprocess(df,dropcols,depvar,wordsremove,resize,resizeratio)
		
		X=df_out.drop(columns=['eid',depvar])
		y=df_out[depvar]
		 
		mod=model.fit(X,y)   
		
		if shapshow==1:
			explainer = shap.TreeExplainer(mod)
			expected_value = explainer.expected_value
			shap_values = explainer.shap_values(X)

			print("SHAP for all variables")
			self.ABS_SHAP(shap_values,X)
			shap.summary_plot(shap_values, X,max_display=20,plot_type='dot')
			plt.show()

		return mod

	def simple_eval(self,df,model,dropcols,depvar,wordsremove,resize,resizeratio,shapshow=1):
		
		df_out=self.preprocess(df,dropcols,depvar,wordsremove,resize,resizeratio)
		
		X=df_out.drop(columns=['eid',depvar])
		y=df_out[depvar]
		
		df_out['risk']=model.predict_proba(X)[:, 1]
		df_out['y_pred']=model.predict(X)
		df_out['y_test']=y.tolist()

		if shapshow==1:
			explainer = shap.TreeExplainer(model)
			expected_value = explainer.expected_value
			shap_values = explainer.shap_values(X)

			return shap_values, X, df_out
		else:
			return df_out

	def preprocess(self,df,dropcols,depvar,wordsremove,resize=1,resizeratio=20):
		
		df=self.col_spec_chars(df)
		
		dropvars=[col for col in df.columns if col in dropcols or re.search(wordsremove,col)]
		if len(dropcols)>0:
			df_out=df.drop(dropvars,axis=1)
		else:
			df_out=df
			
		if resize==1:
		
			mask_disease=(df_out[depvar]==1)  
			df_out=pd.concat([df_out[mask_disease],df_out[~mask_disease].sample(len(df_out[mask_disease])*resizeratio)],axis=0)

		
			
		return df_out
			

