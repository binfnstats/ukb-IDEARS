import pandas as pd
import numpy as np
import re
import statsmodels.api as sm
import seaborn as sns
import pickle

import warnings
warnings.filterwarnings('ignore')

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

from time import time
from datetime import datetime # Current date time in local system 

from mpl_toolkits.mplot3d import Axes3D

import sklearn
from sklearn.metrics import roc_curve, auc,classification_report
from sklearn.model_selection import train_test_split,KFold

from sklearn.preprocessing import QuantileTransformer
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.inspection import partial_dependence,plot_partial_dependence

import shap
from xgboost import XGBClassifier,plot_importance

class IDEARs_funcs(object):

	"""
	Class of functions to run XGBoost and SHAP on UKBiobank Data as part of the IDEARS pipeline
	"""
	
	def __init__(self):
		"""
		Initilising models.
		"""
		self.path="/Users/michaelallwright/Dropbox (Sydney Uni)/michael_PhD/Projects/UKB/Data/"
		self.path_figures= "/Users/michaelallwright/Documents/GitHub/UKB/PD/figures/"


		self.config = dict(scale_pos_weight = 6,subsample = 1, min_child_weight = 5, max_depth = 5, gamma= 2, 
				  colsample_bytree= 0.6,smote=1,reps=2)

		self.holdout_ratio=0.2
		self.agemin=50
		self.agemax=70
	
		self.rundate=datetime.date(datetime.now())

		self.wordsremovePD='inpatient_record|patient_polymorph|time_since_interview|_HES|records_in_hes|treatment_speciality|\
	Diag_PD|Age_Diag_Dementia|Age_Diag_PD|  Parkinson|interviewer|date_of_attending_assessment_centre_f53|years_after_dis|\
	Frontotemporal|daysto|hospital_recoded|from_hospital|Age_Today|year_of_birth|pollution_|pesticide_exposure|\
	parental_ad_status_|birth_weight|parkins|sex_inference|sample_dilut|samesex|mobile_phone|inflammation|frail|\
	admission_polymorphic|faster_mot|drive_faster_than|time_to_complete_round|Genotype|genetic_principal|employment|Free-text|xxxx' 

		self.variablemap=dict({'testosterone_f30850_0_0':'testosterone',
		'age_when_attended_assessment_centre_f21003_0_0':'age at baseline',
		'parental_pd':'parent with PD',
		'neutrophill_percentage_f30200_0_0':'neutrophill percentage',
		'igf1_f30770_0_0':'IGF1',
		'suffer_from_nerves_f2010_0_0':'suffer from nerves',
		'avg_duration_to_first_press_of_snapbutton_in_each_round':'average duration to first press of snap button',
		'neutrophill_lymphocyte_ratio':'neutrophill to lymphocyte count ratio',
		'creactive_protein_f30710_0_0':'c-reactive protein',
		'Retired':'retired at baseline',
		'triglycerides_f30870_0_0':'triglycerides',
		'creatinine_enzymatic_in_urine_f30510_0_0':'creatine enzymatic in urine',
		'total_bilirubin_f30840_0_0':'bilirubin',
		'cholesterol_f30690_0_0':'cholesterol',
		'apolipoprotein_a_f30630_0_0':'apoplipoprotein A',
		'glycated_haemoglobin_hba1c_f30750_0_0':'glycated haemoglobin',
		'creatinine_f30700_0_0':'creatine',
		'vitamin_d_f30890_0_0':'vitamin D',
		'platelet_crit_f30090_0_0':'platelet crit',
		'number_of_treatmentsmedications_taken_f137_0_0':'number of treatments or medictions',
		'hip_circumference_f49_0_0':'hip circumference',
		'usual_walking_pace_f924_0_0':'usual walking pace',
		'AST_ALT_ratio':'AST:ALT ratio',
		'Total ICD10 Conditions at baseline':'Total ICD10 Conditions at baseline',
		'waist_circumference_f48_0_0':'waist circumference',
		'sex_f31_0_0':'gender',
		'forced_vital_capacity_fvc_f3062_0_0':'forced vital capacity',
		'standing_height_f50_0_0':'height',
		'mean_reticulocyte_volume_f30260_0_0':'mean reticulocyte volume',
		'hand_grip_strength_left_f46_0_0':'hand grip strength (left)',
		'lymphocyte_count_f30120_0_0':'lymphocte count',
		'chest_pain_or_discomfort_f2335_0_0':'chest pain or discomfort',
		'platelet_count_f30080_0_0':'platelet count',
		'alanine_aminotransferase_f30620_0_0':'alanine aminotransferase',
		'hand_grip_strength_right_f47_0_0':'hand grip strength (right)'}) 

	def holdout_data(self,df,agemin=50,agemax=70,depvar='dementia',apoe=3):
		mask_age=(df['age_when_attended_assessment_centre_f21003_0_0']>=agemin)&(df['age_when_attended_assessment_centre_f21003_0_0']<=agemax)

		df=self.maskapoedf(df[mask_age],apoe=apoe)
		df=self.meanimp(df)

		mask=(df[depvar]==1)
		print('Total dementia in data '+str(sum(mask)))
		df_val=pd.concat([df[mask].sample(round(self.holdout_ratio*df[mask].shape[0])),df[~mask].sample(round(self.holdout_ratio*df[~mask].shape[0]))],axis=0)
		mask_val=(df['eid'].isin(df_val['eid']))
		df_train=df[~mask_val]

		return df_train,df_val


	def model(self):
		"""
		Model parameters
		"""
		mod_xgb=XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
		   colsample_bynode=1, learning_rate=0.1,
		   max_delta_step=0,  missing=None, 
		   n_estimators=60, n_jobs=4, nthread=4, objective='binary:logistic',
		   random_state=0, reg_alpha=0, reg_lambda=1, scale_pos_weight=self.config['scale_pos_weight'],
		   min_child_weight=self.config['min_child_weight'],
		   gamma=self.config['gamma'], colsample_bytree=self.config['colsample_bytree'],max_depth=self.config['max_depth'],
		   seed=42, silent=None, subsample=1, verbosity=1,eval_metric='auc')

		return mod_xgb

	def mapvar(self,x):
	    if x in self.variablemap:
	        x=self.variablemap[x]
	    return x

	def findcols(self,df,string):

		"""
		helper function for columns
		"""

		return [col for col in df.columns if string in col]

	def quantile_vars(self,df:pd.DataFrame,quant=5):

		for var in df.columns:

			if df[var].nunique()>quant and re.search('float|int',str(df[var].dtype)):
				df[var]=pd.qcut(df[var],quant,labels=False,duplicates='drop')
			else:
				print('insufficient variable values')
		return df

   

	def maskapoedf(self,df,apoe=1):

		"""
		function to choose APOE4 subsets for analysis
		"""

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

		
	def col_spec_chars(self,df):

		"""
		function to clean column names of bad chars
		"""

		df.columns=df.columns.str.replace(',','_')
		df.columns=df.columns.str.replace('<','_')
		df.columns=df.columns.str.replace('>','_')
		df.columns=df.columns.str.replace('[','_')
		df.columns=df.columns.str.replace(']','_')
		return df

	def shapplot(self,list_shap_values,list_test_sets):

		"""
		plot shap for model outputs
		"""

		test_set = list_test_sets[0]
		shap_values = np.array(list_shap_values[0])
		for i in range(1,len(list_test_sets)):
			test_set = np.concatenate((test_set,list_test_sets[i]),axis=0)
			shap_values = np.concatenate((shap_values,np.array(list_shap_values[i])),axis=1)
		#bringing back variable names    
		X_test = pd.DataFrame(X[test_set],columns=columns)
		shap.summary_plot(shap_values[1], X_test)
		

	def borutafeats(self,df):
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


	def rebalance(self,df,depvar,resizeratio=1):
		mask_disease=(df[depvar]==1)  
		df_out=pd.concat([df[mask_disease],df[~mask_disease].sample(len(df[mask_disease])*resizeratio)],axis=0)
		return df_out


	def borutarun(self,df,depvar,resizeratio=1):
		
		df=self.rebalance(df,depvar,resizeratio)
		
		print(df.shape)
		rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5)
		feat_selector = BorutaPy(rf, n_estimators='auto', verbose=0, random_state=1)
		feat_selector.fit(df.drop(columns=depvar).values,df[depvar].values)
		
		df_boruta=pd.DataFrame({'column':df.drop(columns=depvar).columns.tolist(),
				'ranking':feat_selector.ranking_,'valid':feat_selector.support_ }).sort_values(by='ranking',ascending=True)
		
		return df_boruta,feat_selector.support_,feat_selector.ranking_

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
		


	def runmodel(self,df,dropcols,reps,splits,model,depvar='dementia',tree=1,plot_type='dot',featsfit=30,LRcheck=0,
				verbose=0,resize=1,resizeratio=20):
		
		if len(dropcols)>0:
			df_out=df.drop(dropcols,axis=1)
		else:
			df_out=df
			
		X = df_out.drop(columns=['eid',depvar])
		y = df_out[depvar]
		
		mod=model.fit(X,y) 
		
			
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
					
					
					figure(figsize=(15, 25), dpi=300)
					sns.barplot(y='Attribute',x='Odds Ratio',data=importances,color="b")
					
					plt.show()
					#print(importances)

					#log_reg = sm.Logit(y_train, X_train).fit()
					#print(log_reg.summary())
					
					print("later")
				df_test_out=pd.concat([df_test_out,df_score])
				importances_full=pd.concat([importances_full,importances])
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
			xgb_FI=pd.DataFrame(importances_full.groupby('Attribute')['Importance'].mean()).reset_index().\
	sort_values(by='Importance',ascending=False)


			return df_test_out,importances
		

	def simpletrain(self,df,model,dropcols,depvar,wordsremove,resizeratio=20,shapshow=1):
		
		df_out=self.preprocess(df,dropcols,depvar,wordsremove,resizeratio)
		
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

	def runmodels(self,df,depvar,reps,splits,drops,wordsremove,model,
				 featsfit=30,LRcheck=1,resizeratio=20,agemin=50,agemax=70,verbose=0,apoe=2,tree=1,
				 plot_type='dot',agerun=1):
		
		if agerun==1:
		
			mask_age=(df['age_when_attended_assessment_centre_f21003_0_0']>=agemin)&\
	(df['age_when_attended_assessment_centre_f21003_0_0']<=agemax)
			df=df[mask_age]
		df=self.maskapoedf(df,apoe=apoe)
		df=self.col_spec_chars(df)
		
		print('%s%s%s%s' % ("Total ",depvar," ",df[depvar].sum()))
		dropvars=[col for col in df.columns if col in drops or re.search(wordsremove,col)]
		df=self.meanimp(df)
		outputs=self.runmodel(df=df,dropcols=dropvars,reps=reps,splits=splits,model=model,\
		depvar=depvar,featsfit=featsfit,LRcheck=LRcheck,resizeratio=resizeratio,verbose=verbose,tree=tree,
						plot_type=plot_type)
		
		return outputs

	def shapplot(self,list_test_sets,list_shap_values,X):

		"""
		shap summary plot given lists of test sets and shap values
		"""

		test_set = list_test_sets[0]
		shap_values = np.array(list_shap_values[0])
		for i in range(0,len(list_test_sets)):#maybe put -1 here to remove last one
			test_set = np.concatenate((test_set,list_test_sets[i]),axis=0)
			shap_values = np.concatenate((shap_values,np.array(list_shap_values[i])),axis=0)
		#bringing back variable names
		X_test = pd.DataFrame(np.asmatrix(X)[test_set],columns=X.columns)
		print("SHAP summary dot plot for selected feature number")
		shap.summary_plot(shap_values, X_test,max_display=30,plot_type='dot')   
		
	def meanimp(self,df):

		"""
		Simple mean imputation
		"""

		for col in df.columns:
			if df[col].dtype=="uint8" or df[col].dtype=="float64":
				df[col][pd.isnull(df[col])]=df[col][pd.notnull(df[col])].mean()
		return df

	
		
		
	def ABS_SHAP(self,df_shap,df,max_disp=20,figx=10,figy=10,dpi=200,figname='shap_bar',format_file='.jpg'):

		"""
		SHAP bar with colours for directions
		"""
	   
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
		
		figure(figsize=(figx, figy), dpi=dpi)
		matplotlib.rc('xtick', labelsize=20) 
		matplotlib.rc('ytick', labelsize=20)
		
		ax = k2.plot.barh(x='Variable',y='SHAP_abs',color = colorlist,legend=False,figsize=(figx, figy))
		ax.set_xlabel("SHAP Value (Red = Positive Impact)")

		
		for i, v in enumerate(list(k2['SHAP_abs'])):
			ax.text(v + 0.01, i, str(round(v,4)), color='black', fontweight='bold', fontsize=14, ha='left', va='center')


		plt.savefig(self.path_figures+figname+format_file, dpi=300,bbox_inches='tight')
		plt.show()

		return k2

	def varsplit_shap(self,df_shap,shap_values,var,figname="check"):

		for v in list(df_shap[var].unique()):
			df=df_shap[(df_shap[var]==v)]
			sv=shap_values[(df_shap[var]==v)]

			self.ABS_SHAP(sv, df,25,figx=15,figy=15,figname=figname+' '+str(v))

	def run_entire_data_pd(self,df,drops,wordsremove,runs=5,outfile='shap_tuple.pkl'):

		shap_values_list=[]
		X_list=[]
		df_out_list=[]

		for i in range(runs):
			print(i)
			df_train,df_val=self.holdout_data(df=df,agemin=50,agemax=70,depvar='PD')
			mod1=self.simpletrain(df=df_train,model=self.model(),dropcols=drops,
				wordsremove=wordsremove,depvar='PD',resizeratio=100,shapshow=0)
			shap_values, X, df_out=self.simple_eval(df=df_val,model=mod1,dropcols=drops,
				wordsremove=wordsremove,depvar='PD',resize=0,resizeratio=20,
			shapshow=1)

			shap_values_list.append(shap_values)
			X_list.append(X)
			df_out_list.append(df_out)

		shap_tuple=[shap_values_list,X_list,df_out_list]
		shap_tuple_file=open(self.path+outfile,'wb')
		pickle.dump(shap_tuple,shap_tuple_file)
		shap_tuple_file.close()

		return shap_tuple

	def run_entire_data_dem(self,df,drops,wordsremove,runs=5,outfile='shap_tuple_dem.pkl'):

		
		shap_values_list=[]
		X_list=[]
		df_out_list=[]

		for i in range(runs):
			print(i)
			df_train,df_val=self.holdout_data(df=df,agemin=55,agemax=70,depvar='dementia',apoe=2)
			mod1=self.simpletrain(df=df_train,model=self.model(),dropcols=drops,
				wordsremove=wordsremove,depvar='dementia',resizeratio=100,shapshow=0)
			shap_values, X, df_out=self.simple_eval(df=df_val,model=mod1,dropcols=drops,
				wordsremove=wordsremove,depvar='dementia',resize=0,resizeratio=20,
			shapshow=1)

			shap_values_list.append(shap_values)
			X_list.append(X)
			df_out_list.append(df_out)

		shap_tuple=[shap_values_list,X_list,df_out_list]
		shap_tuple_file=open(self.path+outfile,'wb')
		pickle.dump(shap_tuple,shap_tuple_file)
		shap_tuple_file.close()

		return shap_tuple

	def shapgraphs_tuple(self,tuple,max_disp=20,figname='shap_chart_for..'):
		df=pd.concat(tuple[1],axis=0)
		df.columns=[self.mapvar(c) for c in df.columns]

		print(df.shape)
		df_shap=np.vstack(tuple[0])
		print(df_shap.shape)
		outs=self.ABS_SHAP(df_shap,df,max_disp=max_disp,figx=10,figy=15,dpi=200,figname=figname)
		return outs

	def ROCAUC_tuples(self,df_out_list,labels,cols,figname='ROCAUC for..',format_out='.svg'):
		dfs=[pd.concat(df_out_list[i],axis=0) for i in range(len(df_out_list))]
		y_tests=[dfs[i]['y_test'] for i in range(len(df_out_list))]
		risks=[dfs[i]['risk'] for i in range(len(df_out_list))]
		self.plot_ROCAUC_mult(y_tests,risks,labels,cols,figname=figname,format_out=format_out)




	def plot_ROCAUC_mult(self,y_test,y_score,labels,cols ,figname='check',figx=6,figy=4,format_out='.svg'):

		"""
		Plot multiple ROCAUC graphs next to each other and output as an svg figure
		"""
	
		figure(figsize=(figx, figy), dpi=200)
		
		d = dict()
		aucs=[]
		for i,x in enumerate(y_test):
			fpr, tpr, _ = roc_curve(y_test[i],y_score[i])
			mean_auc=auc(fpr, tpr)
			d["fpr{0}".format(i)] = fpr
			d["tpr{0}".format(i)] = tpr
			d["meanauc{0}".format(i)]= mean_auc
			plt.plot(fpr, tpr, cols[i], alpha = 0.8,label=r'%s (AUC = %0.3f)' % (labels[i],mean_auc))
			aucs.append(mean_auc)
			
		plt.xlim([-0.01, 1.01])
		plt.ylim([-0.01, 1.01])
		plt.ylabel('True Positive Rate', fontsize=14)
		plt.xlabel('False Positive Rate', fontsize=14)
		plt.legend(loc="lower right", fontsize=10)
		plt.xticks(fontsize='18')
		plt.yticks(fontsize='18')
		
		plt.savefig(self.path_figures+figname+format_out, dpi=300,bbox_inches='tight')
		plt.show()
			
		return aucs

	def agenorm2(self,df,var):

		"""
		age normalisations
		"""

		df_sum=pd.DataFrame(df.groupby(['age_when_attended_assessment_centre_f21003_0_0']).agg({var:['mean']})).reset_index()
		df_sum.columns=['age_when_attended_assessment_centre_f21003_0_0','mean'+var]

		df=pd.merge(df,df_sum,on='age_when_attended_assessment_centre_f21003_0_0',how='left')
		df[var]=df[var]/df['mean'+var]
		df.drop(columns=['mean'+var],inplace=True)
		return df


