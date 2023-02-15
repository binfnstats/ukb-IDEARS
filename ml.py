
"""
Module to 

1. Apply boosting methods to select features
2. Run logistic regression and take odds ratios of feature sets
3. Validate and return AUCs, precision scores etc.

"""


import numpy as np
import pandas as pd
import re
import sys
import matplotlib.pyplot as plt
import os

import miceforest as mf

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score

import lightgbm as lgb
from xgboost import XGBClassifier,plot_importance
from sklearn.ensemble import RandomForestClassifier


from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC

from sklearn import linear_model 

from sklearn.preprocessing import StandardScaler

import shap
from boruta import BorutaPy

from scipy.stats import norm
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

import seaborn as sns




class ml_funcs(object):

	def __init__(self):

		self.path='/Users/michaelallwright/Documents/data/ukb/'

		self.remwords=['Polymorphic','dementia','driving','eid','length_of_mobile_phone_use_f1110_0_0',\
'intercourse','job_code','date','records_in_hes','_speciality','death','sample_dilution','hospital_recoded','uk_biobank_assessment_centre',
'number_of_blood_samples_taken','ordering_of_blows','treatmentmedication_code','spirometry','carer_support_indicators',
'willing_to_attempt','patient_recoded','attendancedisability','cervical_smear_test']

		#xgb classifier model
		self.mod_xgb_base=XGBClassifier()

		#POS WEIGHT CHANGE
		self.mod_xgb=XGBClassifier(base_score=0.5, booster='gbtree',  scale_pos_weight=1,colsample_bylevel=1,\
colsample_bynode=1, learning_rate=0.1,max_delta_step=0,  missing=1, n_estimators=60, n_jobs=4, \
nthread=4, objective='binary:logistic',random_state=0, reg_alpha=0, reg_lambda=1,\
min_child_weight=5,gamma=2, colsample_bytree=0.6,max_depth=5,seed=42, silent=None, subsample=1,\
verbosity=1,eval_metric='auc')


		#light GBM model with parameters
		self.lgbm_mod = lgb.LGBMClassifier(max_bin= 500,learning_rate= 0.05,boosting_type= 'gbdt',objective= 'binary',\
metric= 'auc',num_leaves= 10,verbose= -1,min_data= 1000,boost_from_average= True)

		#random forest classifier
		self.mod_rf = RandomForestClassifier(max_depth=5, random_state=0)

		#support vector machine
		self.mod_svm=make_pipeline(StandardScaler(), SVC(gamma='auto',probability=True))

		#logistic regression model
		self.log_reg = linear_model.LogisticRegression(max_iter=10000)
		self.log_reg = linear_model.LogisticRegression(max_iter=10000,solver="saga",penalty='elasticnet',l1_ratio=0.3)


		self.all_known_cols=['dementia','age_when_attended_assessment_centre_f21003_0_0','APOE4_Carriers',\
'pollution','sedentary_time','diabetes_diagnosed_by_doctor_f2443_0_0','low_activity','salad_raw_vegetable_intake_f1299_0_0',\
'fresh_fruit_intake_f1309_0_0','weight_change_compared_with_1_year_ago_f2306_0_0',\
'frequency_of_tiredness_lethargy_in_last_2_weeks_f2080_0_0','ipaq_activity_group_f22032_0_0',
'usual_walking_pace_f924_0_0','hand_grip_strength_left_f46_0_0','hand_grip_strength_right_f47_0_0','body_mass_index_bmi_f21001_0_0',\
'systolic_blood_pressure_automated_reading_f4080_0_0','diastolic_blood_pressure_automated_reading_f4079_0_0',\
'frailty_score','smoking_status_f20116_0_0','cholesterol_f30690_0_0','hdl_cholesterol_f30760_0_0',\
'processed_meat_intake_f1349_0_0','mean_time_to_correctly_identify_matches_f20023_0_0',\
'number_of_incorrect_matches_in_round_f399_0_2','sex_f31_0_0','hypertension','ever_smoked_f20160_0_0','alcohol','TBI',\
'Hear_loss','Qualif_Score']

		self.livingstone_cols=['sex_f31_0_0','age_when_attended_assessment_centre_f21003_0_0','APOE4_Carriers','TBI','hearing_difficultyproblems_f2247_0_0',\
		'alcohol','pollution','hypertension','diabetes_diagnosed_by_doctor_f2443_0_0','Hear_loss','ever_smoked_f20160_0_0','body_mass_index_bmi_f21001_0_0',\
		'depressed','smoking_status_f20116_0_0','ipaq_activity_group_f22032_0_0','Qualif_Score','frequency_of_friendfamily_visits_f1031_0_0']

	def varmap(self):
		varmap = {}
		with open(self.path+"metadata/varmap.txt") as myfile:
			for line in myfile:
				name, var = line.partition("=")[::2]
				name=name.strip()
				var=var.strip()
				varmap[name] = var

		self.variable_map=varmap
		return varmap

	def map_var(self,df,var_):
		df['var_mapped']=df[var_].map(self.varmap())
		mask=pd.notnull(df['var_mapped'])
		df.loc[mask,var_]=df.loc[mask,'var_mapped']
		df.drop(columns='var_mapped',inplace=True)
		return df



	def get_cols_with_string(self,df,remstrings=None):
		#remove columns with certain strings from dataframe

		if remstrings is None:
			remstrings=self.remwords

		remstrings='|'.join(remstrings)
		remvars=[c for c in df.columns if re.search(remstrings,c)]
		return remvars

	def get_time_periods(self,df):
		# based on string at end of column returns columns which are for a later time period
		later_periods=[c for c in df.columns if c[len(c)-3:len(c)]=='1_0' or\
c[len(c)-3:len(c)]=='2_0' or c[len(c)-3:len(c)]=='3_0' or c[len(c)-3:len(c)]=='0_1' or c[len(c)-3:len(c)]=='0_2'] 
		return later_periods

	def get_obj_cols(self,df):
		#should not model object column so this removed them
		all_dtypes=list(str(c) for c in df.dtypes)
		obj_cols=[c for i,c in enumerate(df.columns) if re.search('obj',all_dtypes[i])]
		
		return obj_cols

	def remove_cols(self,df):
		remvars=self.get_cols_with_string(df)
		later_periods=self.get_time_periods(df)
		obj_cols=self.get_obj_cols(df)
		cols_rem=[c for c in list(set(remvars+later_periods+obj_cols)) if c!='eid']
		df.drop(columns=cols_rem,inplace=True)
		return df,cols_rem

	def rename_cols(self,df):
		#ensures cols can be modelled
		df = df.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
		return df

	def data_process(self,df):

		df=self.rename_cols(df)
		#replace any infinity values with nulls
		df.replace([np.inf, -np.inf], np.nan,inplace=True)
		df,remcols=self.remove_cols(df)
		
		return df,remcols


	def impute_mice(self,df,cols,iters=3):
		kernel = mf.ImputationKernel(
		data=df[cols],
		save_all_iterations=True,
		random_state=1991)
		kernel.mice(iters,verbose=True)
		df_imp = kernel.impute_new_data(df[cols])
		df[cols] = df_imp.complete_data(0)
		return df

	def train_test(self,df,depvar='AD',test_size=0.3,random_state=42):
		mask=(df[depvar]==1)
		cases=df.loc[mask,]
		ctrls=df.loc[~mask,]
		test_case=cases.sample(frac=test_size,random_state=random_state)
		test_ctrl=ctrls.sample(frac=test_size,random_state=random_state)
		df_test=pd.concat([test_case,test_ctrl],axis=0)
		mask=~(df['eid'].isin(df_test['eid']))
		df_train=df.loc[mask,]
		return df_train,df_test

	def sum_feats(self,df_full):
		#summarises full dataframe
		df=pd.DataFrame(df_full.groupby(['Attribute']).\
agg({'mean_shap':'mean','model_feature_importance':'mean','shap_model_fi':'mean'}).reset_index()).reset_index()
		df.sort_values(by='mean_shap',ascending=False,inplace=True)

		return df

	def Boruta_feats2(self,df):

		model = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42)

		# let's initialize Boruta
		feat_selector = BorutaPy(
			verbose=2,
			estimator=model,
			n_estimators='auto',
			max_iter=10  # number of iterations to perform
		)

		predvars=[c for c in df.columns if c!='AD']
		X=df[predvars]
		y=df['AD']
		X.fillna(X.mean(),inplace=True)

		feat_selector.fit(np.array(X), np.array(y))
		df_bor=pd.DataFrame(zip(X.columns,feat_selector.support_,feat_selector.ranking_),columns=['Attribute','Pass','Rank'])

		mask=(df_bor['Pass']==True)
		df_bor.loc[mask,'recs']=df_bor.loc[mask,].groupby('Attribute')['Rank'].transform('count')

		mask=(df_bor['Pass']==True)&(df_bor['recs']>=1)
		df_bor=pd.DataFrame(df_bor.loc[mask,].groupby('Attribute').agg({'Rank':['count']})).reset_index()
		df_bor.columns=['Attribute','Counts']

		return df_bor
		


	def Boruta_feats(self,dict_train,df_train):
		# function to run Boruta on dictionary of dataframes and return feature dataframe

		model = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42)

		# let's initialize Boruta
		feat_selector = BorutaPy(
			verbose=2,
			estimator=model,
			n_estimators='auto',
			max_iter=10  # number of iterations to perform
		)
		
		df_bor_full=pd.DataFrame([])
		for i in range(len(dict_train)):
			mask=(df_train['eid'].isin(dict_train[i]))
			df1=df_train.loc[mask,]
			predvars=[c for c in df1.columns if c!='AD']
			X=df1[predvars]
			y=df1['AD']
			X.fillna(X.mean(),inplace=True)

			feat_selector.fit(np.array(X), np.array(y))
			df_bor=pd.DataFrame(zip(X.columns,feat_selector.support_,feat_selector.ranking_),columns=['Attribute','Pass','Rank'])
			df_bor['run']=i
			df_bor_full=pd.concat([df_bor_full,df_bor],axis=0)

		mask=(df_bor_full['Pass']==True)
		df_bor_full.loc[mask,'recs']=df_bor_full.loc[mask,].groupby('Attribute')['Rank'].transform('count')

		mask=(df_bor_full['Pass']==True)&(df_bor_full['recs']>=1)
		df_bor_fin=pd.DataFrame(df_bor_full.loc[mask,].groupby('Attribute').agg({'Rank':['count']})).reset_index()
		df_bor_fin.columns=['Attribute','Counts']
		
		return df_bor_fin

	def model_fit(self,mod,train_x, train_y):
		model=mod.fit(train_x, train_y)
		return model

	def auc_score(self,valid_x,valid_y,model,mod_name='XGB'):
		pred=model.predict_proba(valid_x)[:, 1]
		score = roc_auc_score(valid_y,pred)
		print('AUC '+mod_name+': ',str(score))
		return score

	def prec_recall_score(self,valid_x,valid_y,model,mod_name='XGB'):
		pred=model.predict(valid_x)
		pred_prob=model.predict_proba(valid_x)[:, 1]
		prec_score = precision_score(valid_y,pred)
		rec_score=recall_score(valid_y,pred)
		auc_score = roc_auc_score(valid_y,pred_prob)
		#print('AUC '+mod_name+': ',str(score))
		return prec_score,rec_score,auc_score

	


	def model_aucs(self,dict_dfs,df_train,depvar='AD',models=None,mod_names=None):

		errors=[]
		aucs=[]
		model_names=[]

		if models is None:
			models=[self.mod_xgb_base,self.mod_xgb,self.mod_rf,self.mod_svm,self.lgbm_mod,self.log_reg]
			mod_names=['XGBoost','XGBoost Hyp','Random Forest','Support Vector Machine','Light GBM','Logistic Regression']

		for i in dict_dfs:
			
			#try:

			
			mask=(df_train['eid'].isin(dict_dfs[i]))
			df1=df_train.loc[mask,].drop(columns='eid')
			predvars=[c for c in df1.columns if c!=depvar]
	
			X=df1[predvars]#feats_new_stu
			y=df1[depvar]
			X.fillna(X.mean(),inplace=True)
			train_x, valid_x, train_y, valid_y = train_test_split(X, y, \
test_size=0.3, shuffle=True, stratify=y, random_state=1301)

			for j,mod in enumerate(models):
				model=mod.fit(train_x, train_y)
				score=self.auc_score(valid_x,valid_y,model,mod_names[j])
				aucs.append(score)
				model_names.append(mod_names[j])
		
			#except:
			#	print("error")
			#	errors.append(i)

		df=pd.DataFrame({'Model':model_names,'AUC':aucs})

		return df

	def shap_sign(self,df,X):
		#df is the first SHAP dataset, X is the values dataset, find correlations between SHAP values and variables
		#values.

		corr = []
		for j in df.columns:
			#change to ensure nulls are filled in if they are there
			b = np.corrcoef(df[j].fillna(0),X[j].fillna(X[j].mean()))[1][0]
			corr.append(b)
			
		return corr


	def shap_out(self,sv,X,model):
		#returns SHAP abs df based on shap_values sv and includes correlations
		sv=sv.values
		df=pd.DataFrame(sv,columns=X.columns)
		corr=self.shap_sign(df,X)
		df_abs=abs(df)
		df_abs=pd.DataFrame(df_abs.mean(axis=0)).reset_index()
		df_abs.columns=['Attribute','mean_shap']
		df_abs['corr']=corr

		return df_abs

	def feat_imp(self,model,X):
		#returns built in feature importance for a given model with columns based on the trained dataset
		fi = pd.DataFrame(sorted(zip(model.feature_importances_,X.columns)),\
columns=['model_feature_importance','Attribute'])
		
		return fi

	def feats_out(self,sv,X,model):
		#combines feat imp and SHAP to return weighted breakdown
		
		df_shap=self.shap_out(sv,X,model)
		df=self.feat_imp(model,X)
		df=pd.merge(df_shap,df,on='Attribute',how='outer')

		df['weighted_shap']=df['mean_shap']/df['mean_shap'].sum()
		df['weighted_model_fi']=df['model_feature_importance']/df['model_feature_importance'].sum()
		df['shap_model_fi']=df['weighted_shap']+df['weighted_model_fi']
		mask=(df['mean_shap']>0)|(df['model_feature_importance']>0)
		df=df.loc[mask,]
		df.sort_values(by='shap_model_fi',ascending=False,inplace=True)
		
		return df

	def make_xy(self,df,depvar='AD'):
		predvars=[c for c in df.columns if c!=depvar and c!='eid']
		
		X=df[predvars]#feats_new_stu
		y=df[depvar]

		return X,y

	def fit_model(self,X,y):
		model=self.mod_xgb.fit(X, y)
		return model
	

	def get_shap_feats(self,df,depvar='AD'):
		
		X,y=self.make_xy(df,depvar=depvar)
		model=self.fit_model(X,y)

		explainer = shap.Explainer(model, X)
		shap_values = explainer(X,check_additivity=False)
		df_s=self.feats_out(shap_values,X,model)

		return df_s


	def boxplot_draw(self,df,thresh=0.025,limit=25,var='mean_shap'):

	    df['overall_score']=df.groupby('Attribute')[var].transform('sum')/\
df.groupby('Attribute')[var].transform('count')
	    mask=(df['overall_score']>thresh)
	    df=df.loc[mask,]

	    from matplotlib.pyplot import figure
	    figure(figsize=(10, 20), dpi=300)

	    df['Attr2']=df['Attribute'].map(self.varmap())

	    mask=pd.isnull(df['Attr2'])
	    df.loc[mask,'Attr2']=df.loc[mask,'Attribute']


	    df_s=pd.DataFrame(df.groupby('Attr2')['overall_score'].sum()).reset_index()
	    df_s.sort_values(by='overall_score',inplace=True,ascending=False)
	    df_s['rank']=np.arange(len(df_s))+1
	    mask=(df_s['rank']<=limit)
	    df_s=df_s.loc[mask,]
	    df_s['Attribute_ranked']=df_s['rank'].astype(str)+': '+df_s['Attr2']

	    df_s=df_s[['Attr2','Attribute_ranked']]
	    df=pd.merge(df,df_s,on='Attr2',how='left')
	    df.sort_values(by='overall_score',inplace=True,ascending=False)
	    
	    
	    custom_palette=dict(zip(list(df['Attribute_ranked']),\
	['b' if c<0 else 'r' if c>0  else 'orange' for c in list(df['corr'])]))

	    ax=sns.boxplot(data=df,y='Attribute_ranked',x='mean_shap',palette=custom_palette,showmeans=True)
	    ax.axes.get_yaxis().set_label([])
	    ax.tick_params(axis='both', which='major', labelsize=30)
	    ax.set_xlabel('Mean FI Score', fontsize=30)
	    ax.set_ylabel('', fontsize=30)
	    
	    plt.show()
	    
	    return df





	def iterate_models(self,dict_dfs,df_train,depvar='AD',plot=False,show_shap=True,rand=False,verbose=False):

		df_full=pd.DataFrame([])
		errors=[]
		aucs=[]

		for i in dict_dfs:
			
			try:

					#filter
				mask=(df_train['eid'].isin(dict_dfs[i]))
				df1=df_train.loc[mask,].drop(columns='eid')

				predvars=[c for c in df1.columns if c!=depvar]
		
				X=df1[predvars]#feats_new_stu
				y=df1[depvar]

				X.fillna(X.mean(),inplace=True)



				#df_ad_agenorm3=df_ad_agenorm2[feats_new_stud]
				if rand:
					train_x, valid_x, train_y, valid_y = train_test_split(X, y, \
	test_size=0.4, shuffle=True, stratify=y,random_state=i)
				else:
					train_x, valid_x, train_y, valid_y = train_test_split(X, y, \
	test_size=0.4, shuffle=True, stratify=y)


				#model = xgboost.XGBClassifier().fit(train_x, train_y)
					#
				model=self.mod_xgb.fit(train_x, train_y)
				score=self.auc_score(valid_x,valid_y,model,mod_name='XGB')

				if verbose:
					print(score)
				aucs.append(score)

				#print(aucs)

				# compute SHAP values

				if show_shap is True:
					explainer = shap.Explainer(model, X)
					shap_values = explainer(X)
					df_s=self.feats_out(shap_values,X,model)
					df_full=pd.concat([df_full,df_s],axis=0)
					#print(df_full).head()

					if plot is True:
						shap.summary_plot(shap_values, X)
						#shap.plots.bar(shap_values[0], max_display=15)
						#plt.show()

					

			
			except:
				print("error")
				errors.append(i)

		if show_shap is True:

			df_sum=self.sum_feats(df_full)
			outs=[df_full,df_sum,aucs,errors]
		else:
			outs=[aucs,errors]

		return outs


	def define_cols(self,df,df_feats=None,featsfile='df_lgb_feats.parquet'):

		#extract new ML columns
		if df_feats is None:
			df_feats=pd.read_parquet(self.path+featsfile)

		cols_ml=list(df_feats['Attribute'])
		#livingstone modelled columns
		cols_liv=[c for c in df.columns if c in self.livingstone_cols]
		#interesection of both
		ml_liv=list(set(cols_liv+cols_ml))

		return cols_ml, cols_liv, ml_liv

	def get_vif(self,X,thresh=5):
		vi = pd.DataFrame()
		vi['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
		vi['Attribute'] = X.columns
		vi.sort_values('VIF', ascending=False,inplace=True)
		mask=(vi['VIF']>thresh)
		susp_feats=vi.loc[mask,]
		return susp_feats

	def logit_pvalue(self,model, x):
		""" Calculate z-scores for scikit-learn LogisticRegression.
		parameters:
			model: fitted sklearn.linear_model.LogisticRegression with intercept and large C
			x:     matrix on which the model was fit
		This function uses asymtptics for maximum likelihood estimates.
		"""
		p = model.predict_proba(x)
		n = len(p)
		m = len(model.coef_[0]) + 1
		coefs = np.concatenate([model.intercept_, model.coef_[0]])
		x_full = np.matrix(np.insert(np.array(x), 0, 1, axis = 1))
		ans = np.zeros((m, m))
		for i in range(n):
			ans = ans + np.dot(np.transpose(x_full[i, :]), x_full[i, :]) * p[i,1] * p[i, 0]
		vcov = np.linalg.inv(np.matrix(ans))
		se = np.sqrt(np.diag(vcov))
		t =  coefs/se  
		p = (1 - norm.cdf(abs(t))) * 2
		return p

	def get_odds_ratio_df(self,model,X):
		#returns the odds ratio from the above and associated p values
		df=pd.DataFrame(zip(X.columns,list(np.exp((model.coef_))[0]),list(self.logit_pvalue(model,X)))\
	,columns=['Attribute','odds_ratio','p values'])
		return df

	def run_log_reg(self,dict_dfs,df_train,df_test,df_sum,feats=50,depvar='AD'):
		
		scaler = StandardScaler()

		auc_lgb=[]
		auc_xgb=[]
		auc_log=[]

		df_full=pd.DataFrame([])

		for i in dict_dfs:
			mask=(df_train['eid'].isin(dict_dfs[i]))

			df1=df_train.loc[mask,].drop(columns='eid')
			predvars=[c for c in df1.columns if c!=depvar]
			predvars=list(df_sum['Attribute'].head(feats))
			X=df1[predvars]
			y=df1[depvar]
			X.fillna(X.mean(),inplace=True)

			val_x=df_test[predvars]
			val_x.fillna(val_x.mean(),inplace=True)
			val_y=df_test[depvar]

			scale_cols=[col for col in predvars if X[col].nunique()>10]
			X[scale_cols] = scaler.fit_transform(X[scale_cols])
			val_x[scale_cols] = scaler.fit_transform(val_x[scale_cols])

			mod=self.log_reg.fit(X, y)
			mod2=self.mod_xgb.fit(X, y) 
			mod3=self.mod_rf.fit(X, y)

			
			df_odds=self.get_odds_ratio_df(mod,X)
			df_odds['run']=i
			df_full=pd.concat([df_full,df_odds],axis=0)

		   
			score=self.auc_score(val_x,val_y,mod,mod_name='log reg')
			score2=self.auc_score(val_x,val_y,mod2,mod_name='XGB')
			score3=self.auc_score(val_x,val_y,mod3,mod_name='R forest')

			mod4=self.mod_svm.fit(X, y)
			score4=self.auc_score(val_x,val_y,mod4,mod_name='SVM')

			auc_log.append(score)
			auc_xgb.append(score2)
			auc_lgb.append(score3)


		return df_full


	def comp_log_reg(self,dict_dfs,dict_dfs_test,df_train,df_test,df_sum,feats,depvar='AD',mod_use=None,pvals_rep=False):
		
		scaler = StandardScaler()

		if mod_use is None:
			mod_use=self.log_reg

		auc_log=[]
		prec_log=[]
		rec_log=[]
		var_sets=[]
		iters=[]

		df_full=pd.DataFrame([])

		for i in dict_dfs:

			if dict_dfs is None:
				df1=df_train.copy()
			else:
				mask=(df_train['eid'].isin(dict_dfs[i]))
				df1=df_train.loc[mask,]

			if dict_dfs_test is None:
				df_t1=df_test.copy()
			else:
				mask=(df_test['eid'].isin(dict_dfs_test[i]))
				df_t1=df_test.loc[mask,]

			df1.drop(columns='eid',inplace=True)
			df_t1.drop(columns='eid',inplace=True)
				
			

			predvars=feats#list(df_sum['Attribute'].head(feats))
			liv_vars=self.livingstone_cols
			allvars=list(set(predvars+liv_vars))

			

			var_names=['New vars','Known vars','Known + new vars']
			vars_list=[predvars,liv_vars,allvars]


			for j,vars_ in enumerate(vars_list):

				X=df1[vars_]
				print(df1.shape)
				y=df1[depvar]
				X.fillna(X.mean(),inplace=True)

				vars_=[c for c in vars_ if X[c].nunique()>=2]
				X=X[vars_]

				val_x=df_t1[vars_]
				val_x.fillna(val_x.mean(),inplace=True)
				val_y=df_t1[depvar]

				scale_cols=[col for col in vars_ if X[col].nunique()>10]

				X[scale_cols] = scaler.fit_transform(X[scale_cols])
				val_x[scale_cols] = scaler.fit_transform(val_x[scale_cols])

				print("Fit 1 -"+str(X.shape[1])+'feats')
				mod=mod_use.fit(X, y)
				score=self.auc_score(val_x,val_y,mod,mod_name='log reg')
				prec_score,rec_score,auc_score=self.prec_recall_score(val_x,val_y,mod,mod_name='log reg')
				df_odds=self.get_odds_ratio_df(mod,X)

				#select feats with p value below 0.05 then run again
				if pvals_rep:
					mask=(df_odds['p values']<0.1)
					new_feats=list(df_odds.loc[mask,'Attribute'])
					X=X[new_feats]
					val_x=val_x[new_feats]
					print("Fit 2 -"+str(X.shape[1])+'feats')
					mod=mod_use.fit(X, y)
					score=self.auc_score(val_x,val_y,mod,mod_name='log reg')
					df_odds=self.get_odds_ratio_df(mod,X)

				iters.append(i)
				var_sets.append(var_names[j])
				auc_log.append(auc_score)
				prec_log.append(prec_score)
				rec_log.append(rec_score)
				#mod2=self.mod_xgb.fit(X, y) 
				#mod3=self.mod_rf.fit(X, y)

			
				
				df_odds['run']=i
				df_odds['Variables']=var_names[j]

				


				df_full=pd.concat([df_full,df_odds],axis=0)
		   
				

		df_perf=pd.DataFrame({'Iteration':iters,'Variables':var_sets,'AUC':auc_log,'Precision':prec_log,"Recall":rec_log})


		return df_full,df_perf


	def comp_log_reg_single_varset(self,dict_dfs,df,df_test,df_sum,feats=50,depvar='AD'):
		
		scaler = StandardScaler()



		auc_log=[]
		iters=[]

		df_full=pd.DataFrame([])

		for i in dict_dfs:

			try:
				df_train,df_test=self.train_test(df,depvar='polyneuropathy')
				mask=(df['eid'].isin(dict_dfs[i]))
				df1=df.loc[mask,].drop(columns='eid')
				print(df1.shape)
				
				predvars=list(df_sum['Attribute'].head(feats))

				X=df_train[predvars]
				y=df_train[depvar]
				X.fillna(X.mean(),inplace=True)

				predvars=[c for c in predvars if X[c].nunique()>=2]
				X=X[predvars]

				val_x=df_test[predvars]
				val_x.fillna(val_x.mean(),inplace=True)
				val_y=df_test[depvar]

				scale_cols=[c for c in predvars if X[c].nunique()>10]

				X[scale_cols] = scaler.fit_transform(X[scale_cols])
				val_x[scale_cols] = scaler.fit_transform(val_x[scale_cols])

				mod=self.log_reg.fit(X, y)
				score=self.auc_score(val_x,val_y,mod,mod_name='log reg')

				iters.append(i)
				auc_log.append(score)
				#mod2=self.mod_xgb.fit(X, y) 
				#mod3=self.mod_rf.fit(X, y)

			
				df_odds=self.get_odds_ratio_df(mod,X)
				df_odds['run']=i
				df_full=pd.concat([df_full,df_odds],axis=0)
			except:
				print("error")
				

		df_perf=pd.DataFrame({'Iteration':iters,'AUC':auc_log})


		return df_full,df_perf

	def make_econ_bar(self,df,sort_var='mean_shap',recs=20,title='APOE4 Carriers Feature Importance',
		sub_title="Mean SHAP Score",footer="""Source: UK Biobank""",outfile='chart.png',labels_show=False,
		tick_vals_use=[0, 0.05, 0.1, 0.15, 0.2],shrink=True,y_max=20.5,y_err=None,figsize=(3,6),out=True,label=None):
    
	     # Setup plot size.
	    fig, ax = plt.subplots(figsize=figsize)

	    y_max=recs

	    # Create grid 
	    # Zorder tells it which layer to put it on. We are setting this to 1 and our data to 2 so the grid is behind the data.
	    ax.grid(which="major", axis='x', color='#758D99', alpha=0.6, zorder=1)

	    # Remove splines. Can be done one at a time or can slice with a list.
	    ax.spines[['top','right','bottom']].set_visible(False)

	    # Make left spine slightly thicker
	    ax.spines['left'].set_linewidth(1.1)
	    ax.spines['left'].set_linewidth(1.1)

	    # Setup data
	    df_bar = df.sort_values(by=sort_var,ascending=False).head(recs)
	    df_bar=df_bar.sort_values(by=sort_var,ascending=True)

	    if label is not None:
	    	labelx=list(df_bar['p value diff'])
	    	print(labelx)
	    else:
	    	labelx=None
	    	
	    
	    custom_palette=['#006BA2' if c<0 else '#E3120B' for c in list(df_bar['corr'])]


	    # Plot data
	    ax.barh(df_bar['Attribute'], df_bar[sort_var].round(3), color=custom_palette, zorder=2,label=labelx)#
	    #ax.bar_label(labelx, label_type='center')


	    # Set custom labels for x-axis
	    ax.set_xticks(tick_vals_use)
	    ax.set_xticklabels(tick_vals_use)

	    # Reformat x-axis tick labels
	    ax.xaxis.set_tick_params(labeltop=True,      # Put x-axis labels on top
	                             labelbottom=False,  # Set no x-axis labels on bottom
	                             bottom=False,       # Set no ticks on bottom
	                             labelsize=11,       # Set tick label size
	                             pad=-1)             # Lower tick labels a bit

	    # Reformat y-axis tick labels
	    ax.set_yticklabels(df_bar['Attribute'],      # Set labels again
	                       ha = 'left')              # Set horizontal alignment to left
	    ax.yaxis.set_tick_params(pad=300,            # Pad tick labels so they don't go over y-axis
	                             labelsize=11,       # Set label size
	                             bottom=False)       # Set no ticks on bottom/left

	    # Shrink y-lim to make plot a bit tighter
	    if shrink:
	    	ax.set_ylim(-0.5,y_max )

	    # Add in line and tag
	    ax.plot([-1.30, .87],                 # Set width of line
	            [1.02, 1.02],                # Set height of line
	            transform=fig.transFigure,   # Set location relative to plot
	            clip_on=False, 
	            color='#E3120B', 
	            linewidth=.6)
	    ax.add_patch(plt.Rectangle((-1.30,1.02),                # Set location of rectangle by lower left corder
	                               0.22,                       # Width of rectangle
	                               -0.02,                      # Height of rectangle. Negative so it goes down.
	                               facecolor='#E3120B', 
	                               transform=fig.transFigure, 
	                               clip_on=False, 
	                               linewidth = 0))

	    # Add in title and subtitle
	    ax.text(x=-1.30, y=.96, s=title, transform=fig.transFigure, ha='left', fontsize=13, weight='bold', alpha=.8)
	    ax.text(x=-1.30, y=.925, s=sub_title, transform=fig.transFigure, ha='left', fontsize=11, alpha=.8)

	    # Set source text
	    ax.text(x=-1.30, y=.08, s=footer, transform=fig.transFigure, ha='left', fontsize=9, alpha=.7)
	    
	    if labels_show:
	        for i,bars in enumerate(ax.containers):
	            ax.bar_label(bars)
	            #ax.bar_label(labelx[i])

	    if out:
		    # Export plot as high resolution PNG
		    plt.savefig(outfile,    # Set path and filename
		                dpi = 300,                     # Set dots per inch
		                bbox_inches="tight",           # Remove extra whitespace around plot
		                facecolor='white')  

	    return plt           # Set background color to white








