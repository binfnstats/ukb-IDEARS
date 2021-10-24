"""
Created on Mon Sep 27 16:39:52 2021

@author: michaelallwright

This file contains a set of functions that are to be used in the new version of UKB
to do the key analyses

"""

#key information to run financial passport for - user and start/end date. Move to config file?

from typing import Any, List
import pandas as pd
import numpy as np
import re
import icd10
import datetime as dt
import ast

class data_proc_main(object):
	"""
	Module for feature engineering
	"""

	def __init__(self):
		"""
		Initilising models.
		"""
		self.path="/Users/michaelallwright/Dropbox (Sydney Uni)/michael_PhD/Projects/UKB/Data/"
		self.fullfile='ukb_tp0_new.parquet'
		self.cols_nonmiss_file='cols80.csv'


		self.dis_map={'cerebrovasc':['I60','I62','I63','I65','I66','I67','I68','I69'],
		 'stroke':['I64'],'TBI':['S0'],'Hear_loss':['H919','H901','H902','H903','H904',\
	'H905','H906','H907','H908','H909','H910','H911','H912','Z974']}

		self.keycols=['eid','date_of_attending_assessment_centre_f53_0_0','age_when_attended_assessment_centre_f21003_0_0']

		self.excwords=['Prefer not to answer','nan','None of the above']
		self.excs='source_of_report|first_reported|icd10|icd9|operative_procedures|treatment_speciality|\
	external_ca|patient_recoded|hospital_polymorphic|_report|assay_date|device_id'

		self.excs_cts='aliquot|assessment_centre|acquisition_time|main_speciality|date_of_|patient_classi|\
	methods_of_discharge|inpatient_record_format|weight_method|_missing_reason|eid'


		self.speccols=['sex_f31','average_total_household_income_before_tax_f738','usual_walking_pace_f924',
 		'frequency_of_friendfamily_visits_f1031','drive_faster_than_motorway_speed_limit_f1100',
		'weekly_usage_of_mobile_phone_in_last_3_months_f1120','qualifications_f6138',
		'gender','avgincome','walkspeed','freqfriendfamily','faster_mot_speed','weekly_mobphone_mins',
		'qualif_score','APOE4_Carriers']
		self.icd10_file='ukb_ICD10.parquet'
		self.year=2021
		self.month=1
		self.day=31

		self.years_wait=2
		self.years_max=10
		self.min_part_dis=200



	
	def convert_null(self,df):
		for col in df.columns:
			mask=(df[col]=='Prefer not to answer')|(df[col]=='nan')
			df[col][mask]=np.NaN
		return df

	def findcols(self,df,string):
		return [col for col in df if string in col]

	def returndesc(self,string):
		'''
		functions to apply the icd10 mapping and return disease and disease block
		'''
		code=icd10.find(str(string))
		if code:
			desc=code.description
		else:
			desc=string
		return desc

	def returndescblock(self,string):
		
		try:
			code=icd10.find(str(string))
			desc_block=str(code.block_description)
			
			return desc_block
		except:
			pass



	def dis_date_file(self):

		df=pd.read_parquet('%s%s' % (self.path,self.icd10_file))

		"""
		format dates and work out age today
		"""
		df['date_of_attending_assessment_centre_f53_0_0']=pd.to_datetime(df['date_of_attending_assessment_centre_f53_0_0'])
		df['Age_Today']=df['age_when_attended_assessment_centre_f21003_0_0']+(dt.datetime(self.year, self.month, self.day)-\
		df['date_of_attending_assessment_centre_f53_0_0']).dt.days/365.25

		"""
		ICD10 columns for extraction and split of dates and diseases data
		"""

		cols1=[col for col in df.columns if '41270' in col or 'eid' in col]
		cols2=[col for col in df.columns if '41280' in col or 'eid' in col]

		df_dis=df[cols1]
		df_date=df[cols2]

		"""
		make so 1 record per individual per ICD10
		"""
		df_dis = pd.melt(df_dis, id_vars='eid', value_name='VALUE')
		df_dis=df_dis[pd.notnull(df_dis['VALUE'])]

		df_dis.columns=['eid','variable','disease']
		#df_dis['disease']=df_dis['disease'].str.replace('b','')

		
		df_dis['disease']=df_dis['disease'].str.replace("'","")
		df_dis['variable']=df_dis['variable'].str.replace('diagnoses_icd10_','')

		df_date = pd.melt(df_date, id_vars='eid', value_name='VALUE')
		df_date=df_date[pd.notnull(df_date['VALUE'])]
		df_date['variable']=df_date['variable'].str.replace('41280','41270')
		df_date['variable']=df_date['variable'].str.replace('date_of_first_inpatient_diagnosis_icd10_','')

		df_date.columns=['eid','variable','dis_date']

		
		df_date['dis_date']=df_date['dis_date'].str.replace('b','')
		df_date['dis_date']=df_date['dis_date'].str.replace("'","")
		df_date['dis_date']=pd.to_datetime(df_date['dis_date'])

		df_dis_date=pd.merge(df_dis,df_date,on=['eid','variable'],how='left')

		df_dis_date=pd.merge(df_dis_date,df[['eid','Age_Today','date_of_attending_assessment_centre_f53_0_0']])
		df_dis_date['Age_disease']=df_dis_date['Age_Today']-\
		(dt.datetime(self.year, self.month, self.day)-pd.to_datetime(df_dis_date['dis_date']))\
		.dt.days/365.25

		df_dis_date['disease_name']=df_dis_date['disease'].apply(self.returndesc)
		df_dis_date['disease_block']=df_dis_date['disease'].apply(self.returndescblock)
		df_dis_date.rename(columns={'date_of_attending_assessment_centre_f53_0_0':'date_assess','dis_date':'disease_date'}\
	,inplace=True)

		mask_bef=(df_dis_date['disease_date']<df_dis_date['date_assess'])

		mask_aft=(df_dis_date['disease_date']>=df_dis_date['date_assess']+pd.offsets.DateOffset(years=self.years_wait))&\
		(df_dis_date['disease_date']<=df_dis_date['date_assess']+pd.offsets.DateOffset(years=self.years_max))

		mask_10y=(df_dis_date['disease_date']>df_dis_date['date_assess']+pd.offsets.DateOffset(years=self.years_max))

		df_dis_date['dis_bef']=0
		df_dis_date['dis_bef'][mask_bef]=1

		df_dis_date['dis_aft']=0
		df_dis_date['dis_aft'][mask_aft]=1

		df_dis_date['dis_exc']=0
		df_dis_date['dis_exc'][mask_10y|mask_bef]=1

		df_dis_date['total_bef']=df_dis_date.groupby('disease')['dis_bef'].transform('sum')
		df_dis_date['total_aft']=df_dis_date.groupby('disease')['dis_aft'].transform('sum')

		"""
		mapping of certain diseases by their codes in dis_map above
		"""

		for var in self.dis_map:
			df_dis_date[var]=0
			mask=(df_dis_date['disease'].str.contains('|'.join(self.dis_map[var])))
			df_dis_date[var][mask]=df_dis_date['dis_bef']

		df_dis_date=df_dis_date[(df_dis_date['total_bef']>self.min_part_dis)]

		cols=[k for k in self.dis_map]
		spec_conds=pd.DataFrame(df_dis_date.groupby(['eid'])[cols].sum()).reset_index()
		dis_ohe_icd10=pd.DataFrame(df_dis_date.groupby(['eid','disease_name'])['dis_bef'].sum()\
	.unstack('disease_name')).reset_index()

		totaldis=pd.DataFrame(df_dis_date.groupby('eid')['dis_bef'].sum()).reset_index()
		totaldis.columns=['eid','total_dis']

		disblock=pd.DataFrame(df_dis_date.groupby(['eid','disease_block'])['dis_bef'].max().\
	unstack('disease_block')).reset_index()
		disblock.fillna(0,inplace=True)

		dis_ohe_icd10=pd.merge(dis_ohe_icd10,spec_conds,on='eid',how='outer')
		dis_ohe_icd10=pd.merge(df['eid'],dis_ohe_icd10,how='left',on='eid')
		dis_ohe_icd10=pd.merge(dis_ohe_icd10,disblock,how='left',on='eid')
		dis_ohe_icd10=pd.merge(dis_ohe_icd10,totaldis,how='left',on='eid')
		dis_ohe_icd10.fillna(0,inplace=True)

		df_dis_date.to_parquet(self.path+'df_dis_date_test.parquet')
		dis_ohe_icd10.to_parquet(self.path+'dis_ohe_icd10_test.parquet')


		return df_dis_date

	def famhistory(self):
		df=pd.read_parquet(self.path+'df_fam_hist.parquet')
		df = pd.melt(df, id_vars='eid', value_name='VALUE')
		df['father_parkinson']=0
		df['father_parkinson'][(df['VALUE'].str.contains('Parkins|parkins'))&\
		(df['variable'].str.contains('father'))]=1

		df['mother_parkinson']=0
		df['mother_parkinson'][(df['VALUE'].str.contains('Parkins|parkins'))&\
		(df['variable'].str.contains('mother'))]=1

		df['father_dementia']=0
		df['father_dementia'][(df['VALUE'].str.contains('Dementia|dementia'))&\
		(df['variable'].str.contains('father'))]=1

		df['mother_dementia']=0
		df['mother_dementia'][(df['VALUE'].str.contains('Dementia|dementia'))&\
		(df['variable'].str.contains('mother'))]=1

		df_fam_pddem=pd.DataFrame(df.groupby(['eid'])\
		['father_parkinson','mother_parkinson','father_dementia','mother_dementia'].sum()).reset_index()
		df_fam_pddem['eid']=df_fam_pddem['eid'].astype(str)
		df_fam_pddem['parental_pd']=df_fam_pddem[['father_parkinson','mother_parkinson']].sum(axis=1)
		df_fam_pddem['parental_dem']=df_fam_pddem[['father_dementia','mother_dementia']].sum(axis=1)
		df_fam_pddem.drop(columns=['father_parkinson','mother_parkinson','father_dementia','mother_dementia'],inplace=True)

		df_fam_pddem.to_parquet(self.path+'df_fam_pddem.parquet')

		return df_fam_pddem


	def ukb_diseases(self):

		df=pd.read_parquet('%s%s' % (self.path,'ukb_diseases_test.parquet'))
		dis_full=pd.DataFrame([])

		for i,col in enumerate(df.columns):
			if 'eid' not in col and 'assessment_centre' not in col:
				print(i)
				df1=df[[col,'eid','date_of_attending_assessment_centre_f53_0_0']][pd.notnull(df[col])]
				df1.columns=['disease_date','eid','date_assess']
				df1['disease_date']=pd.to_datetime(df1['disease_date']).dt.date
				df1['date_assess']=pd.to_datetime(df1['date_assess']).dt.date
				df1['disease']=str(col)
				dis_full=pd.concat([dis_full,df1],axis=0)

		dis_full['disease']=dis_full['disease'].str.replace('date_|_first_|reported_','',regex=True)

		mask_bef=(dis_full['disease_date']<dis_full['date_assess'])

		mask_aft=(dis_full['disease_date']>=dis_full['date_assess']+pd.offsets.DateOffset(years=2))&\
		(dis_full['disease_date']<=dis_full['date_assess']+pd.offsets.DateOffset(years=10))

		mask_10y=(dis_full['disease_date']>dis_full['date_assess']+pd.offsets.DateOffset(years=10))

		dis_full['dis_bef']=0
		dis_full['dis_bef'][mask_bef]=1

		dis_full['dis_aft']=0
		dis_full['dis_aft'][mask_aft]=1

		dis_full['dis_exc']=0
		dis_full['dis_exc'][mask_10y|mask_bef]=1

		dis_full['total_bef']=dis_full.groupby('disease')['dis_bef'].transform('sum')
		dis_full['total_aft']=dis_full.groupby('disease')['dis_aft'].transform('sum')

		dis_ohe=pd.DataFrame(dis_full.groupby(['eid','disease'])['dis_bef'].sum().unstack('disease')).reset_index()

		"""
		bring null records back
		"""

		dis_ohe=pd.merge(df['eid'],dis_ohe,how='left',on='eid')
		dis_ohe.fillna(0,inplace=True)

		dis_ohe['total_dis']=dis_ohe[[col for col in dis_ohe.columns if 'eid' not in col]].sum(axis=1)

		dis_ohe=pd.read_parquet(path+'dis_ohe.parquet')

		return dis_ohe

	def onehotencoder(self,df,cols,excwords,maxrecs=10,mincount=0.8):
	
		#create nulls where unknown for future imputation
		for col in cols:
			mask_exc=(df[col].isin(excwords))
			df[col][mask_exc]=np.nan  
		ohe_cols=\
		[col for col in cols if len(df[col].value_counts())<maxrecs
		and df[col].count()/df[col].shape[0]>mincount]

		rejcols=[c for c in cols if c not in ohe_cols and len(df[c].value_counts())<maxrecs]

		print(len(rejcols)+len(ohe_cols)-len(cols))
		
		print(r'Total ohe variables = %0.0f ' % (len(ohe_cols)))
			
		df_ohe_cols=df[ohe_cols]
			
		df_out=pd.get_dummies(df_ohe_cols, prefix=df_ohe_cols.columns,
				columns=df_ohe_cols.columns)
		
		df_out['eid']=df['eid'].tolist()
		
		return df_out,rejcols

	

	def create_dic(self,df,maxnum=15):
		return [(col,set(df[col][pd.notnull(df[col])].unique())) for col in df.columns if len(df[col].unique())<=maxnum]

	def ordinal_lookup(self,df):
		list1=self.create_dic(df,maxnum=15)
		valslist=pd.DataFrame(list1)
		valslist.columns=['column','values']
		valslist['merge']=valslist['values'].astype(str)
		vallist2=pd.DataFrame(valslist['values'].value_counts()).reset_index()
		vallist2.columns=['values','recs']
		vallist2['merge']=vallist2['values'].astype(str)
		vallistcomp=pd.merge(valslist,vallist2,on='merge',how='left')
		vallistcomp.sort_values(by='recs',ascending=False,inplace=True)
		vallistcomp.to_csv(self.path+'vallistcomp_test.csv')

		return vallistcomp


	def ordcols_import(self):
		ordcols_full=pd.read_csv(self.path+'vallistcomp_edited.csv')
		ordcols=ordcols_full[(ordcols_full['Do']!='Exclude')&(ordcols_full['Do']!='ohe')&\
			pd.notnull(ordcols_full['Do'])][['column','Do']]
		ordcols["inv_map"]=""
		for i,col in enumerate(ordcols['column']):
			try:
				mask=(ordcols['column']==col)
				val=np.asarray(ordcols['Do'][mask])[0]
				
				orig_map=ast.literal_eval(str(val))
			
				inv_map = {v: k for k, v in orig_map.items()}
				ordcols["inv_map"][mask]=str(inv_map)
			except:
				pass
		ordcols['Do']=ordcols['Do'].str.replace("'Do not know':1,","")

		return ordcols,ordcols_full


	def cols_nonmiss(self):
		cols_nomiss=pd.read_csv(self.path+self.cols_nonmiss_file)
		cols_nomiss=list(cols_nomiss['colname'])
		cols_nomiss=[col for col in cols_nomiss if not re.search(self.excs,col) ]
		return cols_nomiss

	def splitvars(self):
		cols_nomiss=self.cols_nonmiss()
		df=pd.read_parquet(self.path+self.fullfile,columns=cols_nomiss)
		df=self.convert_null(df)
		return df


	def vars_det(self,df):
		ordcols,ordcols_full=self.ordcols_import()
		excvars=list(ordcols_full[(ordcols_full['Do']=='Exclude')]['column'])
		ordvars=list(ordcols['column'])
		ctsvars=[col for col in df.columns if re.search('float',str(df[col].dtype))
        and not re.search(self.excs_cts,col)]
		ohe_vars=[col for col in df.columns if col not in ordvars+ctsvars+excvars
        and not re.search(self.excs_cts,col)]

		excluded_vars=[c for c in df.columns if c not in ordvars+ctsvars+ohe_vars]
		return ctsvars,ohe_vars,excluded_vars,ordvars

	def ord_data(self,df):
		ordcols=self.ordcols_import()[0]
		ordvars=[col for col in ordcols['column']]
		df_ord=df[['eid']+ordvars]
		for i,col in enumerate(ordcols['column']):
			df_ord[col]=df[col].map(ast.literal_eval(ordcols['Do'].iloc[i]))
		return df_ord

	def ohe_data(self,df):
		ohe_vars=self.vars_det(df)[0]
		ohe_df,ohe_excluded=self.onehotencoder(df=df[ohe_vars+['eid']],cols=ohe_vars,excwords=self.excwords)
		return ohe_df,ohe_excluded

	def cts_data(self,df):
		ctsvars=self.vars_det(df)[0]
		cts_df=df[ctsvars+['eid']]
		return cts_df

	def model_data(self,df):

		excluded_vars=self.vars_det(df)[2]
		ohe_df=self.ohe_data(df)
		cts_df=self.cts_data(df)
		df_ord=self.ord_data(df)
		df_model=pd.merge(df_ord,cts_df,on='eid',how='left')
		df_model=pd.merge(df_model,ohe_df,on='eid',how='left')
		df_model=pd.merge(df[self.keycols],df_model,on='eid',how='left')

		return df_model,excluded_vars

	
		


	

class data_fixes(object):

	def __init__(self):
		"""
		Initilising models.
		"""
		self.path="/Users/michaelallwright/Dropbox (Sydney Uni)/michael_PhD/Projects/UKB/Data/"








