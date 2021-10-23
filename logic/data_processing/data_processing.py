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

class data_proc(object):
	"""
	Incexp Model for affordability.
	"""

	def __init__(self):
		"""
		Initilising models.
		"""
		self.path="/Users/michaelallwright/Dropbox (Sydney Uni)/michael_PhD/Projects/UKB/Data/"
		self.fullfile='ukb_tp0_new.parquet'
		self.dis_map={'cerebrovasc':['I60','I62','I63','I65','I66','I67','I68','I69'],
		 'stroke':['I64'],'TBI':['S0'],'Hear_loss':['H919','H901','H902','H903','H904',\
	'H905','H906','H907','H908','H909','H910','H911','H912','Z974']}
		self.icd10_file='ukb_ICD10.parquet'
		self.year=2021
		self.month=1
		self.day=31

		self.years_wait=2
		self.years_max=10
		self.min_part_dis=200




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

		cols=[k for k in dis_map]
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






