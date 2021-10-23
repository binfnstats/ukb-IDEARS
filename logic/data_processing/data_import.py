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

class dataload(object):
	"""
	Incexp Model for affordability.
	"""

	def __init__(self):
		"""
		Initilising models.
		"""
		self.path="/Users/michaelallwright/Dropbox (Sydney Uni)/michael_PhD/Projects/UKB/Data/"
		self.inpfile='all_092021.csv'
		self.fullfile='ukb_tp0_new.parquet'
		self.exclusions=excs='source_of_report|first_reported|icd10|icd9|operative_procedures|treatment_speciality|external_ca|patient_recoded|\
	hospital_polymorphic|_report|assay_date|device_id'


	def findcols(self,df,string):
		return [col for col in df if string in col]

	def nonnull_eids(self,df,col):
		eids=df['eid'][pd.notnull(df[col])]
		df_out=df[['eid',col]][pd.notnull(df[col])]
		return eids,df_out


	def convert_null(df):
		for col in df.columns:
			mask=(df[col]=='Prefer not to answer')
			df[col][mask]=np.NaN
		return df

	def read_all_samp(self,file='all_092021.csv'):
		df=pd.read_csv(self.path+file,nrows=100)
		return df

	def treatcols(self):
		df=self.read_all_samp()
		treatcols=self.findcols(df,'treatmentmedication_code')
		return treatcols

	def ICD10_out(self):
		df=self.read_all_samp()
		ICD10cols=[col for col in df.columns if '41270' in col or '41280' in col or 'eid' in col]
		icdextcols=['age_when_attended_assessment_centre_f21003_0_0','date_of_attending_assessment_centre_f53_0_0',\
	'date_of_death_f40000_0_0']
		ICD10cols=ICD10cols+icdextcols
		df=pd.read_csv('%s%s' % (self.path,self.inpfile),usecols=ICD10cols)
		df.to_parquet('%s%s' % (self.path,'ICD10s_test.parquet'))
		return df


	def famhist(self,df):
		cols_famhist=['eid']+findcols(df,'illnesses_of_father')+findcols(df,'illnesses_of_mother')
		df_fam_hist=pd.read_csv(self.path+'all_092021.csv',usecols=cols_famhist)
		df_fam_hist['eid']=df_fam_hist['eid'].astype(str)
		df_fam_hist.to_parquet(path+'df_fam_hist.parquet')
		return df_fam_hist

	def loadfullfile(self):
		df=pd.read_parquet(self.path+self.fullfile)
		return df

	def output_treats(self,load=0):
		treatcols=self.treatcols()
		df=pd.read_csv(self.path+self.inpfile,usecols=list(treatcols+['eid']))
		df['eid']=df['eid'].astype(str)
		df.to_parquet(self.path+'ukb_treatments_test.parquet')
		return df

	def disease_cols(self,df):
		disease_cols=self.findcols(df,'first_reported')+['eid']+['date_of_attending_assessment_centre_f53_0_0']
		return disease_cols

	def death_eids(self,df):
		death_eids=self.nonnull_eids(df,'date_of_death_f40000_0_0')[0]
		return death_eids

	def non_missing_cols(self,df,ratio=0.8):
		for col in df.columns:
			df[col][(df[col]=='nan')]=np.NaN
			cols_rat=[col for col in df.columns if df[col].count()/df.shape[0]>ratio and not re.search(excs,col)]
			cols_rat_df=pd.DataFrame(cols_rat)
			cols_rat_df.columns=['colname']

		return cols_rat_df

	def export_df(self,df,cols,file):
		df=df[cols]
		df.to_parquet(self.path+file)
		return df



