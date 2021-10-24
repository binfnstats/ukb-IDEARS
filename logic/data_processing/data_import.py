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
import yaml

class dataload():
	import yaml
	"""
	Incexp Model for affordability.
	"""

	def __init__(self,yam=yaml.YAMLObject):
		"""
		Initilising models.
		"""
		
		#self.config=read_yaml(yam)
		self.path="/Users/michaelallwright/Dropbox (Sydney Uni)/michael_PhD/Projects/UKB/Data/"
		self.inpfile='all_092021.csv'
		self.fullfile='ukb_tp0_new.parquet'
		self.exclusions='source_of_report|first_reported|icd10|icd9|operative_procedures|treatment_speciality|external_ca|patient_recoded|\
	hospital_polymorphic|_report|assay_date|device_id'

		self.PDcols=['eid','worked_with_pesticides_f22614_0_0','home_area_population_density_urban_or_rural_f20118_0_0',
       'single_episode_of_probable_major_depression_f20123_0_0','probable_recurrent_major_depression_moderate_f20124_0_0',
		'probable_recurrent_major_depression_severe_f20125_0_0','bipolar_and_major_depression_status_f20126_0_0',
		'neuroticism_score_f20127_0_0' ,'recent_feelings_or_nervousness_or_anxiety_f20506_0_0',
		'daytime_dozing_sleeping_narcolepsy_f1220_0_0']




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

	def read_all_samp(self):
		df=pd.read_csv(self.path+self.inpfile,nrows=100)
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

	def PD_specific_out(self):
		"""
		Get specific variables required for PD based on meta-analyses
		"""
		df=pd.read_csv('%s%s' % (self.path,self.inpfile),usecols=self.PDcols)
		df['eid']=df['eid'].astype(str)
		df.to_parquet('%s%s' % (self.path,'PD_specific.parquet'))
		return df


	def famhist(self,df):
		cols_famhist=['eid']+self.findcols(df,'illnesses_of_father')+self.findcols(df,'illnesses_of_mother')
		df_fam_hist=pd.read_csv(self.path+'all_092021.csv',usecols=cols_famhist)
		df_fam_hist['eid']=df_fam_hist['eid'].astype(str)
		df_fam_hist.to_parquet(self.path+'df_fam_hist_test.parquet')
		return df_fam_hist

	def deaths_df(self):
		deaths=pd.read_parquet(self.path+self.fullfile,columns=['eid','date_of_death_f40000_0_0'])
		deaths.to_parquet(self.path+'deaths_test.parquet')
		return deaths


	def loadfullfile(self):
		df=pd.read_parquet(self.path+self.fullfile)
		return df

	def df_treats(self,load=0):
		treatcols=self.treatcols()
		df=pd.read_csv(self.path+self.inpfile,usecols=list(treatcols+['eid']))
		df['eid']=df['eid'].astype(str)
		df.to_parquet(self.path+'ukb_treatments_test.parquet')
		return df

	def disease_cols(self,df):
		disease_cols=self.findcols(df,'first_reported')+['eid']+['date_of_attending_assessment_centre_f53_0_0']
		return disease_cols

	def import_ukb_disease(self):
		df=self.read_all_samp()
		diseasecols=self.disease_cols(df)
		df=pd.read_csv(self.path+self.inpfile,usecols=list(diseasecols+['eid']))
		df['eid']=df['eid'].astype(str)
		df.to_parquet(self.path+'ukb_diseases_test.parquet')

		return df


	def death_eids(self,df):
		death_eids=self.nonnull_eids(df,'date_of_death_f40000_0_0')[0]
		return death_eids

	def non_missing_cols(self,df,ratio=0.8):
		for col in df.columns:
			df[col][(df[col]=='nan')]=np.NaN
			cols_rat=[col for col in df.columns if df[col].count()/df.shape[0]>ratio and not re.search(self.excs,col)]
			cols_rat_df=pd.DataFrame(cols_rat)
			cols_rat_df.columns=['colname']

		return cols_rat_df

	def export_df(self,df,cols,file):
		df=df[cols]
		df.to_parquet(self.path+file)
		return df

		




