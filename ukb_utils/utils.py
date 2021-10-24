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
import datetime as dt

class basic_funcs(object):
	"""
	Module for feature engineering
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
		
	def convert_null(self,df):
		for col in df.columns:
			mask=(df[col]=='Prefer not to answer')|(df[col]=='nan')
			df[col][mask]=np.NaN
		return df