import pandas as pd
import numpy as np
from datetime import datetime

import seaborn as sns
import os
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt
from decimal import Decimal

from scipy import stats
from scipy.stats import t
from sklearn.preprocessing import StandardScaler


class AnalysisCharts(object):
	"""
	This produces the charts qhich show dose dependency
	"""

	def __init__(self):
		self.path="/Users/michaelallwright/Documents/data/ukb/"

	def slope(self,df,var,depvar):

		df1=df.copy()
	
		mask=(pd.notnull(df1[var]))
		df1=df1.loc[mask,]

		trans = StandardScaler()

		#standardise variable
		df1[var+'_std']=trans.fit_transform(np.asarray(df1[var]).reshape(-1, 1))
		
		# linear regression of variables
		slope, intercept, r_value, p_value, std_err = stats.linregress(df1[var+'_std'],df1[depvar])
		
		return df1,slope, intercept, r_value, p_value, std_err

	def pvalue_slopes(self,df,var,depvar='AD',splitvar='APOE4_Carriers',splitval=0):
		
		df1=df[[splitvar,depvar,var]].copy()
		mask=(df1[splitvar]>splitval)

		#print(df1[var])
		mask_nonull=pd.notnull(df1[var])

		# lower and upper dataframes
		df1_l=df1.loc[~mask&mask_nonull,]
		df1_u=df1.loc[mask&mask_nonull,]

		df1_l,slope_l, intercept_l, r_value_l, p_value_l, std_err_l = self.slope(df1_l,var,depvar)
		df1_u,slope_u, intercept_u, r_value_u, p_value_u, std_err_u = self.slope(df1_u,var,depvar)
		

		#z score
		numerator = slope_u - slope_l
		denominator = pow((pow(std_err_u,2) + pow(std_err_l,2)), 1/2)
		z=numerator/denominator  
		print(z)

		p_value = stats.norm.sf(abs(z))
		print(p_value)
		
		#val_comp is 0 if we are comparing the top value as numerator from the split vars else it's 1
		slope_diff=(slope_u/slope_l) 
		
		
		print('Slope Difference: '+str(slope_diff))
		return slope_diff,slope_u,slope_l,p_value,p_value_u,p_value_l
		
