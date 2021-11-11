#!/usr/bin/env python3
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 16:39:52 2021

@author: michaelallwright

This file contains a set of functions that are to be used in the new version of UKB
to do the key analyses

"""

#key information to run financial passport for - user and start/end date. Move to config file?

import pandas as pd
import numpy as np

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
	Incexp Model for affordability.
	"""

	def __init__(self):
		"""
		Initilising models.
		"""
		self.path="/Users/michaelallwright/Dropbox (Sydney Uni)/michael_PhD/Projects/UKB/Data/"
		self.pathfig='/Users/michaelallwright/Documents/GitHub/UKB/PD/figures/'


	def findcols(self,df,string):
		return [col for col in df if string in col]
	
	def df_quint(self,df,var='perc_correct_matches_rounds'):
		if df[var].nunique()>5:
			df[var+'_quint']=pd.qcut(df[var],5,labels=False,duplicates='drop')
		else:
			print('insufficient variable values')
		return df

	def pop_char_dem_func(self,df,depvar='dementia'):
		
		mask_m=(df['sex_f31_0_0']==0)
		maska0=(df['APOE4_Carriers']==0)
		maska1=(df['APOE4_Carriers']==1)
		maska2=(df['APOE4_Carriers']==2)

		n=df.shape[0]
		age=str(round(df['age_when_attended_assessment_centre_f21003_0_0'].mean(),2))+'+/-'+\
	str(round(df['age_when_attended_assessment_centre_f21003_0_0'].std(),2))
		males=str(df[mask_m].shape[0])+' ('+str(round(df[mask_m].shape[0]/df.shape[0],2))+')'
		females=str(df[~mask_m].shape[0])+' ('+str(round(df[~mask_m].shape[0]/df.shape[0],2))+')'
		apoe0=str(df[maska0].shape[0])+' ('+str(round(df[~maska0].shape[0]/df.shape[0],2))+')'
		apoe1=str(df[maska1].shape[0])+' ('+str(round(df[~maska1].shape[0]/df.shape[0],2))+')'
		apoe2=str(df[maska2].shape[0])+' ('+str(round(df[~maska2].shape[0]/df.shape[0],2))+')'

		return n,age,males,females,apoe0,apoe1,apoe2

	def pop_char_dem(self,df,depvar='dementia'):
		vars=['n','Age at baseline (years)','Males','Females','APOE4 (0 alleles)','APOE4 (1 allele)','APOE4 (2 alleles)']
		mask_case=(df[depvar]==1)
		cases=self.pop_char_dem_func(df[mask_case])
		controls=self.pop_char_dem_func(df[~mask_case])
		total=self.pop_char_dem_func(df)

		df_sum=pd.DataFrame({'Variable':vars,'Cases':cases,'Controls':controls,'Total':total})

		return df_sum




	def agenorm2(self,df,var):
		df_sum=pd.DataFrame(df.groupby(['age_when_attended_assessment_centre_f21003_0_0']).agg({var:['mean']})).reset_index()
		df_sum.columns=['age_when_attended_assessment_centre_f21003_0_0','mean'+var]

		df=pd.merge(df,df_sum,on='age_when_attended_assessment_centre_f21003_0_0',how='left')
		df[var]=df[var]/df['mean'+var]
		df.drop(columns=['mean'+var],inplace=True)
		return df

	def gend_norm(self,df,var):
		df_sum=pd.DataFrame(df.groupby(['sex_f31_0_0']).\
	agg({var:['mean']})).reset_index()
		df_sum.columns=['sex_f31_0_0','mean'+var]

		df=pd.merge(df,df_sum,on=['sex_f31_0_0'],how='left')
		df[var]=df[var]/df['mean'+var]
		df.drop(columns=['mean'+var],inplace=True)
		return df

	def gend_norm2(self,df,vars):
		df_sum=pd.DataFrame(df.groupby(['sex_f31_0_0'])[vars].mean()).reset_index()
		df_sum.columns=['sex_f31_0_0']+['mean'+v for v in vars]

		df=pd.merge(df,df_sum,on=['sex_f31_0_0'],how='left')
		df[vars]=df[vars]/df[['mean'+ v for v in vars]]
		df.drop(columns=['mean'+v for v in vars],inplace=True)
		return df

	def age_gend_norm2(self,df,var):
		df_sum=pd.DataFrame(df.groupby(['age_when_attended_assessment_centre_f21003_0_0','sex_f31_0_0']).\
	agg({var:['mean']})).reset_index()
		df_sum.columns=['age_when_attended_assessment_centre_f21003_0_0','sex_f31_0_0','mean'+var]

		df=pd.merge(df,df_sum,on=['age_when_attended_assessment_centre_f21003_0_0','sex_f31_0_0'],how='left')
		df[var]=df[var]/df['mean'+var]
		df.drop(columns=['mean'+var],inplace=True)
		return df

	def age_gend_norm_mult(self,df,vars):
		df_sum=pd.DataFrame(df.groupby(['age_when_attended_assessment_centre_f21003_0_0','sex_f31_0_0'])[vars].mean()).reset_index()
   
		df_sum.columns=['age_when_attended_assessment_centre_f21003_0_0','sex_f31_0_0']+['mean'+v for v in vars]

		df=pd.merge(df,df_sum,on=['age_when_attended_assessment_centre_f21003_0_0','sex_f31_0_0'],how='left')

		for v in vars:
			df[v]=df[v]/df['mean'+v]
		df.drop(columns=['mean'+v for v in vars],inplace=True)
		return df

		
	def std_scale(self,df,vars=[]):
		for var in vars:
			trans = StandardScaler()
			df[var+'std']=trans.fit_transform(np.asarray(df[var]).reshape(-1, 1))
		return df

	def std_scale_newvar(self,df,vars=[],name='inflammation'):
		for var in vars:
			trans = StandardScaler()
			df[var+'std']=trans.fit_transform(np.asarray(df[var]).reshape(-1, 1))
		
		df[name]=df[[v+'std' for v in vars]].sum(axis=1)
		df.drop(columns=[v+'std' for v in vars],inplace=True)
		return df




	def slope(self,df,var,depvar):
	
		mask_aspnnull=(pd.notnull(df[var]))
		trans = StandardScaler()
		df[var+'_std']=trans.fit_transform(np.asarray(df[var]).reshape(-1, 1))
		
		slope_ap, intercept_ap, r_value_ap, p_value_ap, std_err_ap = \
		stats.linregress(df[mask_aspnnull][var+'_std'],df[mask_aspnnull][depvar])
		
		slope_ap=slope_ap
		
		return slope_ap, intercept_ap, r_value_ap, p_value_ap, std_err_ap

	def calc_rr(self,df,var,slicevar='APOE4 Status',splitval=0,depvar='dementia',
		xlabel='Total number of conditions at baseline',ylabel='Relative Risk',leg=0,quint=1):

		if quint==1:
			df=self.df_quint(df,var=var)
			var=var+'_quint'

		df_sum=pd.DataFrame(df.groupby([slicevar,var]).agg({depvar:['sum','count']})).reset_index()
		df_sum.columns=[slicevar,var,depvar+'_sum',depvar+'_count']
		df_sum['UI']=df_sum[slicevar].astype(str)+'_'+df_sum[var].astype(str)
		df_sum['ART']=df_sum[depvar+'_sum']/df_sum[depvar+'_count']
		
		df_sum['ARC']=0
		df_sum['RR']=0
		
		slope_diff,pval=self.pvalue_slopes(df,var=var,depvar=depvar,splitvar=slicevar,splitval=splitval)
		
		for q in df_sum['UI'].unique():
			mask=(df_sum['UI']==q)
			ARC=df_sum[~mask][depvar+'_sum'].sum()/df_sum[~mask][depvar+'_count'].sum()
			df_sum['ARC'][mask]=ARC
			df_sum['RR']=df_sum['ART']/df_sum['ARC']
			
		df_sum2=pd.DataFrame(df_sum.groupby([slicevar,var])['RR'].mean().unstack(slicevar)).reset_index()
		figure(figsize=(15, 10))
		
		ax=sns.lineplot(data=df_sum, x=var, y='RR',hue=slicevar,estimator='mean',palette = 'Greys_r',linewidth = 4)#,palette = Greys_r)
		ax.set(xlabel=xlabel,ylabel=ylabel)
		
		plt.setp(ax.get_legend().get_texts(), fontsize='30') # for legend text
		plt.setp(ax.get_legend().get_title(), fontsize='32') # for legend title
		
		if leg==0:
			ax.get_legend().remove()
		
		plt.xticks(fontsize='24')
		plt.yticks(fontsize='24')
		plt.xlabel(xlabel, fontsize=24)
		plt.ylabel(ylabel, fontsize=24)
		
		if pval<0.01:
			symb="**"
		elif pval<0.05:
			symb="*"
		else:
			symb=" (ns)"
		if round(pval,4)==0:
			valsymb="{:.2E}".format(Decimal(pval))
		else:
			valsymb=str(round(pval,5))
			
		
			
		plt.text(0.5,0.8,'slope ratio: '+str("{:.0%}".format(round(slope_diff,5))),horizontalalignment='center',
				verticalalignment='center', transform = ax.transAxes, fontsize='24')
		plt.text(0.5,0.7,'(p = '+valsymb+symb+')',horizontalalignment='center',
				verticalalignment='center', transform = ax.transAxes, fontsize='24')
		plt.savefig(self.path+'fig3'+"_"+var+'.svg', dpi=300)
		plt.show()

	def pvalue_slopes(self,df,var,depvar,splitvar,splitval):
	
		mask_aspnnull=(pd.notnull(df[var]))
		mask_split=(df[splitvar]>splitval)

		trans = StandardScaler()
		df[var+'_std']=trans.fit_transform(np.asarray(df[var]).reshape(-1, 1))
		
		slope_ap, intercept_ap, r_value_ap, p_value_ap, std_err_ap = \
		stats.linregress(df[mask_aspnnull][var+'_std'],df[mask_aspnnull][depvar])
		
		
		slope1, intercept1, r_value1, p_value1, std_err1 = \
		stats.linregress(df[mask_aspnnull&mask_split][var+'_std'],df[mask_aspnnull&mask_split][depvar])

		slope2, intercept2, r_value2, p_value2, std_err2 = \
		stats.linregress(df[mask_aspnnull&~mask_split][var+'_std'],df[mask_aspnnull&~mask_split][depvar])
		
		
		
		x=df[mask_aspnnull&mask_split][var+'_std']
		tinv = lambda p, df: abs(t.ppf(p/2, df))
		ts = tinv(0.05, len(x)-2)
		CI1=ts*std_err1

		x=df[mask_aspnnull&~mask_split][var+'_std']
		tinv = lambda p, df: abs(t.ppf(p/2, df))
		ts = tinv(0.05, len(x)-2)
		CI1=ts*std_err2


		numerator = slope1 - slope2
		denominator = pow((pow(std_err1,2) + pow(std_err2,2)), 1/2)
		z=numerator/denominator  
		print(z)

		p_value = stats.norm.sf(abs(z))
		print(p_value)
		
		slope_diff=(slope2/slope1) -1
		
		print('Slope Difference: '+str(slope_diff))
		
		
		return slope_diff,p_value

	def varsplit(self,df,splitvar,splitval,cols,depvar):
		print("yah!")
		mask=(df[splitvar]>splitval)
		allvals=[]
		split1vals=[]
		split2vals=[]
		for col in [c for c in cols if c!=depvar]: 
			#if 'age_when' not in col:
				#df=self.agenorm2(df,col)
			allval=self.slope(df=df,var=col,depvar=depvar)
			splitval1=self.slope(df=df[mask],var=col,depvar=depvar)
			print(df[mask].shape)
			print(df[mask]['PD'].sum())
			splitval2=self.slope(df=df[~mask],var=col,depvar=depvar)
			allvals.append(allval)
			split1vals.append(splitval1)
			split2vals.append(splitval2)

		sum_df=pd.DataFrame({'column_name':[c for c in cols if c!=depvar],'allvals':allvals,
			'split1vals':split1vals,'split2vals':split2vals})
		for var in ['allvals','split1vals','split2vals']:
			sum_df[var+'_slope']=sum_df[var].apply(lambda x:x[0])
			sum_df[var+'_slope_pval']=sum_df[var].apply(lambda x:x[3])
		sum_df.drop(columns=['allvals','split1vals','split2vals'],inplace=True)

		return sum_df

	def ttest(self,df,bdown,var):
		ttest_vals=stats.ttest_ind(df[(df['dis_stage']==bdown)][var], 
				   df[(df['dis_stage']=='No PD')][var])
		
		return ttest_vals

	def disease_traj(self,dis_date='parkins_date',disease='PD',vars=['igf1_f30770_0_0'],
	splitvar='sex_f31_0_0',agemin=50,agemax=70,labels=['Female','Male'],varnames='IGF1',plots='lineplots',ttest_use=False):

		"""
		Brings in the full data and disease date so we can model variables averages across trajectory of patient journey
		
		"""

		colsimport=list(set(list(['eid','age_when_attended_assessment_centre_f21003_0_0',splitvar,disease]+vars)))
	
		df_model=pd.read_parquet(self.path+'df_all_final.parquet',columns=colsimport)
		df_dates=pd.read_parquet(self.path+'labels_dates_test.parquet')[['eid',dis_date]]
		df_model_orig=pd.read_parquet(self.path+'df_model_test.parquet')[['eid','date_of_attending_assessment_centre_f53_0_0']]

		df_dates['eid']=df_dates['eid'].astype(str)
		df_model['eid']=df_model['eid'].astype(str)
		df_model_orig['eid']=df_model_orig['eid'].astype(str)
		df_test=pd.merge(df_model,df_dates,on='eid',how='left')
		df_test=pd.merge(df_test,df_model_orig,on='eid',how='left')
		mask3=(df_test['age_when_attended_assessment_centre_f21003_0_0']>=agemin)&\
	 (df_test['age_when_attended_assessment_centre_f21003_0_0']<=agemax)

		mask_keep=(df_test[splitvar]==0)|(df_test[splitvar]==1)&mask3
		print(df_test.shape)
		df_test=df_test[mask_keep]
		print(df_test.shape)

		df_test['disease']=0
		mask=pd.notnull(df_test[dis_date])
		df_test['disease'][mask]=1

		df_test['date_of_attending_assessment_centre_f53_0_0']=\
	pd.to_datetime(df_test['date_of_attending_assessment_centre_f53_0_0'])
		df_test['years_'+disease]=-round((df_test[dis_date]-df_test['date_of_attending_assessment_centre_f53_0_0']).dt.days/365)
		



		k=len(vars)

		fig = plt.figure(figsize=(15*k, 10*k))
		grid = plt.GridSpec(k, k, hspace=0.3, wspace=0.3)

		ttestvals=[]
		pvallist=[]
		varnameslist=[]
		splitnames=[]
		comp_groups=[]

		
		for j,v in enumerate(vars):
			for i,t in enumerate(df_test[splitvar][pd.notnull(df_test[splitvar])].unique()):

				#print(i)
				#print(labels[i])
				mask=pd.isnull(df_test[dis_date])
				mask_split=(df_test[splitvar]==i)
				mask2=((df_test['years_'+disease]>-10)&(df_test['years_'+disease]<5))|pd.isnull(df_test['years_'+disease])
				mask3=(df_test['age_when_attended_assessment_centre_f21003_0_0']>=agemin)&\
				(df_test['age_when_attended_assessment_centre_f21003_0_0']<=agemax)

			   
				ax=fig.add_subplot(grid[j, i])
				avg=df_test[mask&mask3&mask_split][v].mean()

				mask_12=(df_test['years_PD']<-5)&(df_test['years_PD']>-10)
				mask_5=(df_test['years_PD']<-0)&(df_test['years_PD']>=-5)
				mask_05=(df_test['years_PD']<5)&(df_test['years_PD']>=0)
				mask_no_pd=(pd.isnull(df_test['years_PD']))&(df_test['PD']==0)


				mask_v=(df_test[v]<df_test[v].quantile(0.95))&(df_test[v]>=df_test[v].quantile(0.05))
				df_test['dis_stage']=np.nan
				df_test['dis_stage'][mask_12]='-10 to -5 yrs'#"2: 5-10 years pre diag"
				df_test['dis_stage'][mask_5]='-5 to 0 yrs'#"3: 0-5 years pre diag"
				df_test['dis_stage'][mask_05]='0 to 5 yrs'#"4: 0-5 years post diag"
				df_test['dis_stage'][mask_no_pd]='No PD'#"1: No PD"

				if plots=='lineplots':

					

					ax=sns.lineplot(data=df_test[mask2&mask3&mask_split], 
					x='years_'+disease, y=v,estimator='mean',
					palette = 'Greys_r',linewidth = 4)
					ax.axhline(avg,color='red')

					range_vals=df_test[mask2&mask3&mask_split][v].quantile(0.95)-\
				df_test[mask2&mask3&mask_split][v].quantile(0.05)


					plt.text(0,max(0,avg+range_vals/100),labels[i]+' levels for non PD',fontsize=24)
					plt.title=str(v) + str(labels[i])

					plt.xticks(fontsize='24')
					plt.yticks(fontsize='24')
					plt.xlabel('years_'+disease+'_'+str(labels[i]), fontsize=24)
					plt.ylabel(v, fontsize=24)
					#plt.savefig('figures/'+'fig4'+"_"+v+' '+str(labels[i])+'.svg', dpi=300)
					#plt.show()

				elif plots=='boxplots':


					mask_use=mask2&mask3&mask_split
					df_test.sort_values(by='dis_stage',inplace=True)
					ax=sns.boxplot(x=df_test['dis_stage'][mask_use],y=df_test[v][mask_use],
						order=['No PD','-10 to -5 yrs','-5 to 0 yrs','0 to 5 yrs'],showfliers = False,color='skyblue')
					plt.xticks(fontsize='24')
					plt.yticks(fontsize='24')
					#plt.ylabel(v, fontsize=24)
					plt.ylabel('',fontsize='24')
					plt.xlabel(labels[i]+'s :'+v, fontsize=24)

					#whisker locations - find top within group whisker
					max_val=df_test[v][mask_use].max()
					min_val=df_test[v][mask_use].min()
					q_25=df_test[v][mask_use].quantile(0.25)
					q_75=df_test[v][mask_use].quantile(0.75)
					iqr=q_75-q_25
				   
					iqr_pos = q_75+1.5*iqr if (q_75+1.5*iqr)<max_val else max_val
					iqr_neg = q_25-1.5*iqr if (q_25-1.5*iqr)>min_val else min_val
					


					# statistical annotation
					#x1, x2 = 0, 3   # columns 'Sat' and 'Sun' (first column: 0, see plt.xticks())
					#y, h, col = iqr_pos + (iqr_pos-iqr_neg)/5, (iqr_pos-iqr_neg)/20, 'k'
					#plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
					#plt.text((x1+x2)*.5, y+h, "ns", ha='center', va='bottom', color=col)


					#if df_test[mask_use][v].nunique()>5:
					   # plt.ylim(df_test[mask_use][v].quantile(0),df_test[mask_use][v].quantile(0.99))
					#plt.savefig('figures/'+'fig4'+"_"+v+' '+str(labels[i])+'.svg', dpi=300)
					

					

				if ttest_use:

					#print(df_test['dis_stage'].value_counts())

					mask_v=pd.notnull(df_test[v])

					mask_use=mask2&mask3&mask_split&mask_v

					#compgroups=list(['1: No PD','2: 5-10 years pre diag','3: 0-5 years pre diag','4: 0-5 years post diag'])
					compgroups=list(['No PD','-10 to -5 yrs','-5 to 0 yrs','0 to 5 yrs'])
				   
					iqr_pos_arr=[]
					iqr_neg_arr=[]
					pvallist_small=[]
					for q,m in enumerate(compgroups):

						ttest_vals=self.ttest(df_test[mask_use],m,v)


						mask_dis_stage=(df_test['dis_stage']==m)

						max_val=df_test[v][mask_use&mask_dis_stage].max()
						min_val=df_test[v][mask_use&mask_dis_stage].min()
						q_25=df_test[v][mask_use&mask_dis_stage].quantile(0.25)
						q_75=df_test[v][mask_use&mask_dis_stage].quantile(0.75)
						iqr=q_75-q_25

						iqr_pos = q_75+1.5*iqr if (q_75+1.5*iqr)<max_val else max_val
						iqr_pos_arr.append(iqr_pos)
						iqr_neg = q_25-1.5*iqr if (q_25-1.5*iqr)>min_val else min_val
						iqr_neg_arr.append(iqr_neg)

					   
						ttest_val_inc=round(df_test[mask_use&mask_dis_stage][v].mean(),3)#round(ttest_vals[0],3)
						pval_inc=round(ttest_vals[1],6)
						ttestvals.append(ttest_val_inc)
						pvallist.append(pval_inc)
						pvallist_small.append(pval_inc)
						varnameslist.append(v)
						splitnames.append(i)
						comp_groups.append(m)

					# statistical annotation



					for k in [1,2,3]:
						sig='(ns)'

						if pvallist_small[k]<0.001:
							sig='(***)'
						elif pvallist_small[k]<0.005:
							sig='(**)'
						elif pvallist_small[k]<0.05:
							sig='(*)'
						x1, x2 = 0, k   # columns 'Sat' and 'Sun' (first column: 0, see plt.xticks())
						y, h, col = iqr_pos + (iqr_pos_arr[k]-iqr_neg_arr[k])/5, k*(iqr_pos_arr[k]-iqr_neg_arr[k])/5, 'black'
						plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
						plt.text((x1+x2)*.5, y+h, 'p= '+str(pvallist_small[k])+' '+str(sig), ha='center', va='bottom', color=col,
							fontsize='24')


					

		
		plt.savefig(self.pathfig+'fig4'+"_"+varnames+'.jpg', dpi=300,bbox_inches='tight')
		plt.show()
		if ttest_use:
			pvals_df=pd.DataFrame({'var':varnameslist,'split':splitnames,'compgroup':comp_groups,'mean_val':ttestvals,'pvals':pvallist})
			df_test=[df_test,pvals_df]



		return df_test



