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
path="../../Data/"
import seaborn as sns
import os
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt
from decimal import Decimal
from FIBRSF import maskapoedf
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
        self.path="../../Data/"


    def findcols(self,df,string):
        return [col for col in df if string in col]
    
    def df_quint(self,df,var='perc_correct_matches_rounds'):
        if df[var].nunique()>5:
            df[var+'_quint']=pd.qcut(df[var],5,labels=False,duplicates='drop')
        else:
            print('insufficient variable values')
        return df

    def agenorm2(self,df,var):
        df_sum=pd.DataFrame(df.groupby(['age_when_attended_assessment_centre_f21003_0_0']).agg({var:['mean']})).reset_index()
        df_sum.columns=['age_when_attended_assessment_centre_f21003_0_0','mean'+var]

        df=pd.merge(df,df_sum,on='age_when_attended_assessment_centre_f21003_0_0',how='left')
        df[var]=df[var]/df['mean'+var]
        df.drop(columns=['mean'+var],inplace=True)
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
        xlabel='Total number of conditions at baseline',ylabel='Relative Risk',leg=0):

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
            if 'age_when' not in col:
                df=self.agenorm2(df,col)
            allval=self.slope(df=df,var=col,depvar='dementia')
            splitval1=self.slope(df=df[mask],var=col,depvar='dementia')
            splitval2=self.slope(df=df[~mask],var=col,depvar='dementia')
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



