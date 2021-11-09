# -*- coding: utf-8 -*-
# @Author: Michael Allwright
# @Date:   2021-09-16 12:53:33
# @Last Modified by:   Michael Allwright
# @Last Modified time: 2021-09-17 08:55:19
#%%[markdown]
"""
# Why
Entry point for the codebase

# What
This script links all parts together to produce the insights that are needed: 

# Who
Michael Allwright

# Extra
You should run this as a notebook within Visual Studio Code
"""

import pandas as pd



class IDEARS_pipeline(object):
    """
    FinPass Model for generating insights: Expenses, Income, Assets, Liabilities
    """

    def __init__(self):
        """
        Initilising all required files for data processing
        """
        self.M1 = 'M1'

        self.A1 = 'A1'
        
        print('All init DONE for data processing')

    def check(self, X, features_names=None) -> pd.DataFrame:
        """
        Return the four numbers needed with all statitics.

        Parameters
        ----------
        X : dataframe with all variables required to model
        feature_names : array of feature names (optional)
        """

        #creation of raw data from csv file - should be one off and then stored as parquet in
        #relevant subfiles

        #all data processing steps run after

        #merge step

        #study variable creation

        #model run and SHAP

        #Analysis



        ## CAUTION: parrallelisation will be implemented once components are fixed
        # first, DB getting...
        self.df_get_transactions = get_transactions(X)
        self.df_get_jointaccount = get_joint_accounts(X)

        
        self.df_result_enrichment = enrich(self.df_get_transactions)

        #third, affordability categorisation
        self.df_result_affordability_cat = extract_affordability_cat(self.df_result_enrichment)
        
        #Fourth, income extraction
        self.df_result_income = extract_income(self.df_result_affordability_cat)

        print(self.df_result_income)
        results = self.aggregate()

        return results # nothing for now
        
    def aggregate(self) -> pd.DataFrame:
        """
        Aggregate output

        Parameters
        ----------
        None, using internal variables
        """
        #Now do the aggregation
        results = pd.DataFrame()
        results['Expenses'] = 0 # should be using all those DFs to generate results
        results['Income'] = 0
        results['Assets'] = 0
        results['Liabilities'] = 0

        #Now do the rest of the aggregations needed
        results['accounts'] = 'account 1'
        

        return results

#%%
finpassmodel_test = FinPassModel()

uuids = ['123'] # can handle more than one

df_test = pd.DataFrame({'uuid':pd.Series(uuids, dtype='str')})

#%%
df_test
#%%
results = finpassmodel_test.predict_finpass(df_test)
# %%
