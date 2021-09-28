# -*- coding: utf-8 -*-
# @Author: Zhitao Xiong
# @Date:   2021-09-16 12:53:33
# @Last Modified by:   Zhitao Xiong
# @Last Modified time: 2021-09-17 08:55:19
#%%[markdown]
"""
# Why
Entry point for the codebase

# What
This script links all parts together to produce the insights that are needed: Expenses, Income, Assets, Liabilities

# Who
Zhitao Xiong

# Extra
You should run this as a notebook within Visual Studio Code
"""
#%%
import pandas as pd
from multiprocessing import Process

from finpass_utils.affordability_cat import extract_affordability_cat
from finpass_utils.db_getjointaccount import get_joint_accounts
from finpass_utils.db_gettransactions import get_transactions
from finpass_utils.enrichment import enrich
from finpass_utils.income import extract_income

#%%
class FinPassModel(object):
    """
    FinPass Model for generating insights: Expenses, Income, Assets, Liabilities
    """

    def __init__(self):
        """
        Initilising all required files for enrichment
        """
        self.M1 = 'M1'

        self.A1 = 'A1'
        
        print('All init DONE for enrichment files')

    def predict_finpass(self, X, features_names=None) -> pd.DataFrame:
        """
        Return the four numbers needed with all statitics.

        Parameters
        ----------
        X : dataframe with all users' IDs that the FinPass needs to be generated
        feature_names : array of feature names (optional)
        """

        ## CAUTION: parrallelisation will be implemented once components are fixed
        # first, DB getting...
        self.df_get_transactions = get_transactions(X)
        self.df_get_jointaccount = get_joint_accounts(X)

        #second, enrichment, need to pass all tables
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
        aggregate the output for FinPass by using the DFs generated in all components

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
reuslts = finpassmodel_test.predict_finpass(df_test)
# %%
