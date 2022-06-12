# some code around remapping variables to ordinal for psycho-social vars

import ast

cols=[]
u_vals=[]
for c in df_psych_soc.columns:
    if df_psych_soc[c].nunique()<10 and df_psych_soc[c].count()>300000:
        cols.append(c)
        uvals_inc=str(set(df_psych_soc.loc[pd.notnull(df_psych_soc[c]),c].unique()))
        u_vals.append(uvals_inc)
        
df_sum=pd.DataFrame({'col':cols,'u_values':u_vals})
dict1=dict({'Yes':1, 'Prefer not to answer':np.nan, 'Do not know':np.nan, 'No':0}  )   

dict2=dict({'Prefer not to answer':np.nan, 'Not at all':0,
            'Several days':2, 'More than half the days':1, 'Nearly every day':3, 'Do not know':np.nan})

dict3=dict({'2-4 times a week':5, 'About once a month':3, 'Prefer not to answer':np.nan,
            'Once every few months':2,
       'About once a week':4, 'Almost daily':6, 'Never or almost never':1, 
       'No friends/family outside household':0, 'Do not know':np.nan} ) 

map_dict=dict({"'"+str({'Yes', 'Prefer not to answer', 'Do not know', 'No'})+"'":dict1,
     "'"+str({'Prefer not to answer', 'Not at all',
      'Several days', 'More than half the days', 'Nearly every day', 'Do not know'})+"'":dict2,
     "'"+str({'Prefer not to answer', 'Not at all', 'Do not know', 'More than half the days',
              'Nearly every day', 'Several days'})+"'":dict2,
     "'"+str({'2-4 times a week', 'About once a month', 'Prefer not to answer', 'Once every few months',
              'About once a week', 'Almost daily', 'Never or almost never', 
              'No friends/family outside household', 'Do not know'})+"'":dict3,
      "'"+str({'2-4 times a week', 'About once a month', 'Prefer not to answer',
                'Once every few months', 'About once a week', 'Almost daily', 'Never or almost never', 
                'Do not know'})+"'":dict3,
     "'"+str({'No', 'Yes', 'Prefer not to answer', 'Do not know'})+"'":dict1
     })
df_sum['u_values'].apply(map_dict)

df_sum_maps=pd.DataFrame(df_sum['u_values'].value_counts()).reset_index()