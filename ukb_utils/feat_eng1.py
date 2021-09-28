'''
File to convert the raw UKB datafile into ML ready ordinal, one hot encoded and continuous features
Also removes the variables for which we have an insufficiently large sample size
'''


ukb_tp0=pd.read_parquet('%s%s' % (path,'ukb_tp0_new.parquet'))
vars2=pd.read_csv("newvarsdict2.csv")

def ordinalmapping(df,refdf,minrat=0.8):
    
    df2=df.copy()
    ordcols=[]
    for col in df.columns:
        if col in refdf['column'].values:
            
            mask=(refdf['column']==col)
            dictused=np.asarray(refdf[mask]['newdict'])[0]
            df2[col]=df2[col].map(dictused)
            if df2[col].count()/df2[col].shape[0]>=minrat:
                ordcols.append(col)
     
    df2=df2[['eid']+ordcols]
    return df2,ordcols

def retctscols(df,minctcts=10,minrat=0.8):
    cts_cols=[col for i,col in enumerate(df.columns) if re.search(str(df.dtypes[i]),'float64|int')
              and df[col].nunique()>minctcts and df[col].count()/df[col].shape[0]>minrat or col=='eid']
    return df[cts_cols],cts_cols

def onehotencoder(df,cols,excwords,maxrecs=10,mincount=0.8):
    
    #create nulls where unknown for future imputation
    for col in df.columns:
        mask_exc=(df[col].isin(excwords))
        df[col][mask_exc]=np.nan  
    ohe_cols=\
    [col for col in cols if len(df[col].value_counts())<maxrecs
     and df[col].count()/df[col].shape[0]>mincount]
    
    print(r'Total ohe variables = %0.0f ' % (len(ohe_cols)))
        
    df_ohe_cols=df[ohe_cols]
         
    df_out=pd.get_dummies(df_ohe_cols, prefix=df_ohe_cols.columns,
               columns=df_ohe_cols.columns)
       
    df_out['eid']=df['eid'].tolist()
    
    return df_out

def getdummiesvar(df,var):
    mask_exc=(~df[var].isin(excwords))
    df_var=df[[var]+['eid']][mask_exc]
    return pd.get_dummies(df_var,prefix=var)

def nulllt0(df,col):
    mask_lt0=(df[col]<0)
    df[col][mask_lt0]=np.nan
    print(df[col].value_counts())
    return df

def makenull(df,df_ohe):

    for col1 in df.columns:
        for col2 in [col for col in df_ohe.columns if col1 in col]:
            mask_null=pd.isnull(df[col1])
            df_ohe[col2][mask_null] = np.nan
    return df_ohe

'''
Import mappings for ordinal variables and excluded mappings as well as def one hot encoded vars
'''

varsdict=vars2[pd.notnull(vars2['dicts'])&(vars2['dicts'].str.contains("{"))]
varsdict['newdict']=varsdict['dicts'].apply(eval)
varsdict['index']=varsdict['index'].apply(eval).apply(lambda x:tuple(sorted(x)))

df_ordinal_vars=pd.merge(varsdict[['index','newdict']],df_varnames[['column','vars2']],
               left_on='index',right_on='vars2',how='inner')

varsexclude=vars2[pd.notnull(vars2['dicts'])&(vars2['dicts'].str.contains("Exclude"))]
varsexclude['index']=varsexclude['index'].apply(eval).apply(lambda x:tuple(sorted(x)))

varsohe=vars2[pd.notnull(vars2['dicts'])&(vars2['dicts'].str.contains("ohe"))&
             (~vars2['index'].str.contains("e4/e4"))]
varsohe['index']=varsohe['index'].apply(eval).apply(lambda x:tuple(sorted(x)))

dfexc=pd.merge(varsexclude[['index','dicts']],df_varnames[['column','vars2']],
               left_on='index',right_on='vars2',how='inner')

dfohe=pd.merge(varsohe[['index','dicts']],df_varnames[['column','vars2']],
               left_on='index',right_on='vars2',how='inner')

mask=(dfohe['column'].str.contains('device_id_|_report|_completion_status'))
dfohe=dfohe[~mask]

'''
inclusion criteria for cts columns, ordinal mspping
'''
df_cts,df_cts_cols=retctscols(df=ukb_tp0)
ukb_tp0_mapped,ordcols_set=ordinalmapping(df=ukb_tp0,refdf=df_ordinal_vars)

'''
columns that have already been covered in the above
'''

excwords=['Prefer not to answer','nan','None of the above']
remcols=[col for col in ukb_tp0.columns if col not in ordcols_set and col not in df_cts_cols]
speccols=['sex_f31','average_total_household_income_before_tax_f738','usual_walking_pace_f924',
 'frequency_of_friendfamily_visits_f1031','drive_faster_than_motorway_speed_limit_f1100',
          'weekly_usage_of_mobile_phone_in_last_3_months_f1120','qualifications_f6138',
         'gender','avgincome','walkspeed','freqfriendfamily','faster_mot_speed','weekly_mobphone_mins',
          'qualif_score','APOE4_Carriers']

keepcols=[col for col in ukb_tp0.columns if col not in ordcols_set and col not in df_cts_cols and col 
          not in speccols] 

ohe_df=onehotencoder(df=ukb_tp0[keepcols+['eid']],cols=keepcols,excwords=excwords)
ohe_df=makenull(ukb_tp0,ohe_df)

'''
Merge the 3 sets of data together
'''

ukb_moddata=pd.merge(ohe_df,ukb_tp0_mapped,on='eid')
ukb_moddata=pd.merge(ukb_moddata,df_cts,on='eid')
ukb_moddata=pd.merge(ukb_moddata,df_specvars1,on='eid')
ukb_moddata['eid']=ukb_moddata['eid'].astype(int).astype(str)

ukb_moddata.to_parquet('%s%s' % (path,'ukb_moddata.parquet'))


