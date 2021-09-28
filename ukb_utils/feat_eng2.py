import pandas as pd
import numpy as np
import re
import warnings
import datetime as dt
import icd10
warnings.filterwarnings('ignore')
path="../../../ukb/data/"

'''
Diseases to map
'''

AD_only="F00|F000|F001|F002|F009|G30|G300|G301|G308|G309"
nonAD_dem="F03|F028|G310|F015"
dem2="F00|F000|F001|F002|F009|G30|G300|G301|G308|G309|F03|F028|G310|F015"
dem="F00|F000|F001|F002|F015|F009|G30|G300|G301|G308|G309|F03|F028|G310"
PD="G20|G211|G218|G213|G212|G219|G214"

year=2021
month=1
day=1

def icd10_ohe(df,disease='Dementia',wordsrem='DEMENTIA',dis_icd="F00|F000|F001|F002|F009|G30|G300|G301|G308|G309",disnum=200):
    
    df2=df.copy()
    
    mask_dis=(df2['disease'].str.contains(dis_icd))
    df2[disease]=0
    df2[disease][mask_dis]=1
    df2[disease+"_date"]=np.nan
    df2[disease+"_date"][mask_dis]=df2['dis_date']
    df2[disease+"_date"]=pd.to_datetime(df2[disease+"_date"])
    df_out=pd.DataFrame(df2.groupby('eid').agg({disease:'max',disease+"_date":'max'})).reset_index()
    df_out.columns=['eid',disease,disease+"_date"]
    df_out=df_out[['eid',disease,disease+"_date"]]
    
    mask_exc=(~df['disease_name'].str.upper().str.contains(wordsrem))&(~df['disease'].str.contains(dis_icd))
    df=df[mask_exc]
    df_out_full=pd.merge(df,df_out,on='eid',how='left')
    
    
    return df_out_full,df_out

def findcols(df,string):
    return [col for col in df.columns if string in col]

def removecols(df):

    unknowncols=findcols(df,'Unknown')
    agecols=[col for col in findcols(df,'age_when') if 'f21003' not in col]
    sexcols=[col for col in findcols(df,'sex') if 'Male' not in col and 'Female' not in col]
    chromcols=findcols(df,'chromosome_')
    deviceidcols=findcols(df,'_device_id')

    df['AST_ALT_ratio']=round((df['aspartate_aminotransferase_f30650']\
    /df['alanine_aminotransferase_f30620']),1)

    mask_sleep_mr=(df['sleep_duration_f1160'].between(6,8))
    mask_sleep_high=(df['sleep_duration_f1160']>8)

    df['sleep']=0
    df['sleep'][mask_sleep_mr]=1
    df['sleep'][mask_sleep_high]=2

    #pollution
    df['polluted']=0
    df['polluted'][(df['particulate_matter_air_pollution_pm25_absorbance_2010_f24007']>9)]=1
    cols_sel=[col for col in df.columns if col not in unknowncols+agecols+sexcols+chromcols+deviceidcols]
    df_out=df[cols_sel]
    return df_out


def moddatacreate(df1,df2,df3,df4,yearsout=2,endyears=10,disease='Dementia',file='df_model_assess6.parquet',num_pat=200):
    mask_ass=(df1['dis_date']<=df1['date_of_attending_assessment_centre_f53_0_0'])
    mask_minpats=(df1['num_patients']>num_pat)

    df_model_assess=pd.DataFrame(df1[mask_minpats&mask_ass].groupby(['eid','disease_name']).size().
                          unstack('disease_name')).reset_index()
    
    totaldis=pd.DataFrame(df1[mask_minpats&mask_ass].groupby('eid')['disease'].count()).reset_index()
    totaldis.columns=['eid','total_dis']
    
    disblock=pd.DataFrame(df1[mask_minpats&mask_ass].groupby(['eid','disease_block']).size().unstack('disease_block')).reset_index()
    disblock.fillna(0,inplace=True)
    
    
    df_model_assess2=pd.merge(df2,df_model_assess,on='eid',how='left')
    df_model_assess2=pd.merge(df_model_assess2,totaldis,on='eid',how='left')
    df_model_assess2=pd.merge(df_model_assess2,disblock,on='eid',how='left')

    df_model_assess3=pd.merge(df3[['eid']],df_model_assess2,on='eid',how='left')
    df_model_assess3.fillna(0,inplace=True)
    df_model_assess4=pd.merge(df3,df_model_assess3,on='eid',how='left')
    df_model_assess5=pd.merge(df_model_assess4,df4[['eid','date_of_attending_assessment_centre_f53',
     'date_of_death_f40000']],on='eid',how='left')
    
    df_model_assess5['daysto'+disease]=(pd.to_datetime(df_model_assess5[disease+'_date'],errors='coerce')\
-pd.to_datetime(df_model_assess5['date_of_attending_assessment_centre_f53'])).dt.days


    mask_death=pd.notnull(df_model_assess5['date_of_death_f40000'])
    mask_control=(df_model_assess5[disease]==0)
    mask_dem=(df_model_assess5[disease]==1)
    mask_years=(df_model_assess5['daysto'+disease]>365.25*yearsout)&\
(df_model_assess5['daysto'+disease]<365.25*endyears)
    cases=df_model_assess5[mask_years&mask_dem]
    controls=df_model_assess5[~mask_death&mask_control]

    df_model_assess6=pd.concat([cases,controls],axis=0)
    
    df_model_assess6.to_parquet('%s%s' % (path,file))
    return df_model_assess6

def returndesc(string):
    '''
    functions to apply the icd10 mapping and return disease and disease block
    '''
    code=icd10.find(str(string))
    if code:
        desc=code.description
    else:
        desc=string
    return desc

def returndescblock(string):
    
    try:
        code=icd10.find(str(string))
        desc_block=str(code.block_description)
        
        return desc_block
    except:
        pass

inpatient_update=pd.read_parquet('%s%s' % (path,'inpatient_update_Jul21.parquet'))
ukb_moddata=pd.read_parquet('%s%s' % (path,'ukb_moddata.parquet'))
ukb_tp0=pd.read_parquet('%s%s' % (path,'ukb_tp0_new.parquet'))


inpatient_update['Age_Today']=inpatient_update['age_when_attended_assessment_centre_f21003_0_0']+\
(dt.datetime(year, month, day)-pd.to_datetime(inpatient_update['date_of_attending_assessment_centre_f53_0_0']))\
.dt.days/365.25
inpatient_update['date_of_attending_assessment_centre_f53_0_0']=\
pd.to_datetime(inpatient_update['date_of_attending_assessment_centre_f53_0_0'])
inpatient_update['eid']=inpatient_update['eid'].astype(str)


'''
death eids find these so they can be excluded
'''
mask=pd.notnull(inpatient_update['date_of_death_f40000_0_0'])
death_eids=inpatient_update['eid'][mask].astype(str)


'''
Creation of df_dis_date
'''

cols1=[col for col in inpatient_update.columns if '41270' in col or 'eid' in col]
cols2=[col for col in inpatient_update.columns if '41280' in col or 'eid' in col]

df_dis=inpatient_update[cols1]
df_date=inpatient_update[cols2]

df_dis = pd.melt(df_dis, id_vars='eid', value_name='VALUE')
df_dis=df_dis[pd.notnull(df_dis['VALUE'])]

df_dis.columns=['eid','variable','disease']
df_dis['disease']=df_dis['disease'].str.replace('b','')

'''
align variables for merge
'''
df_dis['disease']=df_dis['disease'].str.replace("'","")
df_dis['variable']=df_dis['variable'].str.replace('diagnoses_icd10_','')

df_date = pd.melt(df_date, id_vars='eid', value_name='VALUE')
df_date=df_date[pd.notnull(df_date['VALUE'])]
df_date['variable']=df_date['variable'].str.replace('41280','41270')
df_date['variable']=df_date['variable'].str.replace('date_of_first_inpatient_diagnosis_icd10_','')

df_date.columns=['eid','variable','dis_date']

'''
align variables for merge
'''
df_date['dis_date']=df_date['dis_date'].str.replace('b','')
df_date['dis_date']=df_date['dis_date'].str.replace("'","")
df_date['dis_date']=pd.to_datetime(df_date['dis_date'])

df_dis_date=pd.merge(df_dis,df_date,on=['eid','variable'],how='left')

df_dis_date=pd.merge(df_dis_date,inpatient_update[['eid','Age_Today','date_of_attending_assessment_centre_f53_0_0']])
df_dis_date['Age_disease']=df_dis_date['Age_Today']-\
(dt.datetime(year, month, day)-pd.to_datetime(df_dis_date['dis_date']))\
.dt.days/365.25

df_dis_date['disease_name']=df_dis_date['disease'].apply(returndesc)
df_dis_date['disease_block']=df_dis_date['disease'].apply(returndescblock)


'''
Apply functions to return specific disease ICD10 mappings
'''



df_dis_dem,df_dem=icd10_ohe(df_dis_date,disease='Dementia',
                 dis_icd=dem,disnum=200)
df_dis_AD,df_AD=icd10_ohe(df_dis_date,disease='AD',
                 dis_icd=AD_only,disnum=200)
df_dis_othdem,df_dem_othdem=icd10_ohe(df_dis_date,disease='Oth_Dem',
                 dis_icd=nonAD_dem,disnum=200)
df_dis_PD,df_PD=icd10_ohe(df_dis_date,disease='PD',
                 dis_icd=PD,disnum=200)

dem_moddata2years=moddatacreate(df1=df_dis_dem,df2=df_dem,df3=ukb_moddata,df4=ukb_tp0,yearsout=2,file='dem_moddata2years.parquet',disease='Dementia')
PD_moddata2years=moddatacreate(df1=df_dis_PD,df2=df_PD,df3=ukb_moddata,df4=ukb_tp0,yearsout=2,file='PD_moddata2years.parquet',disease='PD')
