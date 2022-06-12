import streamlit as st
import os
import numpy as np

import sys

k=(os.listdir('Documents/Github/ukb-dementia-shap'))

sys.path.append('Documents/Github/ukb-dementia-shap')





from logic.data_processing.data_setup import *
from logic.analysis.analysis import AnalysisCharts
from pandas.tseries.offsets import DateOffset

ds=data_setup()
an=AnalysisCharts()
dp=data_proc_main()


df_dis_date_test2=pd.read_parquet(dp.path+'df_dis_date_test2.parquet')
df_model=pd.read_parquet(dp.path+'df_all_final2022-04-13.parquet')
#df_poly=pd.read_parquet(dp.path+df_polyn_ukb.parquet')

def prep_data(eids_in,eids_inc_depvar,eids_exc_depvar,depvar):
    mask=(df_model['eid'].isin(eids_in))&~(df_model['eid'].isin(eids_exc_depvar))
    df_out=df_model.loc[mask,]

    df_out[depvar]=0
    df_out.loc[df_out['eid'].isin(eids_inc_depvar),depvar]=1
    
    return df_out


# Set the app title


st.title("IDEARs Disease Prediction app")
st.write("A machine learning application to identify the causes behind a disease")
st.write("Copyright @ Michael Allwright 2022")

icd_10_choice = st.selectbox(
'Would you like to model a string or a set of ICD10s?',
("String","ICD10s"))

if icd_10_choice=="ICD10s":
    st.write("Enter ICD10s with no .s and with a '|' between each considered: ")
    icd10s_used = st.text_input(label='ICD10s required')
    dis_mod=st.text_input(label='enter disease label')

else:
    option = st.selectbox('Which disease would you like to model?',("Parkinson's", "Alzheimer's", 'Diabetes','Depression'))
    dict_map=dict({"Parkinson's":"parkinson", "Alzheimer's":"alzheimer", 'Diabetes':"diabetes",'Depression':'depression'})
    dis_mod=dict_map[option]
    st.write('model for',dis_mod)

cols = df_model.columns
dropcols = st.multiselect("Variables to exclude:", cols)
df_model.drop(columns=dropcols,inplace=True)

    
st.write("data processing...")

if icd_10_choice=="ICD10s":
    dis_dict=ds.return_eids(df_dis_date_test2,string=icd10s_used,string_exc='family|screening|insipidus|pregnancy',icd10s=True) 

else:
    dis_dict=ds.return_eids(df_dis_date_test2,string=dis_mod,string_exc='family|screening|insipidus|pregnancy') 


st.write("data prep...")



df=prep_data(eids_in=list(df_model['eid']),eids_inc_depvar=dis_dict['eids_inc_pro'],eids_exc_depvar=dis_dict['eids_exc_pro'],
    depvar=dis_mod)

rat=df[dis_mod].sum()/500
sampsize=int(min(len(df),round(len(df)/rat,0)))

st.write(str(rat)) 

st.dataframe(df1[['eid',dis_mod]].head(20))

st.write(str(round(len(df)/rat,0))) 


df1=df.sample(sampsize)

st.write(str(df1.shape[0])) 



df1=df.copy()

st.write("total case: ",df1[dis_mod].sum())
st.write("total control: ",df1[dis_mod].count()-df1[dict_map[option]].sum())

st.dataframe(df1.head())

ds.shapruns(df=df1,run='trial',remwords='xxxx',depvar=dis_mod,resize=1,resizeratio=5,perc=True,stream=True)

icd10_diabs_type1=ds.search_icd(strings=dis_mod,non_strings='family|screening|noninsulin|insipidus|pregnancy',string_pat=True)[0]



st.write('ICD10s are:', icd10_diabs_type1)
