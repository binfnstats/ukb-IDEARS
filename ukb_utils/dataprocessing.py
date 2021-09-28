import pandas as pd
import numpy as np
import re
import os
import warnings
import datetime as dt
warnings.filterwarnings('ignore')
import snappy
import fastparquet
path="../../../ukb/data/"
file="ukb_all_fields_july2021.csv"

file2="all_fields_aug_2021.csv"
    
    
def findcols(df,string):
    return [col for col in df.columns if string in col]

weirdcols='date_of_cancer_diagnosis|type_of_cancer_icd10|age_at_cancer_diagnosis|histology_of_cancer_tumour|\
behaviour_of_cancer_tumour|type_of_cancer_icd9|cancer_record_format|\
cancer_record_origin'


brain_words='brain|corp|mean|mean_md|mean_icvf|volume|eid|mean_time'

'''
Import first 10 rows
'''
check2=pd.read_csv('%s%s' % (path,file2),nrows=10)

'''
conditional sets of columns for analyses
'''

colstp0=[s for s in check2.columns if 
        (re.search(weirdcols,s) and s[len(s)-1:len(s)]=='0') 
         or (not re.search(weirdcols,s) and (s[len(s)-3:len(s)-2]=='0' or s[len(s)-4:len(s)-3]=='0'))]
colstp0=[col for col in check2.columns if col=='eid']+colstp0

braincols=[col for col in check2.columns if ((re.search(brain_words,col)) and '_2_' in col) or 'eid' in col]

ICD10cols=[col for col in check.columns if '41270' in col or '41280' in col or 'eid' in col]

icdextcols=['age_when_attended_assessment_centre_f21003_0_0','date_of_attending_assessment_centre_f53_0_0',
'date_of_death_f40000_0_0']

ICD10cols=ICD10cols+icdextcols



'''
import and export
'''

ukb_tp0=pd.read_csv('%s%s' % (path,file2),usecols=colstp0)
ukb_tp0.to_parquet('%s%s' % (path,'ukb_tp0_new.parquet'))

df_brain=pd.read_csv('%s%s' % (path,file),usecols=braincols)
df_brain.to_parquet('%s%s' % (path,'df_brain.parquet'))

inpatient_update=pd.read_csv('%s%s' % (path,file),usecols=ICD10cols)
inpatient_update.to_parquet('%s%s' % (path,'inpatient_update_Jul21.parquet'))

'''
mappings
'''
