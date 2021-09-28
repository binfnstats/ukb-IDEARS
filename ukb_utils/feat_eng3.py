import pandas as pd
import numpy as np
import os
import re
import warnings
import datetime as dt
warnings.filterwarnings('ignore')
path="../../../ukb/data/"


def findcols(df,string):
    return [col for col in df.columns if string in col]


type1diabetes=['Type 1 diabetes mellitus with ketoacidosis',
 'Type 1 diabetes mellitus with neurological complications',
 'Type 1 diabetes mellitus with ophthalmic complications',
 'Type 1 diabetes mellitus without complications']

cholestrol=['cholesterol_f30690',
 'hdl_cholesterol_f30760']

type2diabetes=['Type 2 diabetes mellitus with circulatory complications',
 'Type 2 diabetes mellitus with ketoacidosis',
 'Type 2 diabetes mellitus with kidney complications',
 'Type 2 diabetes mellitus with neurological complications',
 'Type 2 diabetes mellitus with ophthalmic complications',
 'Type 2 diabetes mellitus with other specified complications',
 'Type 2 diabetes mellitus without complications']

hyperlip=['Mixed hyperlipidemia']

pollcols=['nitrogen_dioxide_air_pollution_2010_f24003',
 'nitrogen_oxides_air_pollution_2010_f24004',
 'particulate_matter_air_pollution_pm10_2010_f24005',
 'particulate_matter_air_pollution_pm25_2010_f24006',
 'particulate_matter_air_pollution_pm25_absorbance_2010_f24007',
 'particulate_matter_air_pollution_2510um_2010_f24008',
 'nitrogen_dioxide_air_pollution_2005_f24016',
 'nitrogen_dioxide_air_pollution_2006_f24017',
 'nitrogen_dioxide_air_pollution_2007_f24018',
 'particulate_matter_air_pollution_pm10_2007_f24019']

sedcols=['time_spent_using_computer_f1080',
'time_spent_driving_f1090',
'time_spent_watching_television_tv_f1070',
'sleep_duration_f1160']

vegfruitcols=['fresh_fruit_intake_f1309','salad_raw_vegetable_intake_f1299']

keepcols=['eid','Dementia','total_pollution','mean_time_to_correctly_identify_matches_f20023',
          'Diabetes_type1','Diabetes_type2','cholestrol','Mixed hyperlipidemia']

cogcols=['mean_time_to_correctly_identify_matches_f20023']

parkcols_ukb=['daytime_dozing_sleeping_narcolepsy_f1220','coffee_intake_f1498',
         'current_tobacco_smoking_f1239','past_tobacco_smoking_f1249',
         'age_when_periods_started_menarche_f2714','gastric_ulcer']



#coded_data
allcols=['eid','Dementia','weight_loss','low_energy','low_activity','weak_right_handgrip',\
'slow_gait','frailty_index','frailty_level','processed_meat_score','parental_ad_status1','depr_selfr',
         'neuroticism_score_f20127',
         
         'Diabetes_type1','Diabetes_type2',
         'total_pollution','cholestrol','sleep','fresh_fruit_intake_f1309','salad_raw_vegetable_intake_f1299',
        'sed_time','qualif_score','age_when_attended_assessment_centre_f21003','gender',
        'glycated_haemoglobin_hba1c_f30750','systolic_blood_pressure_automated_reading_f4080',
         'diastolic_blood_pressure_automated_reading_f4079',
       'Essential (primary) hypertension',
 'APOE4_Carriers',
        'Genotype_e1/e2','Genotype_e1/e4', 'Genotype_e2/e2',\
'Genotype_e2/e3','Genotype_e2/e4','Genotype_e3/e3','Genotype_e3/e4','Genotype_e4/e4']

parkcols_icd=['Unspecified injury of face and head','alcohol_intake_frequency_f1558',
             'qualif_score']


def recodevars(df):
    df['Diabetes_type1']=df[type1diabetes].sum(axis=1)
    df['Diabetes_type2']=df[type2diabetes].sum(axis=1)

    df['total_pollution']=df[pollcols].sum(axis=1)
    df['cholestrol']=df['cholesterol_f30690']
    df['sleep']=df['sleep_duration_f1160'].apply(lambda x:0 if x<7 else (1 if x<9 else 2))
    df['sed_time']=df[sedcols].sum(axis=1)
    
    df_model_out=pd.merge(df,ukb_tp0newcols,on='eid',how='left')
    df_model_out=pd.merge(df_model_out,smokers,on='eid',how='left')
    df_model_out=pd.merge(df_model_out,ethnics,on='eid',how='left')
    
    df_model_out['APOE4_Carriers']=df_model_out[['Genotype_e1/e2','Genotype_e1/e4', 'Genotype_e2/e2',\
'Genotype_e2/e3','Genotype_e2/e4','Genotype_e3/e3','Genotype_e3/e4','Genotype_e4/e4']].\
apply(lambda x:1 if x['Genotype_e3/e4']==1 or  x['Genotype_e2/e4']==1 or  x['Genotype_e1/e4']==1\
else (2 if x['Genotype_e4/e4']==1 else (0 if x['Genotype_e3/e3']==1\
or x['Genotype_e2/e3']==1 or x['Genotype_e1/e2']==1 or x['Genotype_e2/e2']==1 else np.nan)),axis=1)
    
    return df_model_out



ukb_tp0=pd.read_parquet(path+'ukb_tp0_new.parquet')

dem_moddata2years=pd.read_parquet('%s%s' % (path,'dem_moddata2years.parquet'))
AD_moddata2years=pd.read_parquet('%s%s' % (path,'AD_moddata2years.parquet'))
Othdem_moddata2years=pd.read_parquet('%s%s' % (path,'Othdem_moddata2years.parquet'))
PD_moddata2years=pd.read_parquet('%s%s' % (path,'PD_moddata2years.parquet'))


'''
Operations on UKB baseline dataset for specific study variables
'''

ukb_tp0['eid']=ukb_tp0['eid'].astype(int).astype(str)

ethnics=pd.get_dummies(ukb_tp0['ethnic_background_f21000'])
ethnics.columns=['ethn_'+col for col in ethnics.columns]
ethnics['eid']=ukb_tp0['eid']
smokers=pd.get_dummies(ukb_tp0['smoking_status_f20116'])
smokers.columns=['smok_'+col for col in smokers.columns]
smokers['eid']=ukb_tp0['eid']

meatmap=dict({'Never':0,'Less than once a week':1,'Once a week':2,'2-4 times a week':3,'5-6 times a week':4,
     'Once or more daily':5})
ukb_tp0['processed_meat_score']=ukb_tp0['processed_meat_intake_f1349'].map(meatmap)
ukb_tp0['parental_ad_status1']=ukb_tp0['parental_ad_status'].apply(lambda x:0 if x==0 else 1)
ukb_tp0['depr_selfr']=ukb_tp0['recent_feelings_of_depression_f20510'].apply(lambda x:0 if x=="Not at all" else 1)


colsfrail=['weight_change_compared_with_1_year_ago_f2306','recent_feelings_of_tiredness_or_low_energy_f20519',
'ipaq_activity_group_f22032','usual_walking_pace_f924','hand_grip_strength_left_f46','hand_grip_strength_right_f47']


ukb_tp0['weight_loss']=ukb_tp0['weight_change_compared_with_1_year_ago_f2306'].\
apply(lambda x:1 if x=='Yes - lost weight' else 0)
ukb_tp0['low_energy']=ukb_tp0[colsfrail]['recent_feelings_of_tiredness_or_low_energy_f20519'].apply(lambda x:\
1 if x=='Nearly every day' or x=='More than half the days' else 0)
ukb_tp0['low_activity']=ukb_tp0[colsfrail]['ipaq_activity_group_f22032'].apply(lambda x:1 if x=='low' else 0)

ukb_tp0['weak_left_handgrip']=ukb_tp0[colsfrail]['hand_grip_strength_left_f46'].apply(lambda x:0.5 if x<20 else 0 )
ukb_tp0['weak_right_handgrip']=ukb_tp0[colsfrail]['hand_grip_strength_right_f47'].apply(lambda x:0.5 if x<20 else 0 )

ukb_tp0['slow_gait']=ukb_tp0[colsfrail]['usual_walking_pace_f924'].apply(lambda x:1 if x=='Slow pace' else 0)

ukb_tp0['frailty_index']=\
ukb_tp0[['weight_loss','low_energy','low_activity','weak_left_handgrip','weak_right_handgrip','slow_gait']]\
.sum(axis=1)
ukb_tp0['frailty_level']=ukb_tp0['frailty_index'].apply(lambda x:0 if x<1 else
                                                         (2 if x>=3 else 1))

ukb_tp0newcols=ukb_tp0[['eid','neuroticism_score_f20127','weight_loss','low_energy',\
'low_activity','weak_right_handgrip',\
'slow_gait','frailty_index','frailty_level','processed_meat_score','parental_ad_status1',\
'depr_selfr']]


'''
Recoding of variables
'''

dem_model_allcols['gastric_ulcer']=dem_model_allcols[findcols(dem_model_allcols,'gastric')].sum(axis=1)


modsout=recodevars(dem_moddata2years)
modsoutAD=recodevars(AD_moddata2years)
modsoutPD=recodevars(PD_moddata2years)
modsoutOthdem=recodevars(Othdem_moddata2years)

'''
Full variable outputs
'''

modsout.to_parquet('%s%s' % (path,'dem_model_allcols.parquet'))
modsoutAD.to_parquet('%s%s' % (path,'AD2years.parquet'))
odsoutOthdem.to_parquet('%s%s' % (path,'Othdem_2years.parquet'))
modsoutPD.to_parquet('%s%s' % (path,'PD_model_allcols2years.parquet'))

'''
Outputs for files specific to studies for comparisons
'''
modsout[allcols].to_parquet('%s%s' % (path,'dem_model_selcols.parquet'))
modsout[allcols+['daystoDementia']].to_parquet('%s%s' % (path,'dem_model_sa_selcols.parquet'))