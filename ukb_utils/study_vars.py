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

'''

Individuals components Frailty (adapted from Halton et al. 3 and used in this manuscript) Weight Loss Self-reported: 
“Compared with one year ago, has your weight changed?” Options: Yes: weight loss in the previous year. No: another option.
 Exhaustion Self-reported: “Over the past two weeks, how often have you felt tired or had little energy” 
 Options: Yes: more than half time or every day No: another option Low physical activity 
 Quintiles of sex- age-specific levels of total PA in derived from IPAQ Options: Yes: Lowest level of PA No: low/middle
  to highest levels of PA Slow walking speed Self-reported: How do you describe your usual walking pace? (a proxy for gait speed)
   Options: Yes: slow No: average or brisk pace Low grip strength Measured grip strength expressed in kg by sex-and BMI 
   adjusted cut-off points. Cut-off points: Men If BMI ≤24 & grip strength ≤ 29 If BMI 24·1 - 26 & grip strength ≤ 30 If
    BMI 26·1 - 28 & grip strength ≤ 30 If BMI >28 & grip strength ≤ 32 45 Women If BMI ≤23 & grip strength ≤ 17 If
     BMI 23·1 - 26 & grip strength ≤ 17·3 If BMI 26·1 - 29 & grip strength ≤ 18 If BMI >29 & grip strength ≤ 21


Unintentional weight loss of >10 lbs (≥4.5 kg) or ≥5% of body mass in the last year (obtained from patient, caregiver, or medical records);

Weakness (assessment based on the handgrip strength measurement; interpretation of results takes into account sex and body mass index [BMI]). A Kern digital dynamometer was used for grip strength measurement;

Exhaustion (audited information based on two questions from Center for Epidemiological Studies Depression (CES-D) scale;33 a score from 1 [fatigue or exhaustion felt rarely or not at all] to 4 [fatigue or exhaustion felt most of the time], 3 or 4 points means that the test is positive for decreased physical activity);

Slow gait (walking time over a distance of 15 ft [4.57 m]; interpretation of results takes into account sex and height);

Low physical activity (energy expenditure weekly rate calculated on the basis of the modified questionnaire Minnesota Leisure Time Activity Questionnaire).34,35

Patients who fulfilled none of the criteria were considered nonfrail, patients who fulfilled 1 and 2 criteria were classified as prefrail, and patients who fulfilled ≥3 criteria were classified as frail.

'''

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


ukb_tp0=pd.read_pickle(path+'ukb_tp0_.p')
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


parkcols_ukb=['daytime_dozing_sleeping_narcolepsy_f1220','coffee_intake_f1498',
         'current_tobacco_smoking_f1239','past_tobacco_smoking_f1249',
         'age_when_periods_started_menarche_f2714','gastric_ulcer']

parkcols_icd=['Unspecified injury of face and head','alcohol_intake_frequency_f1558',
             'qualif_score']


newcols=['breastfed_as_a_baby_f1677_0_0','date_e66_first_reported_obesity_f130792_0_0',
        'worked_with_pesticides_f22614_0_0']