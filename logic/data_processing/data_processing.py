"""
Created on Mon Sep 27 16:39:52 2021

@author: michaelallwright

This file contains a set of functions that are to be used in the new version of UKB
to do the key analyses

"""

#key information to run financial passport for - user and start/end date. Move to config file?

from typing import Any, List
import pandas as pd
import numpy as np
import re
import icd10
import datetime as dt
import ast
from sklearn.preprocessing import StandardScaler
from pandas.tseries.offsets import DateOffset

class data_proc_main(object):
	"""
	Module for feature engineering
	"""

	def __init__(self):
		"""
		Initilising models.
		"""

		self.year=2021 #latest data's year, month, day
		self.month=1
		self.day=31

		self.years_wait=2 #years to ignore diagnoses post baseline
		self.years_max=10 #max years to model from
		self.min_part_dis=200 #min number of patients with disease x before included

		self.path="/Users/michaelallwright/Dropbox (Sydney Uni)/michael_PhD/Projects/UKB/Data/"
		self.path2="/Users/michaelallwright/Documents/GitHub/ukb-dementia-shap/static/"
		self.path_pain="/Users/michaelallwright/Documents/GitHub/UKB/Pain/data/"
		self.fullfile='ukb_tp0_new.parquet'
		self.cols_nonmiss_file='cols80.csv'
		self.icd10_file='ukb_ICD10.parquet'

		#special diseases to model which are combs of ICD10s
		self.dis_map={'cerebrovasc':['I60','I62','I63','I65','I66','I67','I68','I69'],
		 'stroke':['I64'],'TBI':['S0'],'Hear_loss':['H919','H901','H902','H903','H904',\
	'H905','H906','H907','H908','H909','H910','H911','H912','Z974']}

		#disease words for labels
		self.diseases_words_monitor='dementia|alzheim|parkin|hunting|diabetes'

		#mappings

		self.genos={'Genotype_e1/e2':0,
				'Genotype_e1/e4':1,
				'Genotype_e2/e2':0,
				'Genotype_e2/e3':0,
				'Genotype_e2/e4':1,
				'Genotype_e3/e3':0,
				'Genotype_e3/e4':1,
				'Genotype_e4/e4':2}

		self.qualif={'qualifications_f6138_0_0_A levels/AS levels or equivalent':3,
					'qualifications_f6138_0_0_CSEs or equivalent':2,
					'qualifications_f6138_0_0_College or University degree':4,
					'qualifications_f6138_0_0_NVQ or HND or HNC or equivalent':1,
					'qualifications_f6138_0_0_O levels/GCSEs or equivalent':0,
					'qualifications_f6138_0_0_Other professional qualifications eg: nursing, teaching':3}

		self.alc_map={'alcohol_intake_frequency_f1558_0_0_Daily or almost daily':4,
					'alcohol_intake_frequency_f1558_0_0_Never':0,
					'alcohol_intake_frequency_f1558_0_0_Once or twice a week':2,
					'alcohol_intake_frequency_f1558_0_0_One to three times a month':1,
					'alcohol_intake_frequency_f1558_0_0_Three or four times a week':3}

		self.pest_map={ 'Often':2, 'Rarely/never':0, 'Sometimes':1}


		self.urb_rur={'England/Wales - Urban - less sparse': 1,
		'England/Wales - Town and Fringe - less sparse': 0,
		'Scotland - Large Urban Area': 1,
		'England/Wales - Village - less sparse': 0,
		'England/Wales - Hamlet and Isolated Dwelling - less sparse': 0,
		'Scotland - Other Urban Area': 1,
		'Scotland - Accessible Rural': 0,
		'Scotland - Accessible Small Town': 0,
		'England/Wales - Village - sparse': 0,
		'Scotland - Remote Rural': 0,
		'Scotland - Remote Small Town': 0,
		'England/Wales - Town and Fringe - sparse': 1,
		'England/Wales - Hamlet and Isolated dwelling - sparse': 0,
		'Scotland - Very Remote Rural': 0,
		'England/Wales - Urban - sparse': 1}

		self.studycols_PD=['eid','PD','sex_f31_0_0','calc','neuroticism_score_f20127_0_0','non_ost', 'depressed','beta_block',\
	'melanoma','hypertension','ipaq_activity_group_f22032_0_0','age_when_attended_assessment_centre_f21003_0_0',\
	'never_eat_eggs_dairy_wheat_sugar_f6144_0_0_Dairy products', 'urate_f30880_0_0', 'non_ost_non_asp','Constipation',\
	'ibuprofen', 'coffee_intake_f1498_0_0','alcohol','pesticide_exposure','urban_rural','TBI','smoking_status_f20116_0_0',\
	'parental_pd']

		self.cols_lancet=list(self.genos)+['dementia','eid','age_when_attended_assessment_centre_f21003_0_0','APOE4_Carriers','TBI',
		'alcohol','pollution','hypertension','diabetes','Hear_loss','ever_smoked_f20160_0_0','body_mass_index_bmi_f21001_0_0',
		'depressed','smoking_status_f20116_0_0','ipaq_activity_group_f22032_0_0','Qualif_Score',
		'frequency_of_friendfamily_visits_f1031_0_0']


		self.studycols_dem=list(self.genos)+['dementia','eid','age_when_attended_assessment_centre_f21003_0_0','APOE4_Carriers',
		'pollution','sedentary_time','diabetes','low_activity','salad_raw_vegetable_intake_f1299_0_0',
		'fresh_fruit_intake_f1309_0_0','weight_change_compared_with_1_year_ago_f2306_0_0',
		'frequency_of_tiredness_lethargy_in_last_2_weeks_f2080_0_0','ipaq_activity_group_f22032_0_0','usual_walking_pace_f924_0_0',
		'hand_grip_strength_left_f46_0_0','hand_grip_strength_right_f47_0_0','body_mass_index_bmi_f21001_0_0',
		'systolic_blood_pressure_automated_reading_f4080_0_0','diastolic_blood_pressure_automated_reading_f4079_0_0',
		'frailty_score','smoking_status_f20116_0_0','cholesterol_f30690_0_0','hdl_cholesterol_f30760_0_0',
		'processed_meat_intake_f1349_0_0','mean_time_to_correctly_identify_matches_f20023_0_0',
		'number_of_incorrect_matches_in_round_f399_0_2','sex_f31_0_0','hypertension','ever_smoked_f20160_0_0','alcohol','TBI',
		'Hear_loss','Qualif_Score']



		#columns to include for merges - key columns for analyses selections
		self.keycols=['eid','date_of_attending_assessment_centre_f53_0_0','age_when_attended_assessment_centre_f21003_0_0']

		#words to convert to nulls
		self.excwords=['Prefer not to answer','nan','None of the above']

		#words to exclude columns with these included
		self.excs='source_of_report|first_reported|icd10|icd9|operative_procedures|treatment_speciality|\
	external_ca|patient_recoded|hospital_polymorphic|_report|assay_date|device_id'

		#extra words to exclude in cts vars
		self.excs_cts='aliquot|assessment_centre|acquisition_time|main_speciality|date_of_|patient_classi|\
	methods_of_discharge|inpatient_record_format|weight_method|_missing_reason|eid'

		# these variables override the ohe requirements
		self.ohe_exceps=['current_employment_status_f6142_0_0','qualifications_f6138_0_0']

		"""
		#special columns to modify
		self.speccols=['sex_f31','average_total_household_income_before_tax_f738','usual_walking_pace_f924',
		'frequency_of_friendfamily_visits_f1031','drive_faster_than_motorway_speed_limit_f1100',
		'weekly_usage_of_mobile_phone_in_last_3_months_f1120','qualifications_f6138',
		'gender','avgincome','walkspeed','freqfriendfamily','faster_mot_speed','weekly_mobphone_mins',
		'qualif_score','APOE4_Carriers']
		"""

	def convert_null(self,df):

		"""
		helper function to convert certain values to nulls
		"""
		for col in df.columns:
			mask=(df[col]=='Prefer not to answer')|(df[col]=='nan')
			df[col][mask]=np.NaN
		return df

	def findcols(self,df,string):

		"""
		helper function to find columns based on string 
		"""

		return [col for col in df if string in col]



	def std_scale_newvar(self,df,vars=[],name='inflammation'):
		for var in vars:
			trans = StandardScaler()
			df[var+'std']=trans.fit_transform(np.asarray(df[var]).reshape(-1, 1))
		
		df[name]=df[[v+'std' for v in vars]].sum(axis=1)
		df.drop(columns=[v+'std' for v in vars],inplace=True)
		return df

	def returndesc(self,string):

		'''
		functions to apply the icd10 mapping and return disease
		'''

		code=icd10.find(str(string))
		if code:
			desc=code.description
		else:
			desc=string
		return desc

	def returndescblock(self,string):

		'''
		function to apply the icd10 mapping and return disease block
		'''
		
		try:
			code=icd10.find(str(string))
			desc_block=str(code.block_description)
			
			return desc_block
		except:
			pass

	def ukb_icd10(self):
		df=pd.read_csv('/Users/michaelallwright/Documents/GitHub/UKB/data/icd10list_ukb.csv')
		df['code']=df['ICD10'].apply(lambda x:x[0:x.find(' ')])
		df['disease']=df['ICD10'].apply(lambda x:x[x.find(' ')+1:len(x)])
		df['code']=df['code'].apply(lambda x:x.replace('.',''))
		df['disease']=df['disease'].apply(lambda x:x.lower())
		df['disease']=df['disease'].apply(lambda x:x.replace('-',''))

		df=df[['code','disease']]

		return df

	def ukb_icd10_r(self):
		df=pd.read_csv('/Users/michaelallwright/Documents/GitHub/UKB/data/code_map2.csv')
		df=df.loc[df['Coding']==19,['Value','Meaning']]
		df['Meaning']=df['Meaning'].apply(lambda x:x.lower())
		icd10_lkup_dict=dict(zip(df['Value'],df['Meaning']))

		return icd10_lkup_dict

	def treatment_mapping(self):

		coding4=pd.read_csv(self.path2+'coding4.tsv',sep="\t")
		treat_codes=pd.read_csv(self.path2+'medications_codes.csv')
		df=pd.read_parquet(self.path2+'ukb_treatments_test.parquet')

		df.fillna(0,inplace=True)
		df[[col for col in df.columns]]=df[[col for col in df.columns]].astype(int)
		coding_dic=dict(zip(coding4['coding'],coding4['meaning']))

		treatcols=[col for col in df.columns if 'treat' in col]
		for col in treatcols:
			df[col+'_name']=df[col].map(coding_dic)

		namecols=[col for col in df.columns if 'name' in col]

		df_sum=pd.melt(df[['eid']+namecols],id_vars=['eid'])

		df_sum=df_sum[pd.notnull(df_sum['value'])]

		beta_block=list(treat_codes['treatment/med'][(treat_codes['Beta Blocker']=="yes")])
		non_ost=list(treat_codes['treatment/med'][(treat_codes['nonsteroidal anti-inflammatory drugs']=="yes")])+['ibuprofen']
		non_ost_non_asp=list(treat_codes['treatment/med'][(treat_codes['nonsteroidal anti-inflammatory drugs']=="yes")&\
	(treat_codes['treatment/med']!="aspirin")])+['ibuprofen']

		calc=list(treat_codes['treatment/med'][(treat_codes['calcium channel blockers']=="yes")])
		anti_inf_steroid=list(treat_codes['treatment/med'][(treat_codes['anti_inf_steroid']=="yes")])

		df_sum['beta_block']=0
		df_sum['beta_block'][(df_sum['value'].isin(beta_block))]=1
		df_sum['non_ost']=0
		df_sum['non_ost'][(df_sum['value'].isin(non_ost))]=1
		df_sum['non_ost_non_asp']=0
		df_sum['non_ost_non_asp'][(df_sum['value'].isin(non_ost_non_asp))]=1


		df_sum['calc']=0
		df_sum['calc'][(df_sum['value'].isin(calc))]=1
		df_sum['anti_inf_steroid']=0
		df_sum['anti_inf_steroid'][(df_sum['value'].isin(anti_inf_steroid))]=1

		new_med_vars=list(set(list(df_sum['value'].value_counts().head(30).index)+\
	list(treat_codes['treatment/med'][pd.notnull(treat_codes['treatment/med'])])))

		for var in new_med_vars:
			df_sum[var]=0
			df_sum[var][(df_sum['value']==var)]=1

		used_med_vars=new_med_vars+['beta_block','non_ost','non_ost_non_asp','calc','anti_inf_steroid']

		df_out=pd.DataFrame(df_sum.groupby(['eid'])[used_med_vars].max()).reset_index()

		df_out.to_parquet(self.path+'treatments_test.parquet')

		return df_out



	def specific_disease_label(self,icd10s,disname):
		df_dis_date=pd.read_parquet(self.path+'df_dis_date_test.parquet')
		mask_icd=(df_dis_date['disease'].isin(icd10s))
		
		dis_label1=pd.DataFrame(df_dis_date[mask_icd].\
	groupby('eid').agg({'dis_aft':'max','disease_date':'min'})).reset_index()

		dis_label1[disname]=1

		dis_label1=dis_label1[['eid',disname,'disease_date']]


		dis_label2=pd.DataFrame(df_dis_date[mask_icd].\
	groupby('eid').agg({'dis_bef':'max','disease_date':'min'})).reset_index()

		dis_label2[disname]=-1

		dis_label2=dis_label2[['eid',disname,'disease_date']]

		dis_label=pd.concat([dis_label1,dis_label2],axis=0)


		return dis_label






	def dis_date_file(self,infile='ukb_ICD10.parquet',outfile='dis_ohe_icd10_test.parquet'):

		df=pd.read_parquet('%s%s' % (self.path,infile))

		
		#format dates and work out age today
		
		df['date_of_attending_assessment_centre_f53_0_0']=pd.to_datetime(df['date_of_attending_assessment_centre_f53_0_0'])
		df['Age_Today']=df['age_when_attended_assessment_centre_f21003_0_0']+(dt.datetime(self.year, self.month, self.day)-\
		df['date_of_attending_assessment_centre_f53_0_0']).dt.days/365.25

	
		#ICD10 columns for extraction and split of dates and diseases data


		cols1=[col for col in df.columns if '41270' in col or 'eid' in col]
		cols2=[col for col in df.columns if '41280' in col or 'eid' in col]

		df_dis=df[cols1]
		df_date=df[cols2]


		#make so 1 record per individual per ICD10

		df_dis = pd.melt(df_dis, id_vars='eid', value_name='VALUE')
		df_dis=df_dis[pd.notnull(df_dis['VALUE'])]

		df_dis.columns=['eid','variable','disease']
		#df_dis['disease']=df_dis['disease'].str.replace('b','')

		
		df_dis['disease']=df_dis['disease'].str.replace("'","")
		df_dis['variable']=df_dis['variable'].str.replace('diagnoses_icd10_','')

		df_date = pd.melt(df_date, id_vars='eid', value_name='VALUE')
		df_date=df_date[pd.notnull(df_date['VALUE'])]
		df_date['variable']=df_date['variable'].str.replace('41280','41270')
		df_date['variable']=df_date['variable'].str.replace('date_of_first_inpatient_diagnosis_icd10_','')

		df_date.columns=['eid','variable','dis_date']

		
		df_date['dis_date']=df_date['dis_date'].str.replace('b','')
		df_date['dis_date']=df_date['dis_date'].str.replace("'","")
		df_date['dis_date']=pd.to_datetime(df_date['dis_date'])

		df_dis_date=pd.merge(df_dis,df_date,on=['eid','variable'],how='left')

		df_dis_date=pd.merge(df_dis_date,df[['eid','Age_Today','date_of_attending_assessment_centre_f53_0_0']])
		df_dis_date['Age_disease']=df_dis_date['Age_Today']-\
		(dt.datetime(self.year, self.month, self.day)-pd.to_datetime(df_dis_date['dis_date']))\
		.dt.days/365.25

		df_dis_date['disease_name']=df_dis_date['disease'].apply(self.returndesc)
		df_dis_date['disease_block']=df_dis_date['disease'].apply(self.returndescblock)

		dictmap=self.ukb_icd10_r()
		df_dis_date['disease_name_new']=df_dis_date['disease'].map(dictmap)
		df_dis_date.rename(columns={'date_of_attending_assessment_centre_f53_0_0':'date_assess','dis_date':'disease_date'}\
	,inplace=True)
	

		#create dummie variables for diseases if before assessment centre (dis_bef) for indep vars and if after 2 years/
		#before 10 years (mask_aft) for dep variable
		mask_bef=(df_dis_date['disease_date']<df_dis_date['date_assess'])

		mask_aft=(df_dis_date['disease_date']>=df_dis_date['date_assess']+pd.offsets.DateOffset(years=self.years_wait))&\
		(df_dis_date['disease_date']<=df_dis_date['date_assess']+pd.offsets.DateOffset(years=self.years_max))

		mask_10y=(df_dis_date['disease_date']>df_dis_date['date_assess']+pd.offsets.DateOffset(years=self.years_max))

		df_dis_date['dis_bef']=0
		df_dis_date['dis_bef'][mask_bef]=1

		df_dis_date['dis_aft']=0
		df_dis_date['dis_aft'][mask_aft]=1

		df_dis_date['dis_exc']=0
		df_dis_date['dis_exc'][mask_10y|mask_bef]=1

		#total participants for each ICD10
		df_dis_date['total_bef']=df_dis_date.groupby('disease')['dis_bef'].transform('sum')
		df_dis_date['total_aft']=df_dis_date.groupby('disease')['dis_aft'].transform('sum')

		df_dis_date.to_parquet(self.path+'df_dis_date_test2.parquet')

		#mapping of certain diseases by their codes in dis_map above
		for var in self.dis_map:
			df_dis_date[var]=0
			mask=(df_dis_date['disease'].str.contains('|'.join(self.dis_map[var])))
			df_dis_date[var][mask]=df_dis_date['dis_bef']

		#filter only for minimum participants
		df_dis_date=df_dis_date[(df_dis_date['total_bef']>self.min_part_dis)]

		#use disease map dictionary to create df of specific focus diseases
		cols=[k for k in self.dis_map]
		spec_conds=pd.DataFrame(df_dis_date.groupby(['eid'])[cols].sum()).reset_index()

		#simple step but one hot encodes all diseases based on the dis_bef criteria
		dis_ohe_icd10=pd.DataFrame(df_dis_date.groupby(['eid','disease_name'])['dis_bef'].sum()\
	.unstack('disease_name')).reset_index()

		#compute total number of conditions
		totaldis=pd.DataFrame(df_dis_date.groupby('eid')['dis_bef'].sum()).reset_index()
		totaldis.columns=['eid','total_dis']

		#compute total number of conditions within each disease block
		disblock=pd.DataFrame(df_dis_date.groupby(['eid','disease_block'])['dis_bef'].max().\
	unstack('disease_block')).reset_index()
		disblock.fillna(0,inplace=True)

		#merge all files together
		dis_ohe_icd10=pd.merge(dis_ohe_icd10,spec_conds,on='eid',how='outer')
		dis_ohe_icd10=pd.merge(df['eid'],dis_ohe_icd10,how='left',on='eid')
		dis_ohe_icd10=pd.merge(dis_ohe_icd10,disblock,how='left',on='eid')
		dis_ohe_icd10=pd.merge(dis_ohe_icd10,totaldis,how='left',on='eid')
		dis_ohe_icd10.fillna(0,inplace=True)

		#output files
		#df_dis_date.to_parquet(self.path+'df_dis_date_test.parquet')
		dis_ohe_icd10.to_parquet(self.path+outfile)


		return df_dis_date,dis_ohe_icd10

	def disease_labels_ICD10s(self,icd10s=['G309', 'G308', 'G300', 'G301'],disease='AD',out='test.parquet',strcont=False):

		df_dis_date_test=pd.read_parquet(self.path+'df_dis_date_test2.parquet')
		
		if strcont:
			dis_lab=df_dis_date_test[(df_dis_date_test['disease'].str.contains(icd10s,regex=True))]
		else:
			dis_lab=df_dis_date_test[(df_dis_date_test['disease'].isin(icd10s))]
		print(dis_lab['disease'].value_counts())

		dis_lab=pd.DataFrame(dis_lab.groupby('eid')['disease_date','date_assess'].min()).reset_index()
		mask=(dis_lab['disease_date']>dis_lab['date_assess']+ DateOffset(years=2))
		dis_lab['time_to_'+disease]=(dis_lab['disease_date']-dis_lab['date_assess']).dt.days/365.25
		dis_lab[disease]=-1
		dis_lab[disease][mask]=1

		dis_lab.rename(columns={'disease_date':disease+'_date'},inplace=True)

		dis_lab.to_parquet(self.path+out)

		return dis_lab

	def disease_labels_ICD10s2(self,icd10s=['G309', 'G308', 'G300', 'G301'],disease='AD',out='test.parquet',strcont=False,bef=False,
		years=2,outfile=True):

		df_dis_date_test=pd.read_parquet(self.path+'df_dis_date_test2.parquet')
		
		if strcont:
			dis_lab=df_dis_date_test[(df_dis_date_test['disease'].str.contains(icd10s,regex=True))]
		else:
			dis_lab=df_dis_date_test[(df_dis_date_test['disease'].isin(icd10s))]
		
		dis_lab_full=dis_lab.copy()

		dis_lab=pd.DataFrame(dis_lab.groupby('eid')['disease_date','date_assess'].min()).reset_index()

		if bef:
			mask=(dis_lab['disease_date']<dis_lab['date_assess']+ DateOffset(years=years))
			word='time_since_'
			dis_lab[word+disease]=(dis_lab['date_assess']-dis_lab['disease_date']).dt.days/365.25

			mask2=(dis_lab_full['disease_date']<dis_lab_full['date_assess']+ DateOffset(years=years))
			word='time_since_'
			dis_lab_full[word+disease]=(dis_lab_full['date_assess']-dis_lab_full['disease_date']).dt.days/365.25

		else:
			mask=(dis_lab['disease_date']>dis_lab['date_assess']+ DateOffset(years=years))
			word='time_to_'
			dis_lab[word+disease]=(dis_lab['disease_date']-dis_lab['date_assess']).dt.days/365.25

			mask2=(dis_lab_full['disease_date']>dis_lab_full['date_assess']+ DateOffset(years=years))
			word='time_to_'
			dis_lab_full[word+disease]=(dis_lab_full['disease_date']-dis_lab_full['date_assess']).dt.days/365.25

		dis_lab[disease]=-1
		dis_lab[disease][mask]=1

		dis_lab.rename(columns={'disease_date':disease+'_date'},inplace=True)

		dis_lab['eid']=dis_lab['eid'].astype(str)
		dis_lab_full['eid']=dis_lab_full['eid'].astype(str)

		if outfile:
			dis_lab.to_parquet(self.path+out)

		return dis_lab,dis_lab_full

	def famhistory(self):

		"""
		converts the family history file to a file with columns for PD and dementia
		"""
		df=pd.read_parquet(self.path+'df_fam_hist.parquet')
		df = pd.melt(df, id_vars='eid', value_name='VALUE')
		df['father_parkinson']=0
		df['father_parkinson'][(df['VALUE'].str.contains('Parkins|parkins'))&\
		(df['variable'].str.contains('father'))]=1

		df['mother_parkinson']=0
		df['mother_parkinson'][(df['VALUE'].str.contains('Parkins|parkins'))&\
		(df['variable'].str.contains('mother'))]=1

		df['father_dementia']=0
		df['father_dementia'][(df['VALUE'].str.contains('Dementia|dementia'))&\
		(df['variable'].str.contains('father'))]=1

		df['mother_dementia']=0
		df['mother_dementia'][(df['VALUE'].str.contains('Dementia|dementia'))&\
		(df['variable'].str.contains('mother'))]=1

		df_fam_pddem=pd.DataFrame(df.groupby(['eid'])\
		['father_parkinson','mother_parkinson','father_dementia','mother_dementia'].sum()).reset_index()
		df_fam_pddem['eid']=df_fam_pddem['eid'].astype(str)
		df_fam_pddem['parental_pd']=df_fam_pddem[['father_parkinson','mother_parkinson']].sum(axis=1)
		df_fam_pddem['parental_dem']=df_fam_pddem[['father_dementia','mother_dementia']].sum(axis=1)
		df_fam_pddem.drop(columns=['father_parkinson','mother_parkinson','father_dementia','mother_dementia'],inplace=True)

		df_fam_pddem.to_parquet(self.path+'df_fam_pddem.parquet')

		return df_fam_pddem


	def ukb_diseases(self):

		"""
		this one hot encodes diseases and separates these but based on the UKB file's raw attempt at this as 
		opposed to the ICD10 attempt above. It also outputs our labels datasets
		"""

		df=pd.read_parquet('%s%s' % (self.path,'ukb_diseases_test.parquet'))
		dis_full=pd.DataFrame([])

		for i,col in enumerate(df.columns):
			if 'eid' not in col and 'assessment_centre' not in col:
				df1=df[[col,'eid','date_of_attending_assessment_centre_f53_0_0']][pd.notnull(df[col])]
				df1.columns=['disease_date','eid','date_assess']
				df1['disease_date']=pd.to_datetime(df1['disease_date']).dt.date
				df1['date_assess']=pd.to_datetime(df1['date_assess']).dt.date
				df1['disease']=str(col)
				dis_full=pd.concat([dis_full,df1],axis=0)

		dis_full['disease']=dis_full['disease'].str.replace('date_|_first_|reported_','',regex=True)

		mask_bef=(dis_full['disease_date']<dis_full['date_assess'])

		mask_aft=(dis_full['disease_date']>=dis_full['date_assess']+pd.offsets.DateOffset(years=2))&\
		(dis_full['disease_date']<=dis_full['date_assess']+pd.offsets.DateOffset(years=10))

		mask_10y=(dis_full['disease_date']>dis_full['date_assess']+pd.offsets.DateOffset(years=10))

		dis_full['dis_bef']=0
		dis_full['dis_bef'][mask_bef]=1

		dis_full['dis_aft']=0
		dis_full['dis_aft'][mask_aft]=1

		dis_full['dis_exc']=0
		dis_full['dis_exc'][mask_10y|mask_bef]=1

		dis_full['total_bef']=dis_full.groupby('disease')['dis_bef'].transform('sum')
		dis_full['total_aft']=dis_full.groupby('disease')['dis_aft'].transform('sum')

		#creation of independent variables for disease before assessment centre
		dis_ohe_ukb=pd.DataFrame(dis_full.groupby(['eid','disease'])['dis_bef'].sum().unstack('disease')).reset_index()
		dis_ohe_ukb=pd.merge(df['eid'],dis_ohe_ukb,how='left',on='eid')
		dis_ohe_ukb.fillna(0,inplace=True)
		dis_ohe_ukb['total_dis']=dis_ohe_ukb[[col for col in dis_ohe_ukb.columns if 'eid' not in col]].sum(axis=1)

		#create labels data for key diseases to focus on
		mask_dis=(dis_full['disease'].str.contains(self.diseases_words_monitor,regex=True))

		labels=pd.DataFrame(dis_full[mask_dis].groupby(['eid','disease'])['dis_aft'].sum()\
	.unstack('disease')).reset_index()
		labels.columns=['eid']+[col+'_label' for col in labels.columns if col!='eid']
		labels=pd.merge(df['eid'],labels,how='left',on='eid')
		labels.fillna(0,inplace=True)

		label_dates=pd.DataFrame(dis_full[mask_dis].groupby(['eid','disease'])['disease_date'].\
	min().unstack('disease')).reset_index()
		label_dates.columns=['eid']+[col+'_date' for col in label_dates.columns if col!='eid']

		for col in label_dates.columns:
			if col!='eid':
				label_dates[col]=pd.to_datetime(label_dates[col])
		label_dates=pd.merge(df['eid'],label_dates,how='left',on='eid')  

		for col in label_dates.columns:
			if col!='eid':
				label_dates[col][pd.isnull(label_dates[col])]='2030-01-01'
				label_dates[col]=pd.to_datetime(label_dates[col])

		label_dates['dementia_date']=label_dates[[col for col in label_dates.columns if 'dementia' in col]].min(axis=1)
		label_dates['dementia_date'][(label_dates['dementia_date']=='2030-01-01')]=np.nan    
		label_dates['parkins_date']=label_dates[[col for col in label_dates.columns if 'g20parkinsons_disease' in col]].min(axis=1)
		label_dates['parkins_date'][(label_dates['parkins_date']=='2030-01-01')]=np.nan  
		pd_dementia_dates=label_dates[['eid','parkins_date','dementia_date']]
		

		excludes=pd.DataFrame(dis_full[mask_dis].groupby(['eid','disease'])['dis_exc']\
	.sum().unstack('disease')).reset_index()
		excludes=pd.merge(df['eid'],excludes,how='left',on='eid')
		excludes.fillna(0,inplace=True)
		excludes.columns=['eid']+[col+'_exc' for col in labels.columns if col!='eid']

		#export all parquet files for use
		dis_ohe_ukb.to_parquet(self.path+'dis_ohe_test.parquet')
		labels.to_parquet(self.path+'labels_test.parquet')
		label_dates.to_parquet(self.path+'labels_dates_test.parquet')
		pd_dementia_dates.to_parquet(self.path+'pd_dem_disease_dates.parquet')
		excludes.to_parquet(self.path+'excludes_test.parquet')

		return dis_ohe_ukb,labels,label_dates


	def onehotencoder(self,df,cols,excwords,maxrecs=10,mincount=0.8,incspec=True):


	
		#create nulls where unknown for future imputation
		for col in cols:
			mask_exc=(df[col].isin(excwords))
			df[col][mask_exc]=np.nan  
		ohe_cols=\
		[col for col in cols if len(df[col].value_counts())<maxrecs
		and df[col].count()/df[col].shape[0]>mincount]

		#include the exceptions here and make a list of set in case there are duplicates
		if incspec:
			ohe_cols=list(set(list(ohe_cols+self.ohe_exceps)))

		rejcols=[c for c in cols if c not in ohe_cols and len(df[c].value_counts())<maxrecs]

		print(len(rejcols)+len(ohe_cols)-len(cols))
		
		print(r'Total ohe variables = %0.0f ' % (len(ohe_cols)))
			
		df_ohe_cols=df[ohe_cols]
			
		df_out=pd.get_dummies(df_ohe_cols, prefix=df_ohe_cols.columns,
				columns=df_ohe_cols.columns)
		
		df_out['eid']=df['eid'].tolist()
		
		return df_out,rejcols

	

	def create_dic(self,df,maxnum=15):
		return [(col,set(df[col][pd.notnull(df[col])].unique())) for col in df.columns if len(df[col].unique())<=maxnum]

	def ordinal_lookup(self,df,outfile='vallistcomp_test.csv'):
		list1=self.create_dic(df,maxnum=15)
		valslist=pd.DataFrame(list1)
		valslist.columns=['column','values']
		valslist['merge']=valslist['values'].astype(str)
		vallist2=pd.DataFrame(valslist['values'].value_counts()).reset_index()
		vallist2.columns=['values','recs']
		vallist2['merge']=vallist2['values'].astype(str)
		vallistcomp=pd.merge(valslist['column'],vallist2,on='merge',how='left')
		vallistcomp.sort_values(by='recs',ascending=False,inplace=True)
		vallistcomp.to_csv(self.path+outfile)

		return vallistcomp


	def ordcols_import(self):
		ordcols_full=pd.read_csv(self.path+'vallistcomp_edited.csv')
		ordcols=ordcols_full[(ordcols_full['Do']!='Exclude')&(ordcols_full['Do']!='ohe')&\
			pd.notnull(ordcols_full['Do'])][['column','Do']]
		ordcols["inv_map"]=""
		for i,col in enumerate(ordcols['column']):
			try:
				mask=(ordcols['column']==col)
				val=np.asarray(ordcols['Do'][mask])[0]
				
				orig_map=ast.literal_eval(str(val))
			
				inv_map = {v: k for k, v in orig_map.items()}
				ordcols["inv_map"][mask]=str(inv_map)
			except:
				pass
		ordcols['Do']=ordcols['Do'].str.replace("'Do not know':1,","")

		return ordcols,ordcols_full


	def cols_nonmiss(self):
		cols_nomiss=pd.read_csv(self.path+self.cols_nonmiss_file)
		cols_nomiss=list(cols_nomiss['colname'])
		cols_nomiss=[col for col in cols_nomiss if not re.search(self.excs,col) ]
		return cols_nomiss

	def load_df(self):
		cols_nomiss=self.cols_nonmiss()
		df=pd.read_parquet(self.path+self.fullfile,columns=cols_nomiss)
		df=self.convert_null(df)
		return df


	def vars_det(self,df):
		ordcols,ordcols_full=self.ordcols_import()
		excvars=list(ordcols_full[(ordcols_full['Do']=='Exclude')]['column'])
		ordvars=list(ordcols['column'])
		ctsvars=[col for col in df.columns if re.search('float',str(df[col].dtype))
		and not re.search(self.excs_cts,col)]
		ohe_vars=[col for col in df.columns if col not in ordvars+ctsvars+excvars
		and not re.search(self.excs_cts,col)]

		excluded_vars=[c for c in df.columns if c not in ordvars+ctsvars+ohe_vars]
		return ctsvars,ohe_vars,excluded_vars,ordvars

	def ord_data(self,df):
		ordcols=self.ordcols_import()[0]
		ordvars=[col for col in ordcols['column']]
		df_ord=df[['eid']+ordvars]
		for i,col in enumerate(ordcols['column']):
			df_ord[col]=df[col].map(ast.literal_eval(ordcols['Do'].iloc[i]))
		return df_ord

	def ohe_data(self,df):
		ohe_vars=self.vars_det(df)[1]
		ohe_df,ohe_excluded=self.onehotencoder(df=df[ohe_vars+['eid']],cols=ohe_vars,excwords=self.excwords)
		return ohe_df,ohe_excluded

	def cts_data(self,df):
		ctsvars=self.vars_det(df)[0]
		cts_df=df[ctsvars+['eid']]
		return cts_df

	def model_data(self,df):

		excluded_vars=self.vars_det(df)[2]
		ohe_df,excvars=self.ohe_data(df)
		excluded_vars=excluded_vars+excvars
		cts_df=self.cts_data(df)
		df_ord=self.ord_data(df)

		df_fam_pddem=pd.read_parquet(self.path+'df_fam_pddem.parquet')
		apoe4_df=pd.read_parquet(self.path+'genotype.parquet')
		ukb_treatments=pd.read_parquet(self.path+'treatments_test.parquet')



		for i,df in enumerate([df_fam_pddem,apoe4_df,ukb_treatments]):
			df['eid']=df['eid'].astype(str)	



		df=pd.merge(df_ord,cts_df,on='eid',how='left')
		df=pd.merge(df_model,ohe_df,on='eid',how='left')
		
		df=pd.merge(df,ukb_treatments,on='eid',how='left')
		df=pd.merge(df,apoe4_df,on='eid',how='left')
		df=pd.merge(df,df_fam_pddem,on='eid',how='left')


		df.to_parquet(self.path+'df_model_test.parquet')

		return df,excluded_vars

	def mod_data_append(self):
		df=pd.read_parquet(self.path+'df_model_test.parquet')
		df['eid']=df['eid'].astype(str)

		print(df.shape)

		df_fam_pddem=pd.read_parquet(self.path+'df_fam_pddem.parquet')
		apoe4_df=pd.read_parquet(self.path+'genotype.parquet')
		ukb_treatments=pd.read_parquet(self.path+'treatments_test.parquet')

		for i,df1 in enumerate([df_fam_pddem,apoe4_df,ukb_treatments]):
			df1['eid']=df1['eid'].astype(str)	

		print(df.shape)

		df=pd.merge(df,ukb_treatments,on='eid',how='left')
		df=pd.merge(df,apoe4_df,on='eid',how='left')
		df=pd.merge(df,df_fam_pddem,on='eid',how='left')

		print(df.shape)

		df.to_parquet(self.path+'df_model_test_treat.parquet')

		return df

	

	def data_merge_dis(self,remwords='alzhei|dementia',disease='AD',icd10s=['G309', 'G308', 'G300', 'G301'],
		outfile=None,use_icd10=True,strcont=False,bef=False,years=2):

		#label with the disease label you want and enter a list of correspoonding ICD10s
		#df_lab=self.disease_labels_ICD10s(icd10s=icd10s,disease=disease,strcont=strcont)

		df_lab=self.disease_labels_ICD10s2(icd10s=icd10s,disease=disease,strcont=strcont,bef=bef,\
	years=years,outfile=outfile)[0]


		df_model=pd.read_parquet(self.path+'df_model.parquet')
		ukb_treatments=pd.read_parquet(self.path+'treatments_test.parquet')
		dis_ohe=pd.read_parquet(self.path+'dis_ohe_test.parquet')
		dis_ohe_icd10=pd.read_parquet(self.path+'dis_ohe_icd10_test.parquet')
		df_fam_pddem=pd.read_parquet(self.path+'df_fam_pddem.parquet')
		deaths=pd.read_parquet(self.path+'deaths_test.parquet')

		apoe4_df=pd.read_parquet(self.path+'genotype.parquet')
		apoe4_df=apoe4_df[pd.notnull(apoe4_df['Genotype'])]
		apoe4_df=self.onehotencoder(apoe4_df,['Genotype'],[],maxrecs=10,mincount=0.1,incspec=False)[0]

		deaths=deaths[(deaths['date_of_death_f40000_0_0']!='nan')]
		deaths['date_of_death_f40000_0_0']=pd.to_datetime(deaths['date_of_death_f40000_0_0'])

		for i,df in enumerate([df_lab,dis_ohe,dis_ohe_icd10,deaths,df_model,ukb_treatments,apoe4_df,df_fam_pddem]):
			df['eid']=df['eid'].astype(str)	

		#deaths not of the condition - we will include deaths from the condition
		#where condition was developed prior to 2 year interval

		df=pd.merge(df_model,ukb_treatments,on='eid',how='left')
		df=pd.merge(df,apoe4_df,on='eid',how='left')
		df=pd.merge(df,df_fam_pddem,on='eid',how='left')

		if bef:
			word='since'
		else:
			word='to'
		df=pd.merge(df,df_lab[['eid',disease,'time_'+word+'_'+disease]],on='eid',how='left')

		if use_icd10:
			df=pd.merge(df,dis_ohe_icd10,on='eid',how='inner')
		else:
			df=pd.merge(df,dis_ohe,on='eid',how='inner')



		nondis_deaths=list(deaths['eid'][~(deaths['eid'].isin(df_lab['eid']))])
		dis_befores=list(df_lab['eid'][(df_lab[disease]==-1)])

		eid_excludes=list(set(nondis_deaths+dis_befores))

		col_includes=list(c for c in df.columns if not re.search(remwords,c) or c==disease)

		df[disease].fillna(0,inplace=True)

		df=df[~(df['eid'].isin(eid_excludes))]

		df=df[col_includes]

		df['eid']=df['eid'].astype(str)

		if outfile:
			df.to_parquet(self.path+outfile)

		return df


	def data_merge(self,use_icd10=True):


		df_model=pd.read_parquet(self.path+'df_model.parquet')
		ukb_treatments=pd.read_parquet(self.path+'treatments_test.parquet')
		dis_ohe=pd.read_parquet(self.path+'dis_ohe_test.parquet')
		dis_ohe_icd10=pd.read_parquet(self.path+'dis_ohe_icd10_test.parquet')
		df_fam_pddem=pd.read_parquet(self.path+'df_fam_pddem.parquet')

		labels=pd.read_parquet(self.path+'labels_test.parquet')
		label_dates=pd.read_parquet(self.path+'labels_dates_test.parquet')
		excludes=pd.read_parquet(self.path+'excludes_test.parquet')
		deaths=pd.read_parquet(self.path+'deaths_test.parquet')
		
		#some operations on APOE4 genotype file
		apoe4_df=pd.read_parquet(self.path+'genotype.parquet')
		apoe4_df=apoe4_df[pd.notnull(apoe4_df['Genotype'])]
		apoe4_df=self.onehotencoder(apoe4_df,['Genotype'],[],maxrecs=10,mincount=0.1,incspec=False)[0]

		deaths=deaths[(deaths['date_of_death_f40000_0_0']!='nan')]
		deaths['date_of_death_f40000_0_0']=pd.to_datetime(deaths['date_of_death_f40000_0_0'])

		for i,df in enumerate([dis_ohe,dis_ohe_icd10,labels,excludes,deaths,df_model,ukb_treatments,apoe4_df,df_fam_pddem]):
			df['eid']=df['eid'].astype(str)

		mask_dem=~(labels[[col for col in labels.columns if 'dementia' in col]].sum(axis=1)>0)
		mask_pd=~(labels[[col for col in labels.columns if 'g20parkinsons_disease' in col]].sum(axis=1)>0)
		mask_ad=~(labels[[col for col in labels.columns if 'alzhei' in col]].sum(axis=1)>0)

		#Exclude everyone who died from something other than dementia, PD etc. and create exclusion sets to process
		
		dem_excs=list(excludes[(excludes[[col for col in excludes.columns if 'dementia' in col]].sum(axis=1)>0)]['eid'])
		death_exc_dem=list(pd.merge(deaths,labels[mask_dem],on='eid',how='inner')['eid'])
		eids_exc_dem=list(dem_excs)+list(death_exc_dem)

		PD_excs=list(excludes[(excludes[[col for col in excludes.columns if 'parkinson' in col]].sum(axis=1)>0)]['eid'])
		death_exc_pd=list(pd.merge(deaths,labels[mask_pd],on='eid',how='inner')['eid'])
		eids_exc_pd=list(PD_excs)+list(death_exc_pd)

		AD_excs=list(excludes[(excludes[[col for col in excludes.columns if 'AD' in col]]\
		.sum(axis=1)>0)]['eid'])+dem_excs
		death_exc_ad=list(pd.merge(deaths,labels[mask_ad],on='eid',how='inner')['eid'])
		eids_exc_ad=list(AD_excs)+list(death_exc_ad)

		if use_icd10:
			df=pd.merge(df_model,dis_ohe_icd10,on='eid',how='inner')
		else:
			df=pd.merge(df_model,dis_ohe,on='eid',how='inner')

		df=pd.merge(df,ukb_treatments,on='eid',how='left')
		df=pd.merge(df,apoe4_df,on='eid',how='left')
		df=pd.merge(df,df_fam_pddem,on='eid',how='left')

		labels['dementia']=labels[[col for col in labels.columns if 'dementia' in col]].max(axis=1)
		labels['PD']=labels[[col for col in labels.columns if 'parkinson' in col]].max(axis=1)
		labels['AD']=labels[[col for col in labels.columns if 'alzh' in col]].max(axis=1)

		df=pd.merge(df,labels[['eid','dementia','PD','AD']],on='eid',how='inner')

		#exclude variables for final outputs
		mask=(df['eid'].isin(eids_exc_dem))
		df_dem=df[~mask]
		df_dem.drop(columns=[col for col in self.findcols(df_dem,'dementia|AD') if col!='dementia'],inplace=True)

		mask=(df['eid'].isin(eids_exc_pd))
		df_pd=df[~mask]
		df_pd.drop(columns=[col for col in self.findcols(df_dem,'parkins') if col!='PD'],inplace=True)

		mask=(df['eid'].isin(eids_exc_ad))
		df_ad=df[~mask]
		df_ad.drop(columns=[col for col in self.findcols(df_dem,'dement') if col!='AD'],inplace=True)

		df_dem.to_parquet(self.path+'df_dem_20211024.parquet')
		df_pd.to_parquet(self.path+'df_pd_20211024.parquet')
		df_ad.to_parquet(self.path+'df_ad_20211024.parquet')
		df.to_parquet(self.path+'df_all_20211024.parquet')

		return df_dem,df_pd,df_ad,df


	def remap_var(self,df,var,dictvar,drop=False):
		df[var]=0
		for col in dictvar:
			mask=(df[col]==1)
			df[var][mask]=dictvar[col]
		if drop:
			df.drop(columns=list(dictvar),inplace=True)
			
		return df

	def frailty_index(self,gender,bmi,left_grip,right_grip):

		"""
		function to compute frailty indices by gender and BMI
		"""
		grip=0
		if gender==0.0:
			if (max(left_grip,right_grip)<=29 and bmi<=24) or (max(left_grip,right_grip)<=30 and bmi>24 and bmi<=26)\
	or (max(left_grip,right_grip)<=30 and bmi>26 and bmi<=28) or (max(left_grip,right_grip)<=32 and bmi>28):
				grip=1
				
		elif gender==1.0:
			if (max(left_grip,right_grip)<=17 and bmi<=23) or (max(left_grip,right_grip)<=17.3 and bmi>23 and bmi<=26)\
	or (max(left_grip,right_grip)<=18 and bmi>26 and bmi<=29) or (max(left_grip,right_grip)<=21 and bmi>29):
				grip=1
		return grip

		
	def studyvars(self,depvar="dementia"):

		"""
		this function maps all the columns used in previous studies and meta-analyses
		it also calculates/ transforms a number of variables such as AST:ALT ratio
		"""

		

		#exclude the PD cases if we are looking at dementia (check this logic)
		if depvar=="dementia":
			df=pd.read_parquet(self.path+'df_dem_20210924.parquet')
			mask=(df['PD']==1)
			df=df[~mask]

		if depvar=="AD":
			#df=pd.read_parquet(self.path+'df_ad_20211024.parquet')
			df=pd.read_parquet(self.path+'df_ad_20211214.parquet')

			if 'PD' in df.columns:
				mask=(df['PD']==1)
				df=df[~mask]


			

		elif depvar=="PD":
			PD_spec=pd.read_parquet('%s%s' % (self.path,'PD_specific.parquet'))
			df=pd.read_parquet(self.path+'df_pd_20211024.parquet')
			PD_spec=PD_spec[[c for c in PD_spec.columns if c not in df.columns or c=='eid']]
			df=pd.merge(df,PD_spec,on='eid',how='left')
			#mapping of PD variables
			df['pesticide_exposure']=df['worked_with_pesticides_f22614_0_0'].map(self.pest_map)
			df['urban_rural']=df['home_area_population_density_urban_or_rural_f20118_0_0'].map(self.urb_rur)

		elif depvar=="all":
			PD_spec=pd.read_parquet('%s%s' % (self.path,'PD_specific.parquet'))
			df=pd.read_parquet(self.path+'df_all_20211024.parquet')
			PD_spec=PD_spec[[c for c in PD_spec.columns if c not in df.columns or c=='eid']]
			df=pd.merge(df,PD_spec,on='eid',how='left')

		
		df['melanoma']=df[self.findcols(df,'melano')].max(axis=1)

		#remapping of these specific variables to ordinal
		df=self.remap_var(df=df,var="APOE4_Carriers",dictvar=self.genos,drop=False)
		df=self.remap_var(df=df,var="Qualif_Score",dictvar=self.qualif,drop=True)

		#neurochemical ratios
		df['AST_ALT_ratio']=df['aspartate_aminotransferase_f30650_0_0']/\
		df['alanine_aminotransferase_f30620_0_0']

		mask_inf=(df['lymphocyte_count_f30120_0_0']==0)|pd.isnull(df['lymphocyte_count_f30120_0_0'])
		df['neutrophill_lymphocyte_ratio']=np.nan
		df['neutrophill_lymphocyte_ratio'][~mask_inf]=df['neutrophill_count_f30140_0_0']/\
		df['lymphocyte_count_f30120_0_0']

		df['diabetes']=df[self.findcols(df,'diabetes')].max(axis=1)

		df['pollution']=df[['nitrogen_dioxide_air_pollution_2010_f24003_0_0',
		'nitrogen_oxides_air_pollution_2010_f24004_0_0',
		'particulate_matter_air_pollution_pm10_2010_f24005_0_0',
		'particulate_matter_air_pollution_pm25_2010_f24006_0_0',
		'particulate_matter_air_pollution_pm25_absorbance_2010_f24007_0_0',
		'particulate_matter_air_pollution_2510um_2010_f24008_0_0',
		'nitrogen_dioxide_air_pollution_2005_f24016_0_0',
		'nitrogen_dioxide_air_pollution_2006_f24017_0_0',
		'nitrogen_dioxide_air_pollution_2007_f24018_0_0']].mean(axis=1)

		df['low_activity']=df['ipaq_activity_group_f22032_0_0'].apply(lambda x:1 if x=='low' else 0)

		colsfrail=['weight_change_compared_with_1_year_ago_f2306','recent_feelings_of_tiredness_or_low_energy_f20519',
		'ipaq_activity_group_f22032_0_0','usual_walking_pace_f924','hand_grip_strength_left_f46','hand_grip_strength_right_f47']


		df['sedentary_time']=df[[ 'time_spent_watching_television_tv_f1070_0_0',
		'time_spent_using_computer_f1080_0_0',
		'time_spent_driving_f1090_0_0']].sum(axis=1)

		#frailty calculations
		
		colsfrail=['weight_change_compared_with_1_year_ago_f2306','recent_feelings_of_tiredness_or_low_energy_f20519',
		'ipaq_activity_group_f22032_0_0','usual_walking_pace_f924','hand_grip_strength_left_f46','hand_grip_strength_right_f47']

		df['low_activity']=df['ipaq_activity_group_f22032_0_0'].apply(lambda x:1 if x=='low' else 0)

		df['grips_frail']=df[['sex_f31_0_0','body_mass_index_bmi_f21001_0_0','hand_grip_strength_left_f46_0_0',\
'hand_grip_strength_right_f47_0_0']].apply(lambda x:self.frailty_index(x['sex_f31_0_0'],x['body_mass_index_bmi_f21001_0_0'],\
 x['hand_grip_strength_left_f46_0_0'],x['hand_grip_strength_right_f47_0_0']),axis=1)

		df['exhaust_frail']=df['frequency_of_tiredness_lethargy_in_last_2_weeks_f2080_0_0'].isin([2,3]).astype(int)
		df['walk_frail']=df['usual_walking_pace_f924_0_0'].isin([0]).astype(int)
		df['ipaq_frail']=df['ipaq_activity_group_f22032_0_0'].isin([0]).astype(int)

		df['frailty_score']=\
		df[['grips_frail','exhaust_frail','walk_frail','ipaq_frail']]\
		.sum(axis=1)
		df['frailty_index']=df['frailty_score'].apply(lambda x:0 if x<1 else
																(2 if x>=3 else 1))

		df['hypertension']=df[self.findcols(df,'hypertension')].max(axis=1)


		df['alcohol']=np.nan
		alc_cols=['alcohol_intake_frequency_f1558_0_0_Daily or almost daily',
		'alcohol_intake_frequency_f1558_0_0_Never',
		'alcohol_intake_frequency_f1558_0_0_Once or twice a week',
		'alcohol_intake_frequency_f1558_0_0_One to three times a month',
		'alcohol_intake_frequency_f1558_0_0_Three or four times a week']

		for col in alc_cols:
			df['alcohol'][(df[col]==1)]=self.alc_map[col]

		df['depressed']=df[['Major depressive disorder, recurrent, unspecified',
		'Major depressive disorder, single episode, moderate',
		'Major depressive disorder, single episode, unspecified']].max(axis=1)

		studycols_dem=list(self.genos)+['dementia','eid','age_when_attended_assessment_centre_f21003_0_0','APOE4_Carriers',
		'pollution','sedentary_time','diabetes','low_activity','salad_raw_vegetable_intake_f1299_0_0',
		'fresh_fruit_intake_f1309_0_0','weight_change_compared_with_1_year_ago_f2306_0_0',
		'frequency_of_tiredness_lethargy_in_last_2_weeks_f2080_0_0','ipaq_activity_group_f22032_0_0','usual_walking_pace_f924_0_0',
		'hand_grip_strength_left_f46_0_0','hand_grip_strength_right_f47_0_0','body_mass_index_bmi_f21001_0_0',
		'systolic_blood_pressure_automated_reading_f4080_0_0','diastolic_blood_pressure_automated_reading_f4079_0_0',
		'frailty_score','smoking_status_f20116_0_0','cholesterol_f30690_0_0','hdl_cholesterol_f30760_0_0',
		'processed_meat_intake_f1349_0_0','mean_time_to_correctly_identify_matches_f20023_0_0',
		'number_of_incorrect_matches_in_round_f399_0_2','sex_f31_0_0','hypertension','ever_smoked_f20160_0_0','alcohol','TBI',
		'Hear_loss','Qualif_Score']

		cols_lancet=list(self.genos)+['dementia','eid','age_when_attended_assessment_centre_f21003_0_0','APOE4_Carriers','TBI',
		'alcohol','pollution','hypertension','diabetes','Hear_loss','ever_smoked_f20160_0_0','body_mass_index_bmi_f21001_0_0',
		'depressed','smoking_status_f20116_0_0','ipaq_activity_group_f22032_0_0','Qualif_Score',
		'frequency_of_friendfamily_visits_f1031_0_0']

		var_renames=dict({'total_dis':'Total ICD10 Conditions at baseline'})

		
		df['Retired']=0
		mask=(df['current_employment_status_f6142_0_0_Retired']==1)
		df['Retired'][mask]=1
	

		for v in var_renames:
			if v in df.columns:
				df.rename(columns={v:var_renames[v]},inplace=True)


		#more combined features
		inflamvars=['neutrophill_count_f30140_0_0','lymphocyte_count_f30120_0_0','creactive_protein_f30710_0_0','platelet_count_f30080_0_0']
		frail_cols=['Total ICD10 Conditions at baseline','number_of_treatmentsmedications_taken_f137_0_0',
		'frailty_index','walk_frail','ipaq_frail','grips_frail']

		#defined combined features as sum of standard scaled features
		df=self.std_scale_newvar(df,inflamvars,name='inflammation')
		df=self.std_scale_newvar(df,frail_cols,name='frailty')





		if depvar=="dementia":

			df_study=df[studycols_dem]
			df_lancet=df[cols_lancet]
			df_lancet.to_parquet(self.path+'df_dem_lancet_Oct.parquet')
			df_study.to_parquet(self.path+'df_dem_frailty_study_Oct.parquet')
			df.to_parquet(self.path+'df_dem_final.parquet')

			df_tuple=[df,df_study,df_lancet]

		if depvar=="AD":

			df_study=df[[c for c in studycols_dem if 'dementia' not in c]+['AD']]
			df_lancet=df[[c for c in cols_lancet if 'dementia' not in c]+['AD']]
			df_lancet.to_parquet(self.path+'df_AD_lancet_Oct.parquet')
			df_study.to_parquet(self.path+'df_AD_frailty_study_Oct.parquet')
			df.to_parquet(self.path+'df_AD_final.parquet')

			df_tuple=[df,df_study,df_lancet]

		elif depvar=="PD":
			
			#drop the object columns we brought in
			df.drop(columns=[col for col in df.columns if col!='eid' and re.search('obj',str(df[col].dtype))],inplace=True)
			df.to_parquet(self.path+'df_PD_final.parquet')
			df_tuple=[df]
		
		elif depvar=="all":

			df.drop(columns=[col for col in df.columns if col!='eid' and re.search('obj',str(df[col].dtype))],inplace=True)
			df.to_parquet(self.path+'df_all_final.parquet')
			df_tuple=[df]



		return df_tuple






		


	
		







