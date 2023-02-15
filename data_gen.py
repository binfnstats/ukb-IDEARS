#purpose of this is to take raw data and transform to dataset for modelling
#should bring raw data in (except genetics) and transform to create specific modelling datasets e.g. MRI etc. 
#which could be different classes
#also need to bring in and transform ordinal variables etc.

import pandas as pd
import numpy as np
import ast
import re
import icd10
import datetime as dt

class data_import(object):

	def __init__(self):	

		self.run_date=dt.date.today()

		self.path='/Users/michaelallwright/Documents/data/ukb/'
		self.field_names=self.path+'metadata/ukb_field_names.xlsx'
		self.static_path='/Users/michaelallwright/Documents/github/ukb/pipeline/static/'
		self.inpfile='full_data/hamish_preprocessing.csv'
		self.all_meds=['nonsteroidal anti-inflammatory drugs','Beta Blocker','calcium channel blockers','anti_inf_steroid']
		self.cols=None
		self.colmap_file='ordinal_set_mapping.csv'
		self.remwords='freezethaw|_acquisition_route|index_for_card|_reportability_|correction_|inpatient_record|noncancer_illness|\
operation_yearage|number_of_times_snapbutton|quality_control|authorisation|probeintensity|invitation|volume_of|\
acquisition_time|_duration|device_id|sample_collection|time_since_interview_|\
polymorphic|aliquot|pct_responsible|gp_was_registered'
		
		#override to import these columns
		self.cols_needed='dementia|parkinsons|worked_with_pesticides_f22614_0_0|\
number_of_selfreported_noncancer_illnesses_f135_0_0|recent_feelings_of_tiredness'
		
		#specific cols for PD
		self.PDcols=['eid','worked_with_pesticides_f22614_0_0','home_area_population_density_urban_or_rural_f20118_0_0',
	   'single_episode_of_probable_major_depression_f20123_0_0','probable_recurrent_major_depression_moderate_f20124_0_0',
		'probable_recurrent_major_depression_severe_f20125_0_0','bipolar_and_major_depression_status_f20126_0_0',
		'neuroticism_score_f20127_0_0' ,'recent_feelings_or_nervousness_or_anxiety_f20506_0_0',
		'daytime_dozing_sleeping_narcolepsy_f1220_0_0']
		
		#mapping locations to urban/rural binary variable
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


		
	#helper function to find columns in dataframe
	def findcols(self,df,string):
		return [col for col in df if re.search(string,col)]

	#read in a sample of the full raw dataframe
	def read_all_samp(self,samp=100):
		df=pd.read_pickle(self.path+'df_ukb_full_samp.p')
		#df=pd.read_csv(self.path+self.inpfile,nrows=samp)
		return df

	#read in specific columns
	def read_all_cols(self,cols):
		df=pd.read_csv(self.path+self.inpfile,usecols=cols,engine='python')
		return df

	def all_col_names(self):
		cols=list(self.read_all_samp().columns)
		self.cols=cols
		return cols

	#read in the field names mapping table
	def get_field_names(self):
		df=pd.read_excel(self.field_names,sheet_name='fieldnames_full')
		return df

	def get_cols_with_string(self,df,remstrings=None):
		#remove columns with certain strings from dataframe

		if remstrings is None:
			remstrings=self.remwords

		#remstrings='|'.join(remstrings)
		remvars=[c for c in df.columns if re.search(remstrings,c)]
		return remvars

	def get_time_periods(self,df):
		# based on string at end of column returns columns which are for a later time period
		later_periods=[c for c in df.columns if c[len(c)-3:len(c)]=='1_0' or\
c[len(c)-3:len(c)]=='2_0' or c[len(c)-3:len(c)]=='3_0'] 
		return later_periods

	def get_obj_cols(self,df):
		#should not model object column so this removed them
		all_dtypes=list(str(c) for c in df.dtypes)
		obj_cols=[c for i,c in enumerate(df.columns) if re.search('obj',all_dtypes[i])]
		
		return obj_cols

	def remove_cols(self,df):
		remvars=self.get_cols_with_string(df)
		later_periods=self.get_time_periods(df)
		#obj_cols=self.get_obj_cols(df)
		cols_rem=[c for c in list(set(remvars+later_periods)) if c!='eid']
		df.drop(columns=cols_rem,inplace=True)
		return df,cols_rem

	#get brain data

	def get_raw_data(self,min_part=250000,outfile='ukb_gt50perc.parquet',out=True):

		"""
		get the full data where a minimum number of participants are in the field names file
		and certain words are not in the string of columns
		Output to parquet file
		"""

		df=self.get_field_names()
		mask=(df['Participants']>min_part)
		cols_gt50=list(df.loc[mask,'col.name'])
		
		#dementia related columns
		extcols=self.findcols(df,'dementia')+['eid']

		cols_gt50=[c for c in cols_gt50 if not re.search(self.remwords,c)]
		cols_gt50=list(set(cols_gt50+extcols))
		df1=pd.read_csv(self.path+self.inpfile,usecols=cols_gt50)
		df1['eid']=df1['eid'].astype(str)

		if out:
			df1.to_parquet(self.path+outfile)  

		return df1

	def get_other_cols(self):
		#bring in other columns required
		df=self.read_all_samp()
		cols_new=['eid']+self.findcols(df,self.cols_needed)
		df=self.read_all_cols(cols_new)
		df['eid']=df['eid'].astype(str)
		return df

	def get_cts_cols(self,df=None,out=True,outfile='ukb_cts.parquet'):
		if df is None:
			df=self.get_raw_data(min_part=250000,out=False)

		cts_cols=[c for c in df.columns if df[c].nunique()>10 and not re.search(self.remwords,c) or c=='eid']
		cts_cols0=[c for c in cts_cols if c[len(c)-3:len(c)]=="0_0" or c=='eid']

		if out:
			df[cts_cols0].to_parquet(self.path+'ukb_cts.parquet')

		return df[cts_cols0]

	def get_merge_map(self):

		# create a mapping from the ordered set of values in a column to that ordered set's numeric conversion based on an Excel mapping created
		df_map=pd.read_csv(self.static_path+self.colmap_file)
		mask=pd.notnull(df_map['map'])
		merge_map=dict(zip([str(sorted(ast.literal_eval(c))) for c in list(df_map.loc[mask,'set'])],[str(ast.literal_eval(c)) for c in list(df_map.loc[mask,'map'])]))

		return merge_map


	def map_cols(self,df=None,imp_parq=True):
		#map columns to ordinal by taking the dictionaries constructed for each file and mapping values 
		if df is None:
			if imp_parq:
				df=pd.read_parquet(self.path+'ukb_gt50perc.parquet')
			else:
				df=self.get_raw_data(min_part=250000,out=False)

		merge_map=self.get_merge_map()
		print("got merge map")

		ordinal_cols=[c for c in df.columns if df[c].nunique()<10 and not re.search(self.remwords,c)]
		df_out=df[['eid']+ordinal_cols].copy()

		# determine each column and corresponding sorted list of values removing null and Prefer not to answer
		cols=[]
		merges=[]
		null_words=['Not known', 'Not applicable', 'Do not know', 'Unsure', 'None of the above','Prefer not to answer','Not sure']
		for c in [c for c in ordinal_cols if c!='eid']:
			mask=pd.isnull(df_out[c])|(df_out[c].isin(null_words))
			cols.append(c)
			merges.append(sorted(c for c in df_out.loc[~mask,c].unique()))

		#using the merge map, construct a dictionary that maps each column from above with its corresponding numerical value dictionary found in merge_map,
		#for the cases where merge_map covers it

		col_mapping=dict(zip([cols[i] for i,c in enumerate(merges) if str(c) in merge_map],\
	[merge_map[str(c)] for c in merges if str(c) in merge_map]))

		print("got col mapping")


		#of the original intended ordinal columns to map, determine which ones did and didn't map so we can output those that did and did not
		
		ordinal_cols_mapped=[c for c in ordinal_cols if c in col_mapping]
		ordinal_cols_unmapped=[c for c in ordinal_cols if c not in col_mapping]

		print("up to mapping")

		df_out=df_out[['eid']+ordinal_cols_mapped]

		for c in ordinal_cols_mapped:
			dic=ast.literal_eval(col_mapping[c])
			df_out[c]=df_out[c].map(dic)

		return df_out,ordinal_cols_unmapped,col_mapping
	

	

	def ohe_cols(self,df,cts_cols,ordcols,perc=0.8,uvals=10):

		#determine set of columns for ohe which consists of the non 
		cols=[c for c in df.columns if c not in cts_cols and c not in ordcols and df[c].count()>perc*df.shape[0]
		and df[c].nunique()<=uvals and not re.search(self.remwords,c) and c!='eid']
		df_ohe=pd.concat([pd.get_dummies(df[[c]]) for c in cols]+[df['eid']],axis=1)
		return df_ohe


	# treatment fields
	def treatcols(self):
		
		if self.cols is None:
			self.all_col_names()

		treatcols=[c for c in self.cols if 'f20003' in c or c=='eid']
		df=pd.read_csv('%s%s' % (self.path,self.inpfile),usecols=treatcols)
		df.fillna(0,inplace=True)
		df=df.astype(int)
			
		return df

	#treatment coding dictionary
	def treat_dic(self):
		df=pd.read_csv(self.static_path+'coding4.tsv',sep="\t")
		dic=dict(zip(df['coding'],df['meaning']))
		return dic

	def get_treat_names(self,df,dic):
		for c in df.columns:
			if c!='eid':
				df[c]=df[c].map(dic)
		return df

	def treat_melt(self,df):
		df=pd.melt(df,id_vars=['eid'])
		return df

	def treat_codes_map(self):

		"""
		function to map words to categories for treatments
		"""
		df=pd.read_csv(self.static_path+'medications_codes.csv')
	
		words=[]
		
		for c in self.all_meds:
			mask=(df[c]=="yes")
			word=list(df.loc[mask,'treatment/med'].astype(str))
			word2='|'.join(word)
		   
			words.append(word2)
		dic=dict(zip(self.all_meds,words))   

		return dic 	

	def find_val(self,x,word_dic,drug='Beta Blocker'):	
		#searches dictionary for each drug
		if re.search(str(word_dic[drug]),x):
			y=1
		else:
			y=0
		return y

	def get_top_treats(self,df,min_par=500):
		df_sum=pd.DataFrame(df.groupby(['value'])['eid'].nunique()).reset_index()
		mask=(df_sum['eid']>min_par)
		vals=list(df_sum.loc[mask,'value'])

		mask=(df['value'].isin(vals))
		df_out=pd.DataFrame(df.loc[mask,].groupby(['eid','value']).size().unstack('value')).reset_index()
		df_out.fillna(0,inplace=True)

		for c in vals:
			mask=(df_out[c]>1)
			df_out.loc[mask,c]=1


		return df_out


	def get_treatment_data(self):

		df_treat=self.treatcols()
		dic=self.treat_dic()
		df_treat=self.get_treat_names(df_treat,dic)
		df_treat=self.treat_melt(df_treat)
		df_treat.dropna(inplace=True)

		word_dic=self.treat_codes_map()

		for m in self.all_meds:
			df_treat[m]=df_treat['value'].astype(str).apply(lambda x: self.find_val(x,word_dic=word_dic,drug=m))

		df_treat_sum=pd.DataFrame(df_treat.groupby('eid')[self.all_meds].max()).reset_index()

		df_top_treats=self.get_top_treats(df=df_treat,min_par=10000)

		df_treat_sum=pd.merge(df_treat_sum,df_top_treats,on='eid',how='outer')

		return df_treat_sum

		

	def get_icd10s(self):

		icdextcols=['age_when_attended_assessment_centre_f21003_0_0','date_of_attending_assessment_centre_f53_0_0',\
	'date_of_death_f40000_0_0','eid']
		if self.cols is None:
			self.all_col_names()

		cols=[c for c in self.cols if '41270' in c or '41280' in c or c in icdextcols]
		df=pd.read_csv('%s%s' % (self.path,self.inpfile),usecols=cols)
		df.to_parquet('%s%s' % (self.path,'ukb_icd10s.parquet'))

		return df

	def get_deaths(self,df):
		# applied to the ICD10 dataset, create a binary variable for death

		mask=pd.notnull(df['date_of_death_f40000_0_0'])

		df['death']=0
		df.loc[mask,'death']=1
		return df[['eid','death']]

	def melt_dis(self,df,val='disease'):
		#turn disease columns into rows applied to both diseases and their dates

		df = pd.melt(df, id_vars='eid', value_name=val)
		df=df[pd.notnull(df[val])]
		df.columns=['eid','variable',val]
		return df

	def split_disease_dfs(self,df):
		# create 2 dataframes, one for diseases and another for their corresponding dates

		cols_dis=[col for col in df.columns if '41270' in col or 'eid' in col]
		cols_date=[col for col in df.columns if '41280' in col or 'eid' in col]

		df_dis=df[cols_dis]
		df_date=df[cols_date]

		#replace variables so these can be merged later and then melt
		
		df_dis=self.melt_dis(df=df_dis,val='disease')
		df_dis['disease']=df_dis['disease'].str.replace("'","")
		df_dis['variable']=df_dis['variable'].str.replace('diagnoses_icd10_','')

		#replace variables so these can be merged later and format appropriately and melt
		df_date=self.melt_dis(df=df_date,val='dis_date')
		df_date['variable']=df_date['variable'].str.replace('41280','41270')
		df_date['variable']=df_date['variable'].str.replace('date_of_first_inpatient_diagnosis_icd10_','')
		
		df_date['dis_date']=df_date['dis_date'].str.replace('b','')
		df_date['dis_date']=df_date['dis_date'].str.replace("'","")
		df_date['dis_date']=pd.to_datetime(df_date['dis_date'])


		#remerge together
		df_dis_date=pd.merge(df_dis,df_date,on=['eid','variable'],how='left')

		#remerge to key variables for calculating disease times later
		df_dis_date=pd.merge(df_dis_date,df[['eid','date_of_attending_assessment_centre_f53_0_0']])

		df_dis_date['date_of_attending_assessment_centre_f53_0_0']=\
	pd.to_datetime(df_dis_date['date_of_attending_assessment_centre_f53_0_0'])

		df_dis_date['years_dis']=((df_dis_date['date_of_attending_assessment_centre_f53_0_0']-df_dis_date['dis_date']).dt.days/365.25)


		return df_dis_date

	def get_icd10_names(self):
		df=pd.read_csv(self.path+'metadata/code_map2.csv')
		df=df.loc[df['Coding']==19,['Value','Meaning']]
		df['Meaning']=df['Meaning'].apply(lambda x:x.lower())
		icd10_lkup_dict=dict(zip(df['Value'],df['Meaning']))

		return icd10_lkup_dict

	def returndesc(self,string):
		code=icd10.find(str(string))
		if code:
			desc=code.description
		else:
			desc=string
		return desc

	def comorbid_map(self):
		df=pd.read_excel(self.static_path+'comorbid_map.xlsx',sheet_name='map')
		c_dict=dict(zip(df['code'],df['comorbid_name']))
		return c_dict

	def rename_diseases(self,df):
		# ICD10 codes mapped to a disease name
		icd10_lkup_dict=self.get_icd10_names()
		df['disease_name']=df['disease'].astype(str).apply(self.returndesc)
		mask=(df['disease_name']==df['disease'])
		df.loc[mask,'disease_name']=df['disease'].map(icd10_lkup_dict)
		return df

	def dis_ohe(self,df,wait_period=2):
		#map which diseases happened before or after
		
		mask=(df['years_dis']<wait_period)
		df['dis_aft']=0
		df.loc[mask,'dis_aft']=1
		mask=(df['years_dis']>0)
		df['dis_bef']=0
		df.loc[mask,'dis_bef']=1

		return df

	def disease_counts(self,df):

		#only count diseases at baseline
		mask=(df['years_dis']>0)
		df=df.loc[mask,]
		df['comorbid_disease']=df['disease'].map(self.comorbid_map())
		df_sum=pd.DataFrame(df.groupby('eid')['comorbid_disease','disease'].count()).reset_index()
		df_sum['eid']=df_sum['eid'].astype(str)
		
		return df_sum

	def dis_ohe_sum(self,df,disease_var='disease_name',dis_var='dis_bef',min_recs=500):

		#words we want to keep the disease for even if below min_recs
		dis_words='alzheim|melanoma|diabetes|hypertension|depress'

		#use this to generate number of conditions before and after for each individual
		df[disease_var]=df[disease_var].apply(lambda x:str(x).lower())

		df1=pd.DataFrame(df.groupby([disease_var])[dis_var].sum()).reset_index()
		df1.columns=[disease_var,dis_var]
		mask=(df1[dis_var]>min_recs)|(df1[disease_var].str.contains(dis_words,regex=True))
		df1=df1.loc[mask,]
		dis_bef_list=list(df1[disease_var])
		mask=(df[disease_var].isin(dis_bef_list))&(df[dis_var]>0)
		df=df.loc[mask,]	

		df1=pd.DataFrame(df.groupby(['eid',disease_var])[dis_var].max().unstack(disease_var)).reset_index()
		df1.fillna(0,inplace=True)

		return df1

	def get_diseases_all(self,min_recs=1000,import_df=True,out_disease=False,outfile_disease='dis_full.parquet'):

		if import_df:
			df=pd.read_parquet('%s%s' % (self.path,'ukb_icd10s.parquet'))
		else:
			df=self.get_icd10s()

		df_death=self.get_deaths(df)
		df=self.split_disease_dfs(df)
		df=self.rename_diseases(df)

		if out_disease:
			df_out_all=pd.merge(df,df_death,on='eid',how='left')
			df_out_all.to_parquet(self.path+outfile_disease)
		df_count=self.disease_counts(df)
		print(df_count['comorbid_disease'].sum())

		df1=self.dis_ohe(df,wait_period=2)
		df_ohe_bef=self.dis_ohe_sum(df1,dis_var='dis_bef',min_recs=min_recs)

		df_ohe_aft=self.dis_ohe_sum(df1,disease_var='disease',dis_var='dis_aft',min_recs=min_recs)

		aft_cols=['aft_'+ c if c!='eid' else c for c in df_ohe_aft.columns]

		df_ohe_aft.columns=aft_cols
		for dfa in [df_ohe_bef,df_count,df_death,df_ohe_aft]:
			dfa['eid']=dfa['eid'].astype(str)

		df_out=pd.merge(df_ohe_bef,df_count,on='eid',how='outer')
		df_out=pd.merge(df_out,df_death,on='eid',how='outer')

		df_out.fillna(0,inplace=True)

		return df_out,df_ohe_aft

	def parental_diseases(self):

		#this brings in the columns relating to parental illnesses and sums them up

		dfa=self.read_all_samp()
		#get illnessess of father and mother
		ill_cols=['eid']+self.findcols(dfa,'illnesses_of_fath')+self.findcols(dfa,'illnesses_of_moth')

		#restrict to illnesses at time period 1
		ill_cols=[c for c in ill_cols if c[len(c)-3:len(c)-2]=="0" or c=='eid']
		df=self.read_all_cols(cols=ill_cols)
		df_cp=df['eid'].copy()

		df=pd.melt(df,id_vars='eid')
		mask=(pd.notnull(df['value']))
		df=df.loc[mask,]
		df=pd.DataFrame(df.groupby(['eid','value']).size()).reset_index()

		df=pd.concat([df['eid'],pd.get_dummies(df['value'])],axis=1)
		cols=[c for c in df.columns if c!='eid']
		df=pd.DataFrame(df.groupby(['eid'])[cols].max()).reset_index()

		col_names=['eid']+['parental_'+str(c) for c in df.columns if c!='eid']
		df.columns=col_names
		df.rename(columns={"parental_Alzheimer's disease/dementia":"parental_dem","parental_Parkinson's disease":'parental_pd'},inplace=True)

		df=pd.merge(df_cp,df,on='eid',how='left')
		df.fillna(0,inplace=True)

		return df

	def grip_normalise(self,gender,bmi,left_grip,right_grip):

		"""
		function to compute normalised grip as part of frailty index
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

	def APOE4_carriers(self,x,y):
		#function to define APOE4 carriers based on 'rs429358' and 'rs7412'
		if x==np.nan and y==np.nan:
			z=np.nan
		elif x>1 and y>1:
			z=2
		elif x>0 or y>0:
			z=1
		elif x==0 and y==0:
			z=0
		else:
			z=np.nan
		return z

	def get_genetics(self):
		#bring genetics into the model based on SNPs identified
		df=pd.read_feather(self.path+'genetic/df_AD_chrom_1_22_20220920.feather')
		df['APOE4_Carriers']=df.apply(lambda x:self.APOE4_carriers(x['rs7412'],x['rs429358']),axis=1)

		#filter out null APOE4s
		mask=pd.notnull(df['APOE4_Carriers'])
		df=df.loc[mask,]
		return df



	def livingstone_calcs(self):
		return None


	def studyvars_add(self,df):

		#TODO - check pesticide exposure coming through
		#df['pesticide_exposure']=df['worked_with_pesticides_f22614_0_0'].map(self.pest_map)
		df['urban_rural']=df['home_area_population_density_urban_or_rural_f20118_0_0'].map(self.urb_rur)

		#no melanoma in dataset
		df['melanoma']=df[self.findcols(df,'melano')].max(axis=1)

		#remapping of these specific variables to ordinal
		#df=self.remap_var(df=df,var="APOE4_Carriers",dictvar=self.genos,drop=False)
		#df=self.remap_var(df=df,var="Qualif_Score",dictvar=self.qualif,drop=True)

		#neurochemical ratios
		df['AST_ALT_ratio']=df['aspartate_aminotransferase_f30650_0_0']/\
		df['alanine_aminotransferase_f30620_0_0']

		mask_inf=(df['lymphocyte_count_f30120_0_0']==0)|pd.isnull(df['lymphocyte_count_f30120_0_0'])
		df['neutrophill_lymphocyte_ratio']=np.nan
		df['neutrophill_lymphocyte_ratio'][~mask_inf]=df['neutrophill_count_f30140_0_0']/\
		df['lymphocyte_count_f30120_0_0']

		df['diabetes']=df['diabetes_diagnosed_by_doctor_f2443_0_0'].max()

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

		colsfrail=['weight_change_compared_with_1_year_ago_f2306_0_0','frequency_of_tiredness_lethargy_in_last_2_weeks_f2080_0_0',
		'ipaq_activity_group_f22032_0_0','usual_walking_pace_f924_0_0','hand_grip_strength_left_f46_0_0',
		'hand_grip_strength_right_f47_0_0']

		dur_cols=['time_spent_watching_television_tv_f1070_0_0',
		'time_spent_using_computer_f1080_0_0',
		'time_spent_driving_f1090_0_0']

		for c in dur_cols:
			mask=(df[c]<0)
			df.loc[mask,c]=np.nan

		#issue here is the nulls - this should get around that
		df['sedentary_time']=df[dur_cols].mean(axis=1)

		#frailty calculations
		
		
		df['low_activity']=df['ipaq_activity_group_f22032_0_0'].apply(lambda x:1 if x=='low' else 0)

		df['grips_frail']=df[['sex_f31_0_0','body_mass_index_bmi_f23104_0_0','hand_grip_strength_left_f46_0_0',\
'hand_grip_strength_right_f47_0_0']].apply(lambda x:self.grip_normalise(x['sex_f31_0_0'],x['body_mass_index_bmi_f23104_0_0'],\
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

		#define alcohol as greater than median intake
		df['alcohol']=0
		mask=(df['alcohol_intake_frequency_f1558_0_0']>df['alcohol_intake_frequency_f1558_0_0'].median())
		df.loc[mask,'alcohol']=1

		#probably inadequate definition
		#df['depressed']=df[['Major depressive disorder, single episode, unspecified']].max(axis=1)


		#colskeep=[c for c in df.columns if re.search('int|float',str(df[c].dtype)) or c=='eid' or 'date' in c]

		return df#[colskeep]


	def dis_list(self,df=None,icd10s=['G30']):
		#dependent variable
	
		icd10s=''.join(icd10s)
		
		if df is None:
			df=pd.read_parquet('%s%s' % (self.path,'ukb_icd10s.parquet'))
		df=self.split_disease_dfs(df)
		mask=((df['dis_date']-pd.to_datetime(df['date_of_attending_assessment_centre_f53_0_0'])).dt.days/365.25>2)
		df_dis_aft=df.loc[mask,]
		df_dis_bef=df.loc[~mask,]

		mask=(df_dis_aft['disease'].str.contains(icd10s,regex=True))
		dis_list_aft=list(df_dis_aft.loc[mask,'eid'].unique())

		mask=(df_dis_bef['disease'].str.contains(icd10s,regex=True))
		dis_list_already=list(df_dis_bef.loc[mask,'eid'].astype(str).unique())

		dis_list_out=[str(c) for c in dis_list_aft if c not in dis_list_already]
		
		return dis_list_out,dis_list_already

	def get_entire_data(self,df=None,import_parquet=True,infile='ukb_gt50perc.parquet',outfile='ukb_df_processed.parquet',gen=True,
		min_dis_recs=5000):

		if df is None:
			if import_parquet:
				df=pd.read_parquet(self.path+infile)
			else:
				df=self.get_raw_data(out=False)


		print("data imported")
		df['eid']=df['eid'].astype(str)

		#get columns with words specified in self.cols_needed
		df_oth=self.get_other_cols()
		print("specific data fields done",str(df_oth.shape))

		df=pd.merge(df,df_oth,on='eid',how='left')
		print("merged to other columns")

		df,cols_rem=self.remove_cols(df)
		#print(cols_rem)

		df_cts=self.get_cts_cols(df=df,out=False)
		print("cts data done",str(df_cts.shape))

		#extract continuous columns as these will not be eligible for one hot encoding step
		cts_cols=[c for c in df_cts.columns if c!='eid']

		df_ord,ordinal_cols_unmapped,col_mapping=self.map_cols(df)
		print("ordinal done",str(df_ord.shape))

		ord_cols=[c for c in df_ord.columns if c!='eid']

		df_ohe=self.ohe_cols(df=df,cts_cols=cts_cols,ordcols=ord_cols)
		print("ohe done",str(df_ohe.shape))

		df_par_dis=self.parental_diseases()
		print("df_par_dis done",str(df_par_dis.shape))

		

		#save memory
		del df

		print("df deleted")

		df_treat=self.get_treatment_data()
		print("treatment data done",str(df_treat.shape))

		df_dis,df_ohe_aft=self.get_diseases_all(min_recs=min_dis_recs)
		print("diseases data done",str(df_dis.shape))

		for df1 in [df_cts,df_ohe,df_ord,df_dis,df_treat,df_ohe_aft,df_par_dis]:
			df1['eid']=df1['eid'].astype(str)

		df=pd.merge(df_cts,df_ohe,on='eid',how='left')
		df=pd.merge(df,df_ord,on='eid',how='left')
		df=pd.merge(df,df_dis,on='eid',how='left')
		df=pd.merge(df,df_treat,on='eid',how='left')
		df=pd.merge(df,df_par_dis,on='eid',how='left')


		#ensure we are merging with APOE4 only

		if gen:
			df_gen=self.get_genetics()
			print("genetics returned",str(df_gen.shape))
			df_gen['eid']=df_gen['eid'].astype(str)
			df=pd.merge(df,df_gen,on='eid',how='inner')

		#in case new columns not picked up in cts and ordinal steps
		df_oth=df_oth[[c for c in df_oth.columns if c not in df.columns or c=='eid']]
		df=pd.merge(df,df_oth,on='eid',how='left')
		print("merge complete")

		bef_stud=df.shape[1]
		print(bef_stud)

		df=self.studyvars_add(df)
		aft_stud=df.shape[1]
		print(aft_stud)
		print('new study variables ',str(aft_stud-bef_stud))


		print("study variables added")

		df.to_parquet(self.path+'ukb_df_processed'+str(self.run_date)+'.parquet')

		outs=[df,df_ohe_aft,ordinal_cols_unmapped,col_mapping]


		return outs

	def create_model_data(self,df=None,import_parquet=True,depvar='AD',icd10s=['G30'],infile='ukb_df_processed.parquet',
		nonull_var='date_of_all_cause_dementia_report_f42018_0_0'):

		if df is None:
			if import_parquet:
				df=pd.read_parquet(self.path+infile)
			else:
				df=get_entire_data(self,outfile='ukb_gt50perc.parquet',df=None,import_parquet=True)[0]


		dis_list_out,dis_list_already=self.dis_list(df=None,icd10s=icd10s)

		#exclude those who already had the disease at baseline or died and were in the control group 
		#or have a dementia related illness diagnosed and are in the control group
		cols=self.findcols(df,nonull_var)
		mask_exc=(((df['death']==1)|(df[cols].count(axis=1)>0))&~(df['eid'].isin(dis_list_out)))|\
(df['eid'].isin(dis_list_already))

		df=df.loc[~mask_exc,]
		df[depvar]=0
		mask=(df['eid'].isin(dis_list_out))
		df.loc[mask,depvar]=1

		#any column with a value for dementia/ parkinsons here
		
		#
		return df











	








