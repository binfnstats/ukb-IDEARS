import pandas as pd
import numpy as np
import random
from pandas.tseries.offsets import DateOffset
from scipy import stats

class data_proc(object):
	def __init__(self):

		self.path="/Users/michaelallwright/Documents/data/ukb/"
		self.snp_file='df_ad_snp_train_2022814.feather'
		self.apoes=['rs429358','rs7412','rs769449','rs405509','rs4420638'] #apoe4 related snps
		self.full_dis_file='/Users/michaelallwright/Documents/data/ukb/dis_full.parquet'

	def case_control(self,df,depvar='AD'):
		"""
		split into case and control

		TODO - remove repeat function in other class
		""" 
		
		mask=(df[depvar]==1)
		df_case=df.loc[mask,]
		df_ctrl=df.loc[~mask,]

		return df_case,df_ctrl

	def ukb_icd10(self):
		df=pd.read_csv(self.path+'metadata/icd10list_ukb.csv')
		df['code']=df['ICD10'].apply(lambda x:x[0:x.find(' ')])
		df['disease']=df['ICD10'].apply(lambda x:x[x.find(' ')+1:len(x)])
		df['code']=df['code'].apply(lambda x:x.replace('.',''))
		df['disease']=df['disease'].apply(lambda x:x.lower())
		df['disease']=df['disease'].apply(lambda x:x.replace('-',''))

		df=df[['code','disease']]

		return df

	def search_icd(self,strings='chronic pain',second_string='',non_strings='xxxxx',string_pat=True):

	   
		icd10_lkup=self.ukb_icd10()

		mask=((icd10_lkup['disease'].str.contains(strings,regex=True))&(icd10_lkup['disease'].str.contains(second_string,regex=True)))&\
(~icd10_lkup['disease'].str.contains(non_strings,regex=True))
		icd10_sub=list(icd10_lkup.loc[mask,'code'])
		icd_df=icd10_lkup.loc[mask,]
		
		if string_pat:
			icd10_sub='|'.join(icd10_sub)
			
		return icd10_sub,icd_df


	def return_eids(self,string='polyneuropathy',icd10s=False,
				string_exc='family|screening|insipidus|pregnancy',years=2):

		# function to take a string and search across the disease list file to find all eids 
		#can search for ICD10 codes specifically or a disease as a string with pipes between each section
		df=pd.read_parquet(self.full_dis_file)

		date_assess='date_of_attending_assessment_centre_f53_0_0'
		disease_date='dis_date'

		df['dis_name_all']=df['disease_name'].apply(lambda x:str(x).lower())


		if icd10s:
			mask_dis=(df['disease'].str.contains(string,regex=True))
		else:
			mask_dis=(df['dis_name_all'].str.contains(string,regex=True))&(~df['dis_name_all'].str.contains(string_exc,regex=True))



		df['dis']=0
		df.loc[mask_dis,'dis']=1


		df_cases=df.loc[mask_dis,]
		mask_exc=~(df['eid'].isin(df_cases['eid']))
		df_ctrls=df.loc[mask_exc,]

	
		cases=pd.DataFrame(df_cases.groupby(['eid']).agg({disease_date:'min',date_assess:'min'})).reset_index()
		ctrls=pd.DataFrame(df_ctrls.groupby(['eid']).agg({disease_date:'min',date_assess:'min','death':'max'})).reset_index()

		cases['eid']=cases['eid'].astype(str)
		ctrls['eid']=ctrls['eid'].astype(str)

		mask_inc_snap=(cases[disease_date]<cases[date_assess])
		mask_inc_pro=(cases[disease_date]>=cases[date_assess]+ DateOffset(years=years))

		cases_inc_pro=list(cases.loc[mask_inc_pro,'eid'].astype(str).unique())
		cases_inc_snap=list(cases.loc[mask_inc_snap,'eid'].astype(str).unique())

		cases_exc_pro=list(cases.loc[~mask_inc_pro,'eid'].astype(str).unique())
		cases_exc_snap=list(cases.loc[~mask_inc_snap,'eid'].astype(str).unique())
		
		mask_death=(ctrls['death']==1)
		ctrls_exc_pro=list(ctrls.loc[mask_death,'eid'].astype(str).unique())
		
		#eids to exclude
		eids_exc_snap=cases_exc_snap
		eids_exc_pro=cases_exc_pro+ctrls_exc_pro
		
		disease_list=pd.DataFrame(df.loc[mask_dis,'dis_name_all'].value_counts())
		df_dict=dict({'eids_inc_snap':cases_inc_snap,'eids_inc_pro':cases_inc_pro,'eids_exc_snap':eids_exc_snap,
			'eids_exc_pro':eids_exc_pro,'disease_list':disease_list,'cases':cases,'ctrl_deaths':ctrls_exc_pro})
	   
	
		return df_dict

	def ctrl_case_ratios(self,df_case,df_ctrl,normvars):

		"""
		determine the ratio of control and case for each normvar denomination
		"""

		cases=pd.DataFrame(df_case.groupby(normvars).size()).reset_index()
		ctrls=pd.DataFrame(df_ctrl.groupby(normvars).size()).reset_index()

		if isinstance(normvars, str):
			normvars=[normvars]

		

		ctrls.columns=normvars+['ctrl_recs']
		cases.columns=normvars+['case_recs']

		ctrl_case=pd.merge(cases,ctrls,on=normvars,how='inner')
		ctrl_case['ratio']=(ctrl_case['ctrl_recs']/ctrl_case['case_recs'])
		
		return ctrl_case


	def varnorm(self,df,normvars,depvar='AD',max_mult=None,delete_df=False):

		"""rebalances dataframe to be equal across case and control as defined by depvar=1/0 across a list of variables which must be present in the data
		#df1=df.copy()
		"""
		df_case,df_ctrl=self.case_control(df,depvar=depvar)

		if delete_df:
			del df

		ctrl_case=self.ctrl_case_ratios(df_case,df_ctrl,normvars)
		
		if max_mult==None:
			max_mult=ctrl_case['ratio'].min()

		ctrl_case['case_samp']=max_mult

		return ctrl_case ,df_ctrl,df_case#,cases

	def varnorm_sample(self,df,normvars,depvar,max_mult=None):
		#
		ctrl_case,df_ctrl,df_case=self.varnorm(df,normvars=normvars,depvar=depvar,delete_df=False)
		ctrl_case['recs_sample']=(ctrl_case['case_recs']*ctrl_case['case_samp']).apply(lambda x:np.floor(x)).astype(int)
		df_ctrl=pd.merge(df_ctrl,ctrl_case[normvars+['recs_sample']],on=normvars)
		df_ctrl=pd.DataFrame(df_ctrl.groupby(normvars).\
		apply(lambda x: x.sample(x['recs_sample'].iat[0]))).reset_index(drop=True)
		
		df_out=pd.concat([df_ctrl,df_case],axis=0)
		return df_out


	def eids_var_to_dict(self,df,normvars=['age_when_attended_assessment_centre_f21003_0_0']):

		"""
		create a dictionary of a normalisation variable and lists of unique eids associated with that variable
		"""

		
		normvar=''.join(normvars)


		normvar_list=[c for c in list(df[normvar].unique())]
		normvar_eids=[list(df.loc[(df[normvar]==c,'eid')]) for c in list(df[normvar].unique())]
		normvar_eid_dict=dict(zip(normvar_list,normvar_eids))

		return normvar_eid_dict

	def normvar_samplesize_dict(self,df,norm_var):

		"""
		determine sample size for each normvar as a dictionary - applies to ctrl_case
		"""
		"""if len(norm_vars)==1:
									df['a_var']=df[norm_vars[0]].astype(str)
								elif len(norm_vars)==2:
									df['a_var']=df[norm_vars[0]].astype(str)+df[norm_vars[1]].astype(str)
						"""
		

		df['sample_size']=(df['ratio'].min()*df['case_recs']).apply(np.floor)
		out_dict=dict(zip(df[norm_var],df['case_recs'].astype(int)))

		return out_dict


	def get_indices(self,list_in,start_pos=1000,length=800):
	
		#returns new list taking ino account length of existing list
	   
		end_pos=start_pos+length
		length_list=len(list_in)
		
		
		if length_list>end_pos:
			#print("yes")
			#end_pos=start_pos+b
			list_out=list_in[start_pos:end_pos]
			start_pos_new=end_pos
			
		else:
			new_len=end_pos-length_list

			list_out1=list_in[start_pos:length_list]
			list_out2=list_in[0:new_len]
			
			list_out=list_out1+list_out2
			start_pos_new=new_len
	 
		return list_out,start_pos_new

	def split_mult_files(self,df,depvar="AD",normvars=['age_when_attended_assessment_centre_f21003_0_0'],iterations=50,mult_fact_max=True,
		multfact=1):

		#purpose to take dataframe and create iterative Monte Carlo based datasets which use all the case data and selectively spread
		#across the control space so all controls which are variable normalised get used

		#convert normvars to one sngle string
		normvar=''.join(normvars)

		df1=df.copy()

		#concatenate values for each norm var
		for i in range(len(normvars)):
			if i==0:
				df1[normvar]=df1[normvars[i]].astype(str)
			else:
				df1[normvar]=df1[normvar]+df1[normvars[i]].astype(str)


		ctrl_case,df_ctrl,df_case=self.varnorm(df1,normvar,depvar=depvar,max_mult=None,delete_df=False)

		case_eids=list(df_case['eid'])

		normvar_eid_dict=self.eids_var_to_dict(df=df_ctrl,normvars=normvars)

		normvar_sample_size_dict=self.normvar_samplesize_dict(df=ctrl_case,norm_var=normvar)

		#reset the dictionaries to ensure the elements are present in both
		normvar_eid_dict=\
	dict(zip([a for a in normvar_eid_dict if a in normvar_sample_size_dict],\
	[normvar_eid_dict[a] for a in normvar_eid_dict if a in normvar_sample_size_dict]))

		normvar_sample_size_dict=\
	dict(zip([a for a in normvar_sample_size_dict if a in normvar_eid_dict],\
	[normvar_sample_size_dict[a] for a in normvar_sample_size_dict if a in normvar_eid_dict]))



		mult_fact_max_val=int(min([np.floor(len(normvar_eid_dict[a])/normvar_sample_size_dict[a]) for a in normvar_eid_dict]))

		if mult_fact_max is True:
			multfact=mult_fact_max_val


		#max_mult

		#print(normvar_eid_dict)

		eids_all=[]

		map_dict=dict()
		start_pos_dict=dict()
		
		for i in range(iterations):


			eids_iter=[]

			for a in normvar_eid_dict:

				
				if i==0:
					list_out,start_pos_new=self.get_indices(normvar_eid_dict[a],start_pos=0,
						length=normvar_sample_size_dict[a]*multfact)
					start_pos_dict[a]=start_pos_new
					#eids_normvars.append(list_out)
					
				else:
					list_out,start_pos_new=self.get_indices(normvar_eid_dict[a],start_pos=start_pos_dict[a],
						length=normvar_sample_size_dict[a]*multfact)
					start_pos_dict[a]=start_pos_new

				eids_iter=eids_iter+list_out

			map_dict[i]=eids_iter+case_eids




		return map_dict

	def varmap(self):
		varmap = {}
		with open(self.path+"metadata/varmap.txt") as myfile:
			for line in myfile:
				name, var = line.partition("=")[::2]
				name=name.strip()
				var=var.strip()
				varmap[name] = var

		self.variable_map=varmap
		return varmap

	def map_var(self,df,var_):
		df['var_mapped']=df[var_].map(self.varmap())
		mask=pd.notnull(df['var_mapped'])
		df.loc[mask,var_]=df.loc[mask,'var_mapped']
		df.drop(columns='var_mapped',inplace=True)
		return df


	def ttest(self,df,var,depvar='polyneuropathy'):
	
		df1=df.loc[pd.notnull(df[var]),[var,depvar]]
		ttest_vals=stats.ttest_ind(df1[(df1[depvar]==1)][var],df1[(df1[depvar]==0)][var])

		return ttest_vals

	def runplots_static(self,df,depvar='poly_chronic',
		fig_name='diabetes_inflamm_polychronicpain',perc=True,compvars=None,agenormvars=[],savefig=True,pltshow=True,
		splitvar='sex_f31_0_0',labels=dict({1:'Female',0:'Male'}),
		normvars=['age_when_attended_assessment_centre_f21003_0_0','sex_f31_0_0'],name='Diabetes'):

		df=df.copy()

		k=len(compvars)

		
		splitvars=list(set(list(df.loc[pd.notnull(df[splitvar]),splitvar].unique())))

	

		compvars_use=[]
		pvals=[]
		genders=[]
		vals_case=[]
		vals_std_case=[]
		vals_ctrl=[]
		vals_std_ctrl=[]

		vals_mean_val_case=[]
		vals_mean_val_ctrl=[]
		vals_std_val_case=[]
		vals_std_val_ctrl=[]



		for j,v in enumerate(compvars):
			for i,x in enumerate(splitvars):
				

				df_diab2_use=df.loc[df[splitvar]==x,]

				pval=str(round(list(self.ttest(df_diab2_use,v,depvar))[1],7))
				#rangevars=df_diab2_use[v].quantile(0.75)-df_diab2_use[v].quantile(0.25)

				mask=(df_diab2_use[depvar]==1)
				mean_val_case=df_diab2_use.loc[mask,v].mean()
				std_case=df_diab2_use.loc[mask,v].std()
				vals_std_val_case.append(std_case)
				std_case=str(round(mean_val_case,2))+' +/- '+str(round(std_case,2))
				mean_val_ctrl=df_diab2_use.loc[~mask,v].mean()
				std_ctrl=df_diab2_use.loc[~mask,v].std()
				vals_std_val_ctrl.append(std_ctrl)
				std_ctrl=str(round(mean_val_ctrl,2))+' +/- '+str(round(std_ctrl,2))



				compvars_use.append(v)
				genders.append(i)
				pvals.append(pval)
				vals_case.append(mean_val_case)
				vals_ctrl.append(mean_val_ctrl)

				vals_std_case.append(std_case)
				vals_std_ctrl.append(std_ctrl)

				vals_mean_val_case.append(mean_val_case)
				vals_mean_val_ctrl.append(mean_val_ctrl)
				
				

		if len(splitvars)>1:
			genders=['Male' if c==0 else 'Female' for c in genders]
		else:
			genders=['All' for c in genders]


		compvars_use=[self.varmap()[c] if c in self.varmap() else c for c in compvars_use]


		df_out=pd.DataFrame({'Variable':compvars_use,splitvar:genders,'case value':vals_case,'case_vals_std':vals_std_case,
			'control value':vals_ctrl,'ctrl_vals_std':vals_std_ctrl,'p-value':pvals,'vals_mean_val_case':vals_mean_val_case,
			'vals_mean_val_ctrl':vals_mean_val_ctrl,
			'vals_std_val_case':vals_std_val_case,
			'vals_std_val_ctrl':vals_std_val_ctrl})


		df_out=df_out.pivot(index='Variable',columns=splitvar,values=['ctrl_vals_std','case_vals_std','vals_mean_val_case',
			'vals_mean_val_ctrl','vals_std_val_case','vals_std_val_ctrl','p-value']).reset_index()

		df_out.columns=[c[0]+'_'+c[1] for c in df_out.columns]

		df_out.insert(loc=0, column='Experiment', value=name)

		#df_out['Experiment']=name
	   
		return df_out

	
