path='../../../data/ukb/ad/'

class snp_process(object):
	def __init__(self):

		self.path='../../../data/ukb/ad/'
		self.snp_file='df_ad_snp_train_2022814.feather'
		self.apoes=['rs429358','rs7412','rs769449','rs405509','rs4420638'] #apoe4 related snps

	def import_df(self):
		"""
		import full snp file
		"""
		df=pd.read_feather(self.path+self.snp_file)
		return df

	def case_control(self,df,depvar='AD'):
		"""
		split into case and control
		""" 
		
		mask=(df[depvar]==1)
		df_case=df.loc[mask,]
		df_ctrl=df.loc[~mask,]
		return df_case,df_ctrl

	def drop_apoe_missing(self,df_case,df_ctrl):
		"""
		drop all participants that don't have complete apoe info
		""" 

		case_eids=list(df_case[self.apoes+['eid']].dropna()['eid'])
		ctrl_eids=list(df_ctrl[self.apoes+['eid']].dropna()['eid'])
		mask=(df_case['eid'].isin(case_eids))
		df_case=df_case.loc[mask,]
		mask=(df_ctrl['eid'].isin(ctrl_eids))
		df_ctrl=df_ctrl.loc[mask,]

		return df_case,df_ctrl

	def exc_cols_prop_case(self,df_case,df_ctrl,perc=0.95):
		"""
		remove all columns which have less than perc of cases as non nulls
		"""
		cols_exc=[c for c in df_case.columns if df_case[c].count()<df_case.shape[0]*perc]
		df_case.drop(columns=cols_exc,inplace=True)
		df_ctrl.drop(columns=cols_exc,inplace=True)
		return df_case,df_ctrl

	def drop_part_less_prop(self,df,perc=0.98):
		mask=(pd.to_numeric(df.count(axis=1)/df.shape[1])>=perc)
		df=df.loc[mask,].reset_index()
		return df


	def null_optimize(self,df=None):
		
		df=self.import_df()

		df_case,df_ctrl=self.case_control(df,depvar='AD')

		del df


		df_case,df_ctrl=self.exc_cols_prop_case(df_case,df_ctrl,perc=0.95)
		df_ctrl=self.drop_part_less_prop(df_ctrl,perc=0.98)
		
		#output individual files
		df_ctrl.to_feather(path+'df_ad_snp_2022905_ctrl.feather')
		df_case.reset_index().to_feather(path+'df_ad_snp_2022905_case.feather')

		#put back together again
		df=pd.concat([df_ctrl,df_case],axis=0)

		#delete to free memory
		del df_ctrl
		del df_case


		
