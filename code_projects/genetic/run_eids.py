# file takes a set of eids and extracts all genotypes for that set

import pandas as pd
from bed_reader import open_bed, sample_file

run='PD_ctrl'

eids=[str(c[0]) for c in pd.read_csv('PD_eids.txt').values]


all_chroms=list(np.arange(1,24))

for chrom_num in all_chroms:
	bed_file='all_output_chrom'+str(chrom_num)+'.bed'
	bim_file='all_output_chrom'+str(chrom_num)+'.bim'
	fam_file='all_output_chrom'+str(chrom_num)+'.fam'

	bed = open_bed(bed_file)
	val = bed.read()

	bim=pd.read_csv(bim_file,sep='\t',header=None)
	fam=pd.read_csv(fam_file,sep=' ',header=None)

	bim.columns=['miss','rsid','a','var','var1','var2']

	df=pd.DataFrame(val,columns=list(bim['rsid']))

	df=df.loc[(df['eid'].isin(eids)),]
	print('size of df',df.shape)

#vars=[c for c in df.columns if df.loc[(df[c]==1),c].shape[0]>20000]

	df_fam=pd.DataFrame(fam)
	df_fam.columns=['eid','eid_copy','father_id','mother_id','gender','pheno']

	#df_out=df[vars]

	df['eid']=df_fam['eid']
	df.to_parquet('df_chr'+str(run)+'_chrom_'+str(chrom)+'.parquet')