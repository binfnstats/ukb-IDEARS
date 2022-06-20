
import pandas as pd
from bed_reader import open_bed, sample_file

chrom=2
bed_file='ukb22418_c2_b0_v2.bed'
bim_file='ukb_snp_chr2_v2.bim'
fam_file='ukb22418_c2_b0_v2_s488244.fam'


all_chroms=list(np.arange(1,24))

for chrom_num in all_chroms:
    files2=[c for c in files if 'c'+str(chrom_num)+'_' in c or 'chr'+str(chrom_num)+'_' in c]
    
    if len(files2)==3:
        bed_file=[f for f in files2 if 'bed' in f][0]
        fam_file=[f for f in files2 if 'fam' in f][0]
        bim_file=[f for f in files2 if 'bim' in f][0]
        bashCommand="sudo ./plink --bed "+bed_file+" --bim "+bim_file+" --fam "+fam_file+\
        " --extract mike_sub/chr"+str(chrom_num)+"_all.txt --make-bed --out mike_sub/all_output_chrom"+str(chrom_num)

        print(chrom_num,bashCommand)



bed = open_bed(bed_file)
val = bed.read()

bim=pd.read_csv(bim_file,sep='\t',header=None)
fam=pd.read_csv(fam_file,sep=' ',header=None)

bim.columns=['miss','rsid','a','var','var1','var2']

df=pd.DataFrame(val,columns=list(bim['rsid']))

#vars=[c for c in df.columns if df.loc[(df[c]==1),c].shape[0]>20000]

print(len(vars))
df_fam=pd.DataFrame(fam)
df_fam.columns=['eid','eid_copy','father_id','mother_id','gender','pheno']

#df_out=df[vars]

df['eid']=df_fam['eid']
df.to_parquet('df_chr'+str(chrom)+'.parquet')