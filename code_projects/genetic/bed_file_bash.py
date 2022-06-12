

import numpy as np

files=['ukb22418_c10_b0_v2.bed', 'ukb22418_c19_b0_v2_s488244.fam', 'ukb_snp_chr8_v2.bim',
       'ukb22418_cXY_b0_v2_s488244.fam', 'ukb22418_cX_b0_v2_s488244.fam', 'ukb22418_c4_b0_v2_s488244.fam',
       'ukb_snp_chr6_v2.bim', 'ukb22418_c13_b0_v2.bed', 'ukb22418_c8_b0_v2_s488244.fam', 'ukb_snp_chr20_v2.bim',
       'ukb_snp_chrMT_v2.bim', '.ukbkey', 'ukb_snp_chr21_v2.bim', 'ukb22418_c16_b0_v2.bed', 
       'ukb22418_c17_b0_v2.bed', 'ukb22418_c14_b0_v2_s488244.fam', 'ukb_snp_chrXY_v2.bim',
       'ukb22418_cX_b0_v2.bed', 'ukb22418_c1_b0_v2_s488244.fam', 'ukb22418_c5_b0_v2_s488244.fam',
       'ukb_snp_chr18_v2.bim', 'ukb22418_cY_b0_v2.bed', 'ukb22418_c19_b0_v2.bed', 'ukb22418_c22_b0_v2.bed',
       'ukb22418_c18_b0_v2_s488244.fam', 'ukb22418_c10_b0_v2_s488244.fam', 'ukb_snp_chr1_v2.bim',
       'ukb_snp_chrX_v2.bim', 'ukb_snp_chrY_v2.bim', 'ukb22418_c5_b0_v2.bed', 
       'ukb_genetic_data_description_v3-1.txt', 'ukb22418_c3_b0_v2.bed', 'ukb22418_c11_b0_v2_s488244.fam',
       'ukb22418_c13_b0_v2_s488244.fam', 'ukb22418_c1_b0_v2.bed', 'ukb22418_cXY_b0_v2.bed',
       'ukb22418_c14_b0_v2.bed', 'ukb22418_c15_b0_v2.bed', 'ukb22418_c2_b0_v2.bed', 'ukb22418_c6_b0_v2.bed',
       'ukb22418_c15_b0_v2_s488244.fam', 'ukb22418_c12_b0_v2_s488244.fam', 'ukb_snp_chr19_v2.bim',
       'ukb22418_cY_b0_v2_s488244.fam', 'ukb_snp_chr10_v2.bim', 'ukb22418_c11_b0_v2.bed', 'ukb_snp_chr11_v2.bim',
       'ukb_snp_chr22_v2.bim', 'ukb22418_c17_b0_v2_s488244.fam', 'ukb_snp_chr17_v2.bim', 
       'ukb22418_c21_b0_v2_s488244.fam', 'ukb_snp_bim.tar', 'ukb22418_c3_b0_v2_s488244.fam', 
       'ukb_snp_chr3_v2.bim', 'ukb_snp.md5', 'ukb_snp_chr9_v2.bim', 'ukb22418_cMT_b0_v2_s488244.fam',
       'ukb22418_c4_b0_v2.bed', 'ukb22418_c21_b0_v2.bed', 'ukb22418_c18_b0_v2.bed', 'ukb_snp_chr12_v2.bim',
       'ukb22418_c9_b0_v2_s488244.fam', 'ukb22418_c20_b0_v2.bed', 'ukb22418_c20_b0_v2_s488244.fam', 
       'ukb22418_c6_b0_v2_s488244.fam', 'ukb22418_c9_b0_v2.bed', 'ukb22418_c2_b0_v2_s488244.fam', 
       'ukb22418_c16_b0_v2_s488244.fam', 'ukb22418_c7_b0_v2_s488244.fam', 'ukb_snp_chr16_v2.bim', 
       'ukb22418_c12_b0_v2.bed', 'ukb1_1615778320_4761.tmp_bulk', 'ukb_snp_chr14_v2.bim',
       'ukb22418_c8_b0_v2.bed', 'ukb22418_c7_b0_v2.bed', 'ukb1_1615766175_22143.tmp_bulk',
       'ukb22418_cMT_b0_v2.bed', 'ukb_snp_chr13_v2.bim', 'ukb_snp_chr2_v2.bim', 'ukb_snp_chr5_v2.bim',
       'ukb_snp_chr15_v2.bim', 'ukb22418_c22_b0_v2_s488244.fam', 'ukb_snp_chr4_v2.bim', 'ukb_snp_chr7_v2.bim']


#section to run bash script and create files which can later be imported
all_chroms=list(np.arange(1,24))

for chrom_num in all_chroms:
    files2=[c for c in files if 'c'+str(chrom_num)+'_' in c or 'chr'+str(chrom_num)+'_' in c]
    
    if len(files2)==3:
        bed_file=[f for f in files2 if 'bed' in f][0]
        fam_file=[f for f in files2 if 'fam' in f][0]
        bim_file=[f for f in files2 if 'bim' in f][0]
        bashCommand="sudo ./plink --bed "+bed_file+" --bim "+bim_file+" --fam "+fam_file+\
        " --extract mike_sub/chr"+str(chrom_num)+"_all.txt --make-bed --out mike_sub/all_output_chrom"+str(chrom_num)

        process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()

        print(chrom_num,bashCommand)